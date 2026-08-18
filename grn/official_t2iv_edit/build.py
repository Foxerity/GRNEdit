from functools import partial

import torch
from timm.models import create_model

import grn.utils_t2iv.dist as dist
from grn.models.ema import get_ema_model
from grn.models.hbq_tokenizer import HBQ_Tokenizer
from grn.official_t2iv_edit.model import MultipleLayers  # noqa: F401 - imported for FSDP policy and model registration
from grn.utils_t2iv.lr_control import filter_params


def load_visual_tokenizer(args, device=None):
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    vae = HBQ_Tokenizer(args=args, latent_channels=args.detail_scale_dim, encoder_out_type='feature_tanh')
    vae.eval()
    vae = vae.to(device)
    for param in vae.parameters():
        param.requires_grad = False
    state_dict = torch.load(args.vae_path, map_location=device, weights_only=False)
    if 'ema' in state_dict:
        print('Load ema vae weights')
        state_dict = state_dict['ema']
    else:
        print('Load non ema vae weights')
        state_dict = state_dict['vae']
    print('Load vae: ', vae.load_state_dict(state_dict, assign=True))
    return vae


def build_vae_gpt(args, device='cuda'):
    vae_local = load_visual_tokenizer(args, device)
    gpt_kw = dict(
        pretrained=False, global_pool='',
        text_channels=args.Ct5, text_maxlen=args.tlen,
        norm_eps=args.norm_eps,
        top_p=args.tp, top_k=args.tk, tau=args.tau,
        checkpointing=args.enable_checkpointing,
        pad_to_multiplier=args.pad_to_multiplier,
        use_flex_attn=args.use_flex_attn,
        num_of_label_value=args.num_of_label_value,
        pn=args.pn,
        train_h_div_w_list=None,
        apply_spatial_patchify=args.apply_spatial_patchify,
        dynamic_scale_schedule=args.dynamic_scale_schedule,
        video_frames=args.video_frames,
        other_args=args,
    )
    print(f'[create official edit gpt_wo_ddp] constructor kw={gpt_kw}\n')
    gpt_kw['vae_local'] = vae_local
    gpt_wo_ddp = create_model(args.model, **gpt_kw)
    assert all(p.requires_grad for _, p in gpt_wo_ddp.named_parameters())
    return vae_local, gpt_wo_ddp


def _load_rush_resume(args, gpt_wo_ddp, gpt_wo_ddp_ema):
    if not args.rush_resume:
        return
    print(f'{args.rush_resume=}')
    if not str(args.rush_resume).lower().endswith('.pth'):
        raise ValueError(f'GRNEdit warm-start requires a .pth checkpoint, got {args.rush_resume!r}.')
    cpu_d = torch.load(args.rush_resume, map_location='cpu', weights_only=False)
    if 'trainer' in cpu_d:
        state_dict = cpu_d['trainer']['gpt_fsdp']
        ema_state_dict = cpu_d['trainer'].get('gpt_ema_fsdp', state_dict)
    else:
        state_dict = cpu_d
        ema_state_dict = state_dict

    load_result = gpt_wo_ddp.load_state_dict(dict(state_dict), strict=False)
    required_stage1 = set(gpt_wo_ddp._stage15_required_keys())
    if set(load_result.missing_keys) not in (set(), required_stage1) or load_result.unexpected_keys:
        raise RuntimeError(
            'The Stage I warm-start must be a complete GRN backbone or complete Stage I model: '
            f'missing={sorted(load_result.missing_keys)}, unexpected={sorted(load_result.unexpected_keys)}.'
        )
    print(load_result)
    if gpt_wo_ddp_ema is not None:
        ema_result = gpt_wo_ddp_ema.load_state_dict(dict(ema_state_dict), strict=False)
        if set(ema_result.missing_keys) not in (set(), required_stage1) or ema_result.unexpected_keys:
            raise RuntimeError(
                'The Stage I EMA warm-start must be a complete GRN backbone or complete Stage I model: '
                f'missing={sorted(ema_result.missing_keys)}, unexpected={sorted(ema_result.unexpected_keys)}.'
            )


def _require_existing_transformer_checkpoint(args):
    if not getattr(args, 'rush_resume', '') and not getattr(args, 'resume_checkpoint', ''):
        raise RuntimeError(
            'Official edit training requires an existing transformer checkpoint. '
            'Set RUSH_RESUME/--rush_resume for official or edit weight warm-start, '
            'or RESUME_CHECKPOINT/--resume_checkpoint for strict training resume. '
            'Training the GRN main transformer from random initialization is disabled.'
        )


def build_model_optimizer(args):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    _require_existing_transformer_checkpoint(args)
    if int(args.zero) != 3:
        raise ValueError('GRNEdit training supports only FSDP FULL_SHARD (--zero 3).')
    vae_local, gpt_wo_ddp = build_vae_gpt(args, device=args.model_init_device)
    count_p = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    num_para = count_p(gpt_wo_ddp)
    if num_para / 1000 < 20:
        gpt_wo_ddp = gpt_wo_ddp.to('cuda')

    print(
        '[Official Edit PT] skip custom full-model init_weights; '
        'main weights must come from rush_resume or resume_checkpoint.',
        flush=True,
    )
    gpt_wo_ddp_ema = get_ema_model(gpt_wo_ddp) if args.use_fsdp_model_ema else None
    _load_rush_resume(args, gpt_wo_ddp, gpt_wo_ddp_ema)
    if hasattr(gpt_wo_ddp, 'freeze_disabled_source_modules'):
        gpt_wo_ddp.freeze_disabled_source_modules()
        if gpt_wo_ddp_ema is not None:
            gpt_wo_ddp_ema.freeze_disabled_source_modules()

    ndim_dict = {name: para.ndim for name, para in gpt_wo_ddp.named_parameters() if para.requires_grad}

    print(f'[Official Edit PT][#para], GPT={num_para:.2f}M parameters\n\n')

    gpt_uncompiled = gpt_wo_ddp
    gpt_wo_ddp = args.compile_model(gpt_wo_ddp, args.tfast)

    gpt_ddp_ema = None
    if args.fsdp_warp_mode == 'full':
        def my_policy(module: torch.nn.Module, recurse: bool, **kwargs) -> bool:
            return True
        auto_wrap_policy = my_policy
    elif args.fsdp_warp_mode == 'trans_block':
        auto_wrap_policy = ModuleWrapPolicy([MultipleLayers])
    else:
        raise ValueError(f'Unsupported FSDP wrap mode: {args.fsdp_warp_mode!r}.')

    if args.enable_hybrid_shard == 1:
        world_size = dist.get_world_size()
        if not (1 < args.inner_shard_degree <= world_size and world_size % args.inner_shard_degree == 0):
            raise ValueError(
                f'INNER_SHARD_DEGREE must divide world size and lie in [2, {world_size}], '
                f'got {args.inner_shard_degree}.'
            )
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
        device_mesh = init_device_mesh(
            'cuda', (world_size // args.inner_shard_degree, args.inner_shard_degree)
        )
    elif args.enable_hybrid_shard == 0:
        sharding_strategy = ShardingStrategy.FULL_SHARD
        device_mesh = None
    else:
        raise ValueError('ENABLE_HYBRID_SHARD must be 0 or 1 in the released training path.')

    if args.fsdp_init_device == 'cpu':
        gpt_wo_ddp = gpt_wo_ddp.cpu()
    gpt_ddp = FSDP(
        gpt_wo_ddp,
        device_id=dist.get_local_rank(),
        sharding_strategy=sharding_strategy,
        mixed_precision=None,
        auto_wrap_policy=auto_wrap_policy,
        use_orig_params=True,
        sync_module_states=True,
        limit_all_gathers=True,
        device_mesh=device_mesh,
    ).to(args.device)
    if args.use_fsdp_model_ema:
        gpt_wo_ddp_ema = gpt_wo_ddp_ema.to(args.device)
        gpt_ddp_ema = FSDP(
            gpt_wo_ddp_ema,
            device_id=dist.get_local_rank(),
            sharding_strategy=sharding_strategy,
            mixed_precision=None,
            auto_wrap_policy=auto_wrap_policy,
            use_orig_params=True,
            sync_module_states=True,
            limit_all_gathers=True,
            device_mesh=device_mesh,
        )
    torch.cuda.synchronize()

    nowd_keys = {
        'cls_token', 'start_token', 'task_token',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
        'text_proj_for_sos.ca.mat_q',
        'scale_tokens', 'class_tokens',
    }
    names, paras, para_groups = filter_params(
        gpt_ddp,
        ndim_dict,
        nowd_keys=nowd_keys,
        allow_frozen_params=True,
    )
    del ndim_dict
    opt_clz = partial(torch.optim.AdamW, betas=(0.9, 0.999), fused=True)
    opt_kw = dict(lr=args.tlr, weight_decay=args.twd)
    print(f'[vgpt] optim={opt_clz}, opt_kw={opt_kw}\n')
    gpt_optim = opt_clz(params=para_groups, **opt_kw)
    del names, paras, para_groups
    return vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim
