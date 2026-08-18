import numpy as np
import torch

from grn.official_t2iv_edit.global_refine import (
    flatten_two_level_list,
    get_scale_pack_info,  # noqa: F401
    get_visual_rope_embeds,
    shift_pt,
    video_decode,  # noqa: F401
)
from grn.utils_t2iv.hbq_util_t2iv import multiclass_labels2onehot_input


def video_stage15_source_conditioned_encode(
    vae,
    inp_B3HW,
    vae_features=None,
    source_vae_features=None,
    device='cuda',
    args=None,
    infer_mode=False,
    rope2d_freqs_grid=None,
    dynamic_resolution_h_w=None,
    tokens_remain=9999999,
    text_lens=[],
    caption_nums=[],
    rank_vary_generator=None,
    vis_verbose=False,
    meta_list=None,
    identity_mask=None,
    **kwargs,
):
    if args is None:
        raise ValueError('args is required for Stage1.5 source-conditioned edit schedule.')
    if getattr(args, 'refine_mode', '') != 'ar_discrete_GRN_bit':
        raise ValueError(
            'Stage1.5 source-conditioned edit schedule only supports --refine_mode ar_discrete_GRN_bit, '
            f'got {getattr(args, "refine_mode", None)!r}.'
        )

    if rank_vary_generator is not None:
        numpy_generator = rank_vary_generator['numpy_generator']
        torch_cuda_generator = rank_vary_generator['torch_cuda_generator']
    else:
        numpy_generator = np.random.default_rng()
        torch_cuda_generator = torch.Generator(device='cuda')

    if vae_features is None:
        raw_features, _, _ = vae.encode_for_raw_features(inp_B3HW, scale_schedule=None, slice=True)
        raw_features_list = [raw_features]
        x_recon_raw = vae.decode(raw_features[-1], slice=True)
        x_recon_raw = torch.clamp(x_recon_raw, min=-1, max=1)
        print(f'raw_features[-1].shape: {raw_features[-1].shape}')
    else:
        raw_features_list = vae_features

    if source_vae_features is None:
        raise ValueError('source_vae_features is required for Stage1.5 source-conditioned edit schedule.')
    source_raw_features_list = source_vae_features
    if len(source_raw_features_list) != len(raw_features_list):
        raise ValueError(
            'source/target feature list length mismatch: '
            f'source={len(source_raw_features_list)}, target={len(raw_features_list)}'
        )
    if identity_mask is None:
        identity_mask = [False] * len(raw_features_list)
    if len(identity_mask) != len(raw_features_list):
        raise ValueError(f'identity_mask length mismatch: {len(identity_mask)} != {len(raw_features_list)}')

    gt_all_bit_indices = []
    pred_all_bit_indices = []
    var_input_list = []
    source_input_list = []
    sequece_packing_scales = []
    h_div_w_template_list = np.array(list(dynamic_resolution_h_w.keys()))
    visual_rope_cache_list = []
    other_info_by_scale = []
    scale_lengths = []
    with torch.amp.autocast('cuda', enabled=False):
        for example_ind, raw_features in enumerate(raw_features_list):
            source_raw_features = source_raw_features_list[example_ind]
            meta = meta_list[example_ind]
            is_identity = bool(identity_mask[example_ind])

            gt_all_bit_indices.append([])
            pred_all_bit_indices.append([])
            var_input_list.append([])
            source_input_list.append([])
            visual_rope_cache_list.append([])
            other_info_by_scale.append([])

            B, C, T, H, W = raw_features[-1].shape
            h_div_w = H / W
            mapped_h_div_w_template = h_div_w_template_list[np.argmin(np.abs(h_div_w - h_div_w_template_list))]
            pn = meta['pn']
            if meta['first_frame_condition']:
                scale_schedule = dynamic_resolution_h_w[mapped_h_div_w_template][pn]['pt2scale_schedule'][T - 1]
            else:
                scale_schedule = dynamic_resolution_h_w[mapped_h_div_w_template][pn]['pt2scale_schedule'][T]
            if not infer_mode:
                next_tokens_remain = tokens_remain - T * H * W - args.add_scale_token - text_lens[example_ind]
                if next_tokens_remain < 0:
                    break
                tokens_remain = next_tokens_remain
                scale_lengths.append(T * H * W + text_lens[example_ind] + args.add_scale_token)
            preserve_scale_schedule = [scale_schedule[0]]

            target = raw_features[0]
            source_target = source_raw_features[0]
            if tuple(source_target.shape) != tuple(target.shape):
                raise ValueError(
                    'source/target raw feature shape mismatch at schedule encode: '
                    f'source={tuple(source_target.shape)}, target={tuple(target.shape)}'
                )

            if not infer_mode and args.log_norm_sigma > 0:
                spt = torch.sigmoid(
                    torch.randn(1, generator=torch_cuda_generator, device=target.device)
                    * args.log_norm_sigma
                    + args.log_norm_mean
                ).item()
                spt = shift_pt(spt, args.alpha)
            else:
                spt = shift_pt(numpy_generator.random(), args.alpha)

            from grn.utils_t2iv.hbq_util_t2iv import raw_feature2bit_label

            target_labels = raw_feature2bit_label(target, hbq_round=args.hbq_round)
            source_labels = raw_feature2bit_label(source_target, hbq_round=args.hbq_round)
            train_labels = source_labels if is_identity else target_labels
            classes = 2

            random_labels = torch.randint(
                0,
                classes,
                size=train_labels.shape,
                generator=torch_cuda_generator,
                device=train_labels.device,
                dtype=train_labels.dtype,
            )
            random_mask = torch.rand(
                size=train_labels.shape,
                generator=torch_cuda_generator,
                device=train_labels.device,
                dtype=target.dtype,
            ) < spt
            mixed_xt = torch.where(random_mask, train_labels, random_labels)
            precise_spt = random_mask.float().mean()
            wandb_plot_index = min(9, int(precise_spt / 0.1))

            if not infer_mode:
                if meta['first_frame_condition']:
                    visual_rope_cache_list[-1].append(
                        get_visual_rope_embeds(
                            rope2d_freqs_grid,
                            scale_schedule[0],
                            device,
                            mapped_h_div_w_template,
                            t_offset=1,
                        )
                    )
                    visual_rope_cache_list[-1].append(
                        get_visual_rope_embeds(
                            rope2d_freqs_grid,
                            (1, H, W),
                            device,
                            mapped_h_div_w_template,
                            t_offset=0,
                        )
                    )
                    visual_rope_cache_list[-1] = [torch.cat(visual_rope_cache_list[-1], dim=-2)]
                else:
                    visual_rope_cache_list[-1].append(
                        get_visual_rope_embeds(
                            rope2d_freqs_grid,
                            scale_schedule[0],
                            device,
                            mapped_h_div_w_template,
                            t_offset=0,
                        )
                    )

            visual_token_dim = mixed_xt.shape[1] * classes

            if not infer_mode and meta['first_frame_condition']:
                first_frame_labels = train_labels[:, :, :1]
                first_frame_tokens = (
                    multiclass_labels2onehot_input(first_frame_labels, classes)
                    .reshape(1, visual_token_dim, -1)
                    .permute(0, 2, 1)
                )
                cur_visual_tokens = (
                    multiclass_labels2onehot_input(mixed_xt[:, :, 1:], classes)
                    .reshape(1, visual_token_dim, -1)
                    .permute(0, 2, 1)
                )
                cur_visual_tokens = torch.cat((cur_visual_tokens, first_frame_tokens), dim=1)
                source_first_frame_tokens = (
                    multiclass_labels2onehot_input(source_labels[:, :, :1], classes)
                    .reshape(1, visual_token_dim, -1)
                    .permute(0, 2, 1)
                )
                source_visual_tokens = (
                    multiclass_labels2onehot_input(source_labels[:, :, 1:], classes)
                    .reshape(1, visual_token_dim, -1)
                    .permute(0, 2, 1)
                )
                source_visual_tokens = torch.cat((source_visual_tokens, source_first_frame_tokens), dim=1)
                indices = train_labels[:, :, 1:]
            else:
                cur_visual_tokens = (
                    multiclass_labels2onehot_input(mixed_xt, classes)
                    .reshape(1, visual_token_dim, -1)
                    .permute(0, 2, 1)
                )
                source_visual_tokens = (
                    multiclass_labels2onehot_input(source_labels, classes)
                    .reshape(1, visual_token_dim, -1)
                    .permute(0, 2, 1)
                )
                indices = train_labels

            indices = indices.type(torch.long).permute(0, 2, 3, 4, 1)
            gt_all_bit_indices[-1].append(indices)
            var_input_list[-1].append(cur_visual_tokens)
            source_input_list[-1].append(source_visual_tokens)
            other_info_by_scale[-1].append(
                {
                    'largest_scale': scale_schedule[-1],
                    'wandb_plot_index': wandb_plot_index,
                    'cur_bits': indices.shape[-1],
                    'cur_lvl': args.detail_num_lvl,
                    'scale_token_id': precise_spt,
                    'predict_tokens': np.prod(scale_schedule[0]),
                    'all_tokens': scale_lengths[-1] if len(scale_lengths) else -1,
                    'first_frame_condition': meta['first_frame_condition'],
                }
            )
            sequece_packing_scales.append(preserve_scale_schedule)

    gt_all_bit_indices = flatten_two_level_list(gt_all_bit_indices)
    pred_all_bit_indices = flatten_two_level_list(pred_all_bit_indices)
    var_input_list = flatten_two_level_list(var_input_list)
    source_input_list = flatten_two_level_list(source_input_list)
    visual_rope_cache_list = flatten_two_level_list(visual_rope_cache_list)
    other_info_by_scale = flatten_two_level_list(other_info_by_scale)

    if infer_mode:
        return [train_labels, target], x_recon_raw, [target], None, None, None

    gt_ms_idx_Bl = []
    for item in gt_all_bit_indices:
        _, tt, hh, ww, dd = item.shape
        item = item.reshape(B, tt * hh * ww, dd)
        gt_ms_idx_Bl.append(item)
    gt_BLC = gt_ms_idx_Bl
    x_BLC = var_input_list
    source_x_BLC = source_input_list
    x_BLC_mask = None
    scale_or_time_ids = None
    return (
        x_BLC,
        source_x_BLC,
        x_BLC_mask,
        scale_or_time_ids,
        gt_BLC,
        pred_all_bit_indices,
        visual_rope_cache_list,
        sequece_packing_scales,
        scale_lengths,
        other_info_by_scale,
    )
