import math
from pprint import pformat
from typing import Tuple, List, Dict, Union

import torch.nn
import grn.utils_t2iv.dist as dist


def filter_params(model, ndim_dict, nowd_keys=(), lr_scale=0.0, allow_frozen_params=False, include_frozen_params=False) -> Tuple[
    List[str], List[torch.nn.Parameter], List[Dict[str, Union[torch.nn.Parameter, float]]]
]:
    """Build optimizer groups.

    include_frozen_params keeps temporarily frozen parameters in the optimizer
    so a staged warmup can later restore them without rebuilding optimizer state.
    """
    with_lr_scale = hasattr(model, 'get_layer_id_and_scale_exp') and 0 < lr_scale <= 1
    print(f'[get_param_groups][lr decay] with_lr_scale={with_lr_scale}, lr_scale={lr_scale}')
    para_groups, para_groups_dbg = {}, {}
    names, paras = [], []
    names_no_grad = []
    count, numel = 0, 0
    for name, para in model.named_parameters():
        name_parts = name.split('.')
        while name_parts and name_parts[0] in {'module', '_orig_mod', '_fsdp_wrapped_module'}:
            name_parts.pop(0)
        name = '.'.join(
            part for part in name_parts if part not in {'_orig_mod', '_fsdp_wrapped_module'}
        )
        if not name:
            raise RuntimeError('Optimizer parameter name became empty after wrapper normalization.')
        if not para.requires_grad:
            names_no_grad.append(name)
            if not include_frozen_params:
                continue  # frozen weights
        count += 1
        numel += para.numel()
        names.append(name)
        paras.append(para)

        if ndim_dict.get(name, 2) == 1 or name.endswith('bias') or any(k in name for k in nowd_keys):
            cur_wd_sc, group_name = 0., 'ND'
        else:
            cur_wd_sc, group_name = 1., 'D'

        if with_lr_scale:
            layer_id, scale_exp = model.get_layer_id_and_scale_exp(name)
            group_name = f'layer{layer_id}_' + group_name
            cur_lr_sc = lr_scale ** scale_exp
            dbg = f'[layer {layer_id}][sc = {lr_scale} ** {scale_exp}]'
        else:
            cur_lr_sc = 1.
            dbg = f'[no scale]'

        if group_name not in para_groups:
            para_groups[group_name] = {'params': [], 'wd_sc': cur_wd_sc, 'lr_sc': cur_lr_sc}
            para_groups_dbg[group_name] = {'params': [], 'wd_sc': cur_wd_sc, 'lr_sc': dbg}
        para_groups[group_name]['params'].append(para)
        para_groups_dbg[group_name]['params'].append(name)

    for g in para_groups_dbg.values():
        g['params'] = pformat(', '.join(g['params']), width=200)

    for rk in range(dist.get_world_size()):
        dist.barrier()
        if dist.get_rank() == rk:
            print(f'[get_param_groups][rank{dist.get_rank()}] {type(model).__name__=} {count=}, {numel=}', flush=True, force=True)
    print('')

    if names_no_grad and not allow_frozen_params and not include_frozen_params:
        raise AssertionError(f'[get_param_groups] names_no_grad = \n{pformat(names_no_grad, indent=2, width=240)}\n')
    if names_no_grad and include_frozen_params:
        print(
            f'[get_param_groups] include_frozen_params=True, keep {len(names_no_grad)} frozen params in optimizer groups. '
            f'preview={pformat(names_no_grad[:40], indent=2, width=240)}',
            flush=True,
        )
    elif names_no_grad and allow_frozen_params:
        print(
            f'[get_param_groups] allow_frozen_params=True, skip {len(names_no_grad)} frozen params. '
            f'preview={pformat(names_no_grad[:40], indent=2, width=240)}',
            flush=True,
        )
    del ndim_dict
    return names, paras, list(para_groups.values())
