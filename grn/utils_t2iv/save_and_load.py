import gc
import os
import subprocess
import time
from typing import List, Mapping, Optional, Tuple

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from grn.utils_t2iv import arg_util
import grn.utils_t2iv.dist as dist


def _checkpoint_args_state(args, trainer):
    args_state = args.state_dict()
    checkpoint_args_manifest = getattr(trainer, 'checkpoint_args_manifest', None)
    if callable(checkpoint_args_manifest):
        extra_args = checkpoint_args_manifest()
        if not isinstance(extra_args, Mapping):
            raise TypeError(
                'trainer.checkpoint_args_manifest() must return a mapping, '
                f'got {type(extra_args)!r}'
            )
        collisions = sorted(set(args_state).intersection(extra_args))
        if collisions:
            raise ValueError(
                'trainer.checkpoint_args_manifest() may only add checkpoint args; '
                f'collisions={collisions}'
            )
        args_state.update(dict(extra_args))
    return args_state


class CKPTSaver(object):
    def __init__(self, is_master: bool, eval_milestone: List[Tuple[float, float]]):
        self.is_master = is_master
        self.time_stamp = torch.tensor([time.time() - 1e5, time.time()], device=dist.get_device())
        self.sp_also: subprocess.Popen = None
        self.sp_best: subprocess.Popen = None
        self.sp_backup: subprocess.Popen = None
        self.acc_str, self.eval_milestone = '[no acc str]', eval_milestone

    def sav(
        self, args: arg_util.Args, g_it: int, next_ep: int, next_it: int, trainer,
        acc_str: Optional[str] = None, eval_milestone: Optional[List[Tuple[float, float]]] = None,
        also_save_to: str = None, best_save_to: str = None,
    ):

        if acc_str is not None: self.acc_str = acc_str
        if eval_milestone is not None: self.eval_milestone = eval_milestone

        fname = f'global_step_{g_it}.pth'
        local_out_ckpt = os.path.join(args.local_out_path, fname)

        # NOTE: all rank should call this state_dict(), not master only!
        trainer_state = trainer.state_dict()
        args_state = _checkpoint_args_state(args, trainer)

        if self.is_master:
            stt = time.time()
            torch.save({
                'args':         args_state,
                'gpt_training': args.gpt_training,
                'arch':         args.model,
                'epoch':        next_ep,
                'iter':         next_it,
                'trainer':      trainer_state,
                'g_it':         g_it,
            }, local_out_ckpt)
            cost = time.time() - stt
            print(f'[CKPTSaver][rank00] saved {local_out_ckpt} cost: {cost:.2f}s', flush=True)

        del trainer_state
        time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
        dist.barrier()
