import os

import torch
from torchvision.utils import make_grid
import torch.distributed as dist
from PIL import Image
import argparse
import hashlib
import math


def _env_disables_wandb():
    value = os.environ.get('GRN_DISABLE_WANDB', os.environ.get('WANDB_DISABLED', 'false'))
    return str(value).lower() in {'1', 'true', 'yes', 'on'}


class _NoOpWandb:
    class Image:
        def __init__(self, *args, **kwargs):
            pass

    def login(self, *args, **kwargs):
        return None

    def init(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        return None


if _env_disables_wandb():
    wandb = _NoOpWandb()
    _WANDB_AVAILABLE = False
else:
    try:
        import wandb
        _WANDB_AVAILABLE = True
    except ImportError:
        wandb = _NoOpWandb()
        _WANDB_AVAILABLE = False


def enabled():
    return _WANDB_AVAILABLE and not _env_disables_wandb()


def is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name):
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def initialize(args, entity, exp_name, project_name):
    if not enabled():
        return
    config_dict = namespace_to_dict(args)
    wandb.login(key=os.environ["WANDB_KEY"])
    wandb.init(
        entity=entity,
        project=project_name,
        name=exp_name,
        config=config_dict,
        id=generate_run_id(exp_name),
        resume="allow",
    )


def log(stats, step=None):
    if enabled() and is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(name, sample, step=None):
    if enabled() and is_main_process():
        sample = array2grid(sample)
        wandb.log({f"{name}": wandb.Image(sample), "train_step": step})


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1,1))
    x = x.mul(255).add_(0.5).clamp_(0,255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x
