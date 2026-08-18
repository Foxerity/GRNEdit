import torch.distributed as tdist

from grn.dataset.official_t2iv.dataset_joint_vi import JointViDataset
from grn.dataset.official_t2iv.dataset_edit_pair import EditPairJointViDataset
from grn.utils_t2iv.sequence_parallel import SequenceParallelManager as sp_manager


def build_official_t2iv_dataset(args):
    return JointViDataset(
        meta_folders=args.meta_folders,
        meta_folder_repeats=args.meta_folder_repeats,
        max_caption_len=args.tlen,
        short_prob=args.short_cap_prob,
        load_vae_instead_of_image=False,
        pn=args.pn,
        video_fps=args.video_fps,
        num_frames=args.video_frames,
        online_t5=args.online_t5,
        num_replicas=sp_manager.get_sp_group_nums() if sp_manager.sp_on() else tdist.get_world_size(),
        rank=sp_manager.get_sp_group_rank() if sp_manager.sp_on() else tdist.get_rank(),
        dataloader_workers=args.workers,
        enable_dynamic_length_prompt=args.enable_dynamic_length_prompt,
        hdfs_mode=args.hdfs_mode,
        dynamic_scale_schedule=args.dynamic_scale_schedule,
        seed=args.seed,
        other_args=args,
    )


def build_official_t2iv_edit_pair_dataset(args):
    return EditPairJointViDataset(
        meta_folders=args.meta_folders,
        meta_folder_repeats=args.meta_folder_repeats,
        max_caption_len=args.tlen,
        short_prob=args.short_cap_prob,
        load_vae_instead_of_image=False,
        pn=args.pn,
        video_fps=args.video_fps,
        num_frames=args.video_frames,
        online_t5=args.online_t5,
        num_replicas=sp_manager.get_sp_group_nums() if sp_manager.sp_on() else tdist.get_world_size(),
        rank=sp_manager.get_sp_group_rank() if sp_manager.sp_on() else tdist.get_rank(),
        dataloader_workers=args.workers,
        enable_dynamic_length_prompt=args.enable_dynamic_length_prompt,
        hdfs_mode=args.hdfs_mode,
        dynamic_scale_schedule=args.dynamic_scale_schedule,
        seed=args.seed,
        other_args=args,
    )
