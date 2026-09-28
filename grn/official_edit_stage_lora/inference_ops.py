"""Source latent packing and spatial/temporal RoPE for Stage-LoRA inference."""

import numpy as np
import torch
from grn.utils_t2iv.hbq_util_t2iv import (
    multiclass_labels2onehot_input,
    raw_feature2bit_label,
)


def shift_pt(pt, alpha):
    """Apply the GRN signal-to-noise shift used by refinement."""
    if alpha > 1000:
        alpha = alpha - 1000
    noise_pt = 1 - pt
    noise_pt = alpha * noise_pt / (1 + (alpha - 1) * noise_pt)
    return 1 - noise_pt


def get_visual_rope_embeds(
    rope2d_freqs_grid,
    scale_schedule,
    device=None,
    mapped_h_div_w_template=None,
    t_offset=0,
):
    # freqs_frames: (2, max_frames, dim_div_2 / 3)
    rope2d_freqs_grid["freqs_frames"] = rope2d_freqs_grid["freqs_frames"].to(device)
    rope2d_freqs_grid["freqs_height"] = rope2d_freqs_grid["freqs_height"].to(device)
    rope2d_freqs_grid["freqs_width"] = rope2d_freqs_grid["freqs_width"].to(device)
    max_height = rope2d_freqs_grid["freqs_height"].shape[1]
    extreme_h_div_w = 3
    assert mapped_h_div_w_template <= extreme_h_div_w
    extreme_h = max_height
    extreme_w = extreme_h / extreme_h_div_w
    upw = np.sqrt(extreme_h * extreme_w / mapped_h_div_w_template)
    uph = mapped_h_div_w_template * upw
    uph, upw = int(uph), int(upw)
    pt, ph, pw = scale_schedule
    assert ph <= uph and pw <= upw
    f_frames = rope2d_freqs_grid["freqs_frames"][:, t_offset : t_offset + pt]
    f_height = rope2d_freqs_grid["freqs_height"][
        :, (torch.arange(ph) * (uph / ph)).round().int()
    ]
    f_width = rope2d_freqs_grid["freqs_width"][
        :, (torch.arange(pw) * (upw / pw)).round().int()
    ]
    rope_embeds = torch.cat(
        [
            f_frames[:, :, None, None, :].expand(-1, -1, ph, pw, -1),
            f_height[:, None, :, None, :].expand(-1, pt, -1, pw, -1),
            f_width[:, None, None, :, :].expand(-1, pt, ph, -1, -1),
        ],
        dim=-1,
    )  # (2, pt, ph, pw, dim_div_2)
    rope_embeds = rope_embeds.reshape(
        2, 1, 1, 1, pt * ph * pw, -1
    )  # (2, 1, 1, 1, pt*ph*pw, dim_div_2)
    return rope_embeds


def source_raw_feature_to_inference_tokens(raw_feature, hbq_round):
    """Convert a source latent to the full, uncorrupted bit one-hot stream."""
    labels = raw_feature2bit_label(raw_feature, hbq_round=hbq_round)
    return (
        multiclass_labels2onehot_input(labels, 2)
        .flatten(2)
        .transpose(1, 2)
        .contiguous()
    )
