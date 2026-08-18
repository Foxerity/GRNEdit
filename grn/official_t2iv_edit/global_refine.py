import numpy as np
import torch

def get_scale_pack_info(**kwargs):
    return

def flatten_two_level_list(two_level_list):
    flatten_list = []
    for item in two_level_list:
        flatten_list.extend(item)
    return flatten_list

def shift_pt(pt, alpha):
    """shift pt (signal ratio) to lower one, recommand alpha=sqrt(height*width/256/256)"""
    if alpha > 1000:
        alpha = alpha - 1000
    noise_pt = 1 - pt
    noise_pt = alpha * noise_pt / (1+(alpha-1)*noise_pt) # shift noise_pt to higer one
    pt = 1 - noise_pt
    return pt

def video_decode(
    vae,
    all_indices,
    scale_schedule,
    label_type,
    args=None,
    noise_list=None,
    trunc_scales=-1,
    **kwargs,
):
    if trunc_scales < 0:
        summed_codes = all_indices[-1]
    else:
        summed_codes = all_indices[trunc_scales-1]
    x_recon = vae.decode(summed_codes, slice=True)
    x_recon = torch.clamp(x_recon, min=-1, max=1)
    x_recon_256 = None
    return x_recon, x_recon_256

def get_visual_rope_embeds(rope2d_freqs_grid, scale_schedule, device=None, mapped_h_div_w_template=None, t_offset=0):
    # freqs_frames: (2, max_frames, dim_div_2 / 3)
    rope2d_freqs_grid['freqs_frames'] = rope2d_freqs_grid['freqs_frames'].to(device)
    rope2d_freqs_grid['freqs_height'] = rope2d_freqs_grid['freqs_height'].to(device)
    rope2d_freqs_grid['freqs_width'] = rope2d_freqs_grid['freqs_width'].to(device)
    max_height = rope2d_freqs_grid['freqs_height'].shape[1]
    max_width = rope2d_freqs_grid['freqs_width'].shape[1]
    extreme_h_div_w = 3
    assert mapped_h_div_w_template <= extreme_h_div_w
    extreme_h = max_height
    extreme_w = extreme_h / extreme_h_div_w
    upw = np.sqrt(extreme_h * extreme_w / mapped_h_div_w_template)
    uph = mapped_h_div_w_template * upw
    uph, upw = int(uph), int(upw)
    pt, ph, pw = scale_schedule
    assert ph <= uph and pw <= upw
    f_frames = rope2d_freqs_grid['freqs_frames'][:, t_offset:t_offset+pt]
    f_height = rope2d_freqs_grid['freqs_height'][:, (torch.arange(ph) * (uph / ph)).round().int()]
    f_width = rope2d_freqs_grid['freqs_width'][:, (torch.arange(pw) * (upw / pw)).round().int()]
    rope_embeds = torch.cat([
        f_frames[   :,     :,  None,   None,   :].expand(-1, -1, ph, pw, -1),
        f_height[   :,  None,      :,  None,   :].expand(-1,  pt,-1, pw, -1),
        f_width[   :,  None,   None,      :,   :].expand(-1,  pt,ph, -1, -1),
    ], dim=-1)  # (2, pt, ph, pw, dim_div_2)
    rope_embeds = rope_embeds.reshape(2, 1, 1, 1, pt*ph*pw, -1)  # (2, 1, 1, 1, pt*ph*pw, dim_div_2)
    return rope_embeds
