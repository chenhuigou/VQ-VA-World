# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# Copyright 2025 Zeyu Wang & Zilong Chen.
# SPDX-License-Identifier: Apache-2.0

from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def patchify(image, patch_size):
    p = patch_size
    c, h, w = image.shape
    assert h % p == 0 and w % p == 0
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    image = image.reshape(-1, p**2 * c)
    return image


def qwen2_5_vl_patchify(image_tensor, temporal_patch_size, patch_size, merge_size):
    if image_tensor.shape[0] % temporal_patch_size != 0:
        repeats = torch.repeat_interleave(
            image_tensor[-1][None, :], temporal_patch_size - (image_tensor.shape[0] % temporal_patch_size), dim=0
        )
        patches = torch.cat([image_tensor, repeats], dim=0)
    channel, height, width = patches.shape[1], patches.shape[2], patches.shape[3]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = patches.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
    )
    return flatten_patches


def get_qwen2_5_vl_mrope_index(
        st_idx: int,
        spatial_merge_size: int,
        tokens_per_second: int,
        image_grid_thw: Optional[List[int]] = None,
        video_grid_thw: Optional[List[int]] = None,
        second_per_grid_t: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.
        """
        assert not (image_grid_thw is not None and video_grid_thw is not None)
        
        if image_grid_thw is not None:
            t, h, w = (
                image_grid_thw[0],
                image_grid_thw[1],
                image_grid_thw[2],
            )
            second_per_grid_t = 0.0
        else:
            t, h, w = (
                video_grid_thw[0],
                video_grid_thw[1],
                video_grid_thw[2],
            )
            if second_per_grid_t is None:
                second_per_grid_t = 1.0

        llm_grid_t, llm_grid_h, llm_grid_w = (
            t,
            h // spatial_merge_size,
            w // spatial_merge_size,
        )

        range_tensor = torch.arange(llm_grid_t).view(-1, 1)
        expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

        ## normalize type, send to device.
        second_per_grid_t = torch.as_tensor(
            second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
        )
        time_tensor = expanded_range * second_per_grid_t * tokens_per_second
        time_tensor_long = time_tensor.long()
        t_index = time_tensor_long.flatten()

        t_index = t_index + st_idx
        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten() + st_idx
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten() + st_idx

        return t_index.tolist(), h_index.tolist(), w_index.tolist()


def prepare_attention_mask_per_sample(split_lens, attn_modes, device="cpu"):
    """
    nested_split_lens: A list of N lists of ints. Each int indicates the length of a split within 
        a sample, where each sample contains multiple splits with different attn modes.
    nested_attn_modes: whether to use full attn in each split.
    """
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ['causal', 'full', 'noise']
        if attn_mode == "causal":
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s), device=device).tril()
            attention_mask[csum:csum + s, :csum] = 1
        else:
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s))
            attention_mask[csum:csum + s, :csum] = 1
        csum += s

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            attention_mask[:, csum : csum + s] = torch.zeros((sample_len, s))
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s))
        csum += s

    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float).masked_fill_(
        ~attention_mask, float("-inf")
    )

    return attention_mask


def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image


def len2weight(x, loss_reduction='square'):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)