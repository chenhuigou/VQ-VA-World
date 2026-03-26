# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 Zeyu Wang & Zilong Chen.
# SPDX-License-Identifier: Apache-2.0
#
# Think Gen inference script for LightFusionWorld.
# Adapted from cdt-hf eval/edit/qwen25vl_wan22_i2i_inference_training_format_system_in_middle.py
# Changes: import paths, cross_attn->mm_attn naming, wan_generate_video->visual_gen

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
import os
import argparse
import copy
import json
import torch
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from io import BytesIO
from safetensors.torch import load_file
from pathlib import Path
import tempfile
import shutil

from transformers import set_seed
from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb
from modeling.lightfusion.qwen25vl_navit_fusion import NaiveCache
from eval.utils import load_model_and_tokenizer


SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here\n'''

# pil_img2rgb imported from data.data_utils
def editing_generate(
    model,
    vae_model,
    tokenizer,
    special_token_ids,
    vae_image_transform,
    vit_image_transform,
    image,
    prompt,
    num_timesteps,
    timestep_shift,
    cfg_text_scale,
    cfg_img_scale,
    cfg_interval,
    cfg_renorm_min,
    cfg_renorm_type,
    flow_solver,
    negative_prompt,
    mm_attn_num_layer_type="min",
):
    past_und_key_values = NaiveCache(model.config.vlm_config.num_hidden_layers)
    if mm_attn_num_layer_type == "min":
        past_uni_key_values = NaiveCache(min(model.config.vlm_config.num_hidden_layers, model.vgen_model.num_layers) - 1)
    elif mm_attn_num_layer_type == "max":
        past_uni_key_values = NaiveCache(max(model.config.vlm_config.num_hidden_layers, model.vgen_model.num_layers) - 1)
    else:
        raise NotImplementedError
    if isinstance(image, list):
        generation_input, new_und_lens, new_uni_lens, new_rope = model.prepare_vae_images(
            curr_und_kvlens=[0],
            curr_uni_kvlens=[0],
            curr_rope=[0],
            images=[image[0]],
            transforms=vae_image_transform,
            special_token_ids=special_token_ids,
        )
        video_sizes = [*generation_input['padded_images'].shape[2:]]

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_und_key_values, past_uni_key_values = model.forward_cache_update_vae(**generation_input, vae_model=vae_model, past_und_key_values=past_und_key_values, past_uni_key_values=past_uni_key_values)

        # vit for image transform
        generation_input, new_und_lens, new_uni_lens, new_rope = model.prepare_vit_images(
            curr_und_kvlens=new_und_lens,
            curr_uni_kvlens=new_uni_lens,
            curr_rope=new_rope,
            images=[image[0]],
            transforms=vit_image_transform,
            special_token_ids=special_token_ids,
        )

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_und_key_values, past_uni_key_values = model.forward_cache_update_vit(**generation_input, past_und_key_values=past_und_key_values, past_uni_key_values=past_uni_key_values)
        
        for img in image[1:]:
            generation_input, new_und_lens, new_uni_lens, new_rope = model.prepare_vae_images(
                curr_und_kvlens=new_und_lens,
                curr_uni_kvlens=new_uni_lens,
                curr_rope=new_rope,
                images=[img],
                transforms=vae_image_transform,
                special_token_ids=special_token_ids,
            )
            video_sizes = [*generation_input['padded_images'].shape[2:]]
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_und_key_values, past_uni_key_values = model.forward_cache_update_vae(**generation_input, vae_model=vae_model, past_und_key_values=past_und_key_values, past_uni_key_values=past_uni_key_values)
            generation_input, new_und_lens, new_uni_lens, new_rope = model.prepare_vit_images(
                curr_und_kvlens=new_und_lens,
                curr_uni_kvlens=new_uni_lens,
                curr_rope=new_rope,
                images=[img],
                transforms=vit_image_transform,
                special_token_ids=special_token_ids,
            )

            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_und_key_values, past_uni_key_values = model.forward_cache_update_vit(**generation_input, past_und_key_values=past_und_key_values, past_uni_key_values=past_uni_key_values)
    else:
        generation_input, new_und_lens, new_uni_lens, new_rope = model.prepare_vae_images(
            curr_und_kvlens=[0],
            curr_uni_kvlens=[0],
            curr_rope=[0],
            images=[image],
            transforms=vae_image_transform,
            special_token_ids=special_token_ids,
        )
        video_sizes = [*generation_input['padded_images'].shape[2:]]

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_und_key_values, past_uni_key_values = model.forward_cache_update_vae(**generation_input, vae_model=vae_model, past_und_key_values=past_und_key_values, past_uni_key_values=past_uni_key_values)

        # vit for image transform
        generation_input, new_und_lens, new_uni_lens, new_rope = model.prepare_vit_images(
            curr_und_kvlens=new_und_lens,
            curr_uni_kvlens=new_uni_lens,
            curr_rope=new_rope,
            images=[image],
            transforms=vit_image_transform,
            special_token_ids=special_token_ids,
        )

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_und_key_values, past_uni_key_values = model.forward_cache_update_vit(**generation_input, past_und_key_values=past_und_key_values, past_uni_key_values=past_uni_key_values)
    ###############################
    cfg_text_past_und_key_values = copy.deepcopy(past_und_key_values)
    cfg_text_past_uni_key_values = copy.deepcopy(past_uni_key_values)
    cfg_text_new_und_lens = copy.deepcopy(new_und_lens)
    cfg_text_new_uni_lens = copy.deepcopy(new_uni_lens)
    cfg_text_new_rope = copy.deepcopy(new_rope)

    cfg_text_generation_input, cfg_text_new_und_lens, cfg_text_new_uni_lens, cfg_text_new_rope = model.prepare_prompts(
        curr_und_kvlens=cfg_text_new_und_lens,
        curr_uni_kvlens=cfg_text_new_uni_lens,
        curr_rope=cfg_text_new_rope,
        prompts=[negative_prompt],
        tokenizer=tokenizer,
        special_token_ids=special_token_ids,
    )

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_text_past_und_key_values, cfg_text_past_uni_key_values = model.forward_cache_update_text(**cfg_text_generation_input, past_und_key_values=cfg_text_past_und_key_values, past_uni_key_values=cfg_text_past_uni_key_values)

    cfg_text_generation_input = model.prepare_vae_latent_cfg(
        curr_und_kvlens=cfg_text_new_und_lens,
        curr_uni_kvlens=cfg_text_new_uni_lens,
        curr_rope=cfg_text_new_rope,
        video_sizes=[video_sizes],
    )
    ###############################

    ###############################
    cfg_img_past_und_key_values = NaiveCache(model.config.vlm_config.num_hidden_layers)
    if mm_attn_num_layer_type == "min":
        cfg_img_past_uni_key_values = NaiveCache(min(model.config.vlm_config.num_hidden_layers, model.vgen_model.num_layers) - 1)
    elif mm_attn_num_layer_type == "max":
        cfg_img_past_uni_key_values = NaiveCache(max(model.config.vlm_config.num_hidden_layers, model.vgen_model.num_layers) - 1)
    else:
        raise NotImplementedError(f"Unknown cross attn layer type {mm_attn_num_layer_type}")
    cfg_img_new_und_lens = [0]
    cfg_img_new_uni_lens = [0]
    cfg_img_new_rope = [0]

    cfg_img_generation_input, cfg_img_new_und_lens, cfg_img_new_uni_lens, cfg_img_new_rope = model.prepare_prompts(
        curr_und_kvlens=cfg_img_new_und_lens,
        curr_uni_kvlens=cfg_img_new_uni_lens,
        curr_rope=cfg_img_new_rope,
        prompts=[prompt],
        tokenizer=tokenizer,
        special_token_ids=special_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_und_key_values, cfg_img_past_uni_key_values = model.forward_cache_update_text(**cfg_img_generation_input, past_und_key_values=cfg_img_past_und_key_values, past_uni_key_values=cfg_img_past_uni_key_values)

    cfg_img_generation_input = model.prepare_vae_latent_cfg(
        curr_und_kvlens=cfg_img_new_und_lens,
        curr_uni_kvlens=cfg_img_new_uni_lens,
        curr_rope=cfg_img_new_rope,
        video_sizes=[video_sizes],
    )
    ###############################

    generation_input, new_und_lens, new_uni_lens, new_rope = model.prepare_prompts(
        curr_und_kvlens=new_und_lens,
        curr_uni_kvlens=new_uni_lens,
        curr_rope=new_rope,
        prompts=[prompt],
        tokenizer=tokenizer,
        special_token_ids=special_token_ids,
    )

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_und_key_values, past_uni_key_values = model.forward_cache_update_text(**generation_input, past_und_key_values=past_und_key_values, past_uni_key_values=past_uni_key_values)

    generation_input = model.prepare_vae_latent(
        curr_und_kvlens=new_und_lens,
        curr_uni_kvlens=new_uni_lens,
        curr_rope=new_rope,
        video_sizes=[video_sizes],
        special_token_ids=special_token_ids,
    )

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = model.visual_gen(
            **generation_input,
            past_und_key_values=past_und_key_values,
            past_uni_key_values=past_uni_key_values,
            flow_solver=flow_solver,
            num_timesteps=num_timesteps,
            timestep_shift=timestep_shift,
            # cfg_scale=cfg_scale,
            # cfg_renorm=cfg_renorm,
            # cfg_past_und_key_values=cfg_past_und_key_values,
            # cfg_past_uni_key_values=cfg_past_uni_key_values,
            # **cfg_generation_input,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            cfg_interval=cfg_interval,
            # cfg_text
            cfg_text_scale=cfg_text_scale,
            cfg_text_packed_und_query_indexes=cfg_text_generation_input["cfg_packed_und_query_indexes"],
            cfg_text_packed_uni_query_indexes=cfg_text_generation_input["cfg_packed_uni_query_indexes"],
            cfg_text_past_und_key_values=cfg_text_past_und_key_values,
            cfg_text_past_uni_key_values=cfg_text_past_uni_key_values,
            cfg_text_key_values_lens_und=cfg_text_generation_input["cfg_key_values_lens_und"],
            cfg_text_key_values_lens_uni=cfg_text_generation_input["cfg_key_values_lens_uni"],
            cfg_text_packed_und_key_value_indexes=cfg_text_generation_input["cfg_packed_und_key_value_indexes"],
            cfg_text_packed_uni_key_value_indexes=cfg_text_generation_input["cfg_packed_uni_key_value_indexes"],
            cfg_text_packed_position_ids=cfg_text_generation_input["cfg_packed_position_ids"],
            # cfg_img
            cfg_img_scale=cfg_img_scale,
            cfg_img_packed_und_query_indexes=cfg_img_generation_input["cfg_packed_und_query_indexes"],
            cfg_img_packed_uni_query_indexes=cfg_img_generation_input["cfg_packed_uni_query_indexes"],
            cfg_img_past_und_key_values=cfg_img_past_und_key_values,
            cfg_img_past_uni_key_values=cfg_img_past_uni_key_values,
            cfg_img_key_values_lens_und=cfg_img_generation_input["cfg_key_values_lens_und"],
            cfg_img_key_values_lens_uni=cfg_img_generation_input["cfg_key_values_lens_uni"],
            cfg_img_packed_und_key_value_indexes=cfg_img_generation_input["cfg_packed_und_key_value_indexes"],
            cfg_img_packed_uni_key_value_indexes=cfg_img_generation_input["cfg_packed_uni_key_value_indexes"],
            cfg_img_packed_position_ids=cfg_img_generation_input["cfg_packed_position_ids"],
        )

    image = vae_model.decode([unpacked_latent[0]])[0]
    image = image.squeeze(1)
    image = ((image * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    return image


def editing_generate_with_thinking(
    model,
    vae_model,
    tokenizer,
    special_token_ids,
    vae_image_transform,
    vit_image_transform,
    image,
    prompt,
    num_timesteps,
    timestep_shift,
    cfg_text_scale,
    cfg_img_scale,
    cfg_interval,
    cfg_renorm_min,
    cfg_renorm_type,
    flow_solver,
    negative_prompt,
    mm_attn_num_layer_type="min",
    system_prompt=SYSTEM_PROMPT,
    max_length=384,
    do_sample=True,
    temperature=0.3,
    
):
    past_und_key_values = NaiveCache(model.config.vlm_config.num_hidden_layers)
    if mm_attn_num_layer_type == "min":
        past_uni_key_values = NaiveCache(min(model.config.vlm_config.num_hidden_layers, model.vgen_model.num_layers) - 1)
    elif mm_attn_num_layer_type == "max":
        past_uni_key_values = NaiveCache(max(model.config.vlm_config.num_hidden_layers, model.vgen_model.num_layers) - 1)
    else:
        raise NotImplementedError
    #prompt = "Draw "+ prompt
    #prompt = prompt + "\n Please notice you need to keep the image consistency, maintain the main object, background and pose same."
    ######### VAE + VIT
    #prompt = prompt + "\n Please notice you need also think to keep the image consistency, maintain the main object, background and pose same."
    #prompt = "Please notice you need to keep the image consistency, maintain the main object, background and pose same.\n" + prompt
    if not isinstance(image, list):
        image = [image]
    else:
        print("image",image)
        print("here:=======",len(image))
    
    #for img in image:
    generation_input, new_und_lens, new_uni_lens, new_rope = model.prepare_vae_images(
        curr_und_kvlens=[0],
        curr_uni_kvlens=[0],
        curr_rope=[0],
        images=[image[0]],
        transforms=vae_image_transform,
        special_token_ids=special_token_ids,
    )
    #print("hereM an")
    video_sizes = [*generation_input['padded_images'].shape[2:]]
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_und_key_values, past_uni_key_values = model.forward_cache_update_vae(**generation_input, vae_model=vae_model, past_und_key_values=past_und_key_values, past_uni_key_values=past_uni_key_values)
    generation_input, new_und_lens, new_uni_lens, new_rope = model.prepare_vit_images(
        curr_und_kvlens=new_und_lens,
        curr_uni_kvlens=new_uni_lens,
        curr_rope=new_rope,
        images=[image[0]],
        transforms=vit_image_transform,
        special_token_ids=special_token_ids,
    )

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_und_key_values, past_uni_key_values = model.forward_cache_update_vit(**generation_input, past_und_key_values=past_und_key_values, past_uni_key_values=past_uni_key_values)

    for img in image[1:]:
        generation_input, new_und_lens, new_uni_lens, new_rope = model.prepare_vae_images(
            curr_und_kvlens=new_und_lens,
            curr_uni_kvlens=new_uni_lens,
            curr_rope=new_rope,
            images=[img],
            transforms=vae_image_transform,
            special_token_ids=special_token_ids,
        )
        video_sizes = [*generation_input['padded_images'].shape[2:]]
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_und_key_values, past_uni_key_values = model.forward_cache_update_vae(**generation_input, vae_model=vae_model, past_und_key_values=past_und_key_values, past_uni_key_values=past_uni_key_values)
        generation_input, new_und_lens, new_uni_lens, new_rope = model.prepare_vit_images(
            curr_und_kvlens=new_und_lens,
            curr_uni_kvlens=new_uni_lens,
            curr_rope=new_rope,
            images=[img],
            transforms=vit_image_transform,
            special_token_ids=special_token_ids,
        )

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_und_key_values, past_uni_key_values = model.forward_cache_update_vit(**generation_input, past_und_key_values=past_und_key_values, past_uni_key_values=past_uni_key_values)    

    ############ Text CFG = VAE + VIT
    cfg_text_past_und_key_values = copy.deepcopy(past_und_key_values)
    cfg_text_past_uni_key_values = copy.deepcopy(past_uni_key_values)
    cfg_text_new_und_lens = copy.deepcopy(new_und_lens)
    cfg_text_new_uni_lens = copy.deepcopy(new_uni_lens)
    cfg_text_new_rope = copy.deepcopy(new_rope)

    cfg_text_generation_input = model.prepare_vae_latent_cfg(
        curr_und_kvlens=cfg_text_new_und_lens,
        curr_uni_kvlens=cfg_text_new_uni_lens,
        curr_rope=cfg_text_new_rope,
        video_sizes=[video_sizes],
    )

    ########### continue main input VAE+VIT+SYSTEM+PROMPT+THINK
    generation_input, new_und_lens, new_uni_lens, new_rope = model.prepare_prompts(
        curr_und_kvlens=new_und_lens,
        curr_uni_kvlens=new_uni_lens,
        curr_rope=new_rope,
        prompts=[system_prompt],
        tokenizer=tokenizer,
        special_token_ids=special_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_und_key_values, past_uni_key_values = model.forward_cache_update_text(**generation_input, past_und_key_values=past_und_key_values, past_uni_key_values=past_uni_key_values)
    ##### add prompt
    generation_input, new_und_lens, new_uni_lens, new_rope = model.prepare_prompts(
        curr_und_kvlens=new_und_lens,
        curr_uni_kvlens=new_uni_lens,
        curr_rope=new_rope,
        prompts=[prompt],
        tokenizer=tokenizer,
        special_token_ids=special_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_und_key_values, past_uni_key_values = model.forward_cache_update_text(**generation_input, past_und_key_values=past_und_key_values, past_uni_key_values=past_uni_key_values)

    #### ADD think after IMAGE CFG, so far, past_und_key_values = VAE+VIT+SYSTEM+PROMPT

    ###############################

    ############# IMAGE CFG = SYSTEM + PROMPT+ THINK
    cfg_img_past_und_key_values = NaiveCache(model.config.vlm_config.num_hidden_layers)
    if mm_attn_num_layer_type == "min":
        cfg_img_past_uni_key_values = NaiveCache(min(model.config.vlm_config.num_hidden_layers, model.vgen_model.num_layers) - 1)
    elif mm_attn_num_layer_type == "max":
        cfg_img_past_uni_key_values = NaiveCache(max(model.config.vlm_config.num_hidden_layers, model.vgen_model.num_layers) - 1)
    else:
        raise NotImplementedError(f"Unknown cross attn layer type {mm_attn_num_layer_type}")
    
    cfg_img_new_und_lens = [0]
    cfg_img_new_uni_lens = [0]
    cfg_img_new_rope = [0]

    cfg_img_generation_input, cfg_img_new_und_lens, cfg_img_new_uni_lens, cfg_img_new_rope = model.prepare_prompts(
        curr_und_kvlens=cfg_img_new_und_lens,
        curr_uni_kvlens=cfg_img_new_uni_lens,
        curr_rope=cfg_img_new_rope,
        prompts=[system_prompt],
        tokenizer=tokenizer,
        special_token_ids=special_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_und_key_values, cfg_img_past_uni_key_values = model.forward_cache_update_text(**cfg_img_generation_input, past_und_key_values=cfg_img_past_und_key_values, past_uni_key_values=cfg_img_past_uni_key_values)

    cfg_img_generation_input, cfg_img_new_und_lens, cfg_img_new_uni_lens, cfg_img_new_rope = model.prepare_prompts(
        curr_und_kvlens=cfg_img_new_und_lens,
        curr_uni_kvlens=cfg_img_new_uni_lens,
        curr_rope=cfg_img_new_rope,
        prompts=[prompt],
        tokenizer=tokenizer,
        special_token_ids=special_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_und_key_values, cfg_img_past_uni_key_values = model.forward_cache_update_text(**cfg_img_generation_input, past_und_key_values=cfg_img_past_und_key_values, past_uni_key_values=cfg_img_past_uni_key_values)


    
    ########## generate think content
    tmp_past_und_key_values = copy.deepcopy(past_und_key_values)
    tmp_past_uni_key_values = copy.deepcopy(past_uni_key_values)

    generation_input = model.prepare_start_tokens(new_und_lens, new_uni_lens, new_rope, special_token_ids, tokenizer)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = model.generate_text(
            past_und_key_values=tmp_past_und_key_values,
            past_uni_key_values=tmp_past_uni_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=special_token_ids['eos_token_id'],
            **generation_input,
        )
        output = tokenizer.decode(unpacked_latent[:,0])
        print("navie real_prompt, output",prompt, "==\n",output)
        think_output = output.lstrip('\n')
    print("="*30, "original think", "="*30)
    print(think_output)
    print("== max_length == is: ",max_length)
    ##### add think in image cfg
    cfg_img_generation_input, cfg_img_new_und_lens, cfg_img_new_uni_lens, cfg_img_new_rope = model.prepare_prompts(
        curr_und_kvlens=cfg_img_new_und_lens,
        curr_uni_kvlens=cfg_img_new_uni_lens,
        curr_rope=cfg_img_new_rope,
        prompts=[think_output],
        tokenizer=tokenizer,
        special_token_ids=special_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_und_key_values, cfg_img_past_uni_key_values = model.forward_cache_update_text(**cfg_img_generation_input, past_und_key_values=cfg_img_past_und_key_values, past_uni_key_values=cfg_img_past_uni_key_values)

    cfg_img_generation_input = model.prepare_vae_latent_cfg(
        curr_und_kvlens=cfg_img_new_und_lens,
        curr_uni_kvlens=cfg_img_new_uni_lens,
        curr_rope=cfg_img_new_rope,
        video_sizes=[video_sizes],
    )

    ##### origin
    ### new_und_lens=system prompt + VIT + VAE + prompt + think

    # ##### add prompt
    # generation_input, new_und_lens, new_uni_lens, new_rope = model.prepare_prompts(
    #     curr_und_kvlens=new_und_lens,
    #     curr_uni_kvlens=new_uni_lens,
    #     curr_rope=new_rope,
    #     prompts=[prompt],
    #     tokenizer=tokenizer,
    #     special_token_ids=special_token_ids,
    # )
    # with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
    #     past_und_key_values, past_uni_key_values = model.forward_cache_update_text(**generation_input, past_und_key_values=past_und_key_values, past_uni_key_values=past_uni_key_values)

    ### think output
    generation_input, new_und_lens, new_uni_lens, new_rope = model.prepare_prompts(
        curr_und_kvlens=new_und_lens,
        curr_uni_kvlens=new_uni_lens,
        curr_rope=new_rope,
        prompts=[think_output],
        tokenizer=tokenizer,
        special_token_ids=special_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_und_key_values, past_uni_key_values = model.forward_cache_update_text(**generation_input, past_und_key_values=past_und_key_values, past_uni_key_values=past_uni_key_values)

    generation_input = model.prepare_vae_latent(
        curr_und_kvlens=new_und_lens,
        curr_uni_kvlens=new_uni_lens,
        curr_rope=new_rope,
        video_sizes=[video_sizes],
        special_token_ids=special_token_ids,
    )

    # return
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = model.visual_gen(
        #unpacked_latent = model.visual_gen_two_scale(
            **generation_input,
            past_und_key_values=past_und_key_values,
            past_uni_key_values=past_uni_key_values,
            flow_solver=flow_solver,
            num_timesteps=num_timesteps,
            timestep_shift=timestep_shift,
            # cfg_scale=cfg_scale,
            # cfg_renorm=cfg_renorm,
            # cfg_past_und_key_values=cfg_past_und_key_values,
            # cfg_past_uni_key_values=cfg_past_uni_key_values,
            # **cfg_generation_input,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            cfg_interval=cfg_interval,
            # cfg_text
            cfg_text_scale=cfg_text_scale,
            cfg_text_packed_und_query_indexes=cfg_text_generation_input["cfg_packed_und_query_indexes"],
            cfg_text_packed_uni_query_indexes=cfg_text_generation_input["cfg_packed_uni_query_indexes"],
            cfg_text_past_und_key_values=cfg_text_past_und_key_values,
            cfg_text_past_uni_key_values=cfg_text_past_uni_key_values,
            cfg_text_key_values_lens_und=cfg_text_generation_input["cfg_key_values_lens_und"],
            cfg_text_key_values_lens_uni=cfg_text_generation_input["cfg_key_values_lens_uni"],
            cfg_text_packed_und_key_value_indexes=cfg_text_generation_input["cfg_packed_und_key_value_indexes"],
            cfg_text_packed_uni_key_value_indexes=cfg_text_generation_input["cfg_packed_uni_key_value_indexes"],
            cfg_text_packed_position_ids=cfg_text_generation_input["cfg_packed_position_ids"],
            # cfg_img
            cfg_img_scale=cfg_img_scale,
            cfg_img_packed_und_query_indexes=cfg_img_generation_input["cfg_packed_und_query_indexes"],
            cfg_img_packed_uni_query_indexes=cfg_img_generation_input["cfg_packed_uni_query_indexes"],
            cfg_img_past_und_key_values=cfg_img_past_und_key_values,
            cfg_img_past_uni_key_values=cfg_img_past_uni_key_values,
            cfg_img_key_values_lens_und=cfg_img_generation_input["cfg_key_values_lens_und"],
            cfg_img_key_values_lens_uni=cfg_img_generation_input["cfg_key_values_lens_uni"],
            cfg_img_packed_und_key_value_indexes=cfg_img_generation_input["cfg_packed_und_key_value_indexes"],
            cfg_img_packed_uni_key_value_indexes=cfg_img_generation_input["cfg_packed_uni_key_value_indexes"],
            cfg_img_packed_position_ids=cfg_img_generation_input["cfg_packed_position_ids"],
        )
    #@return
    image = vae_model.decode([unpacked_latent[0]])[0]
    image = image.squeeze(1)
    image = ((image * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    return image,think_output


def emuedit_inference(
    model, vae_model, tokenizer, 
    special_token_ids, vae_image_transform, vit_image_transform,
    parquet_path, output_dir,
    num_timesteps, timestep_shift,
    cfg_text_scale, cfg_img_scale, cfg_interval, cfg_renorm_min, cfg_renorm_type,
    flow_solver, negative_prompt,
    rank=0, world_size=1, mm_attn_num_layer_type="min"
):
    """
    Process images from the dataset using the editing model.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ① 先把已处理的 hash 读出来  ------------------------------ NEW
    processed_hashes = {p.stem for p in Path(output_dir).glob("*.png")}
    print(f"Found {len(processed_hashes)} images already in {output_dir}")

    # ② 读 parquet
    print(f"Loading dataset from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = df.iloc[rank::world_size]
    
    total_cnt = len(df)
    mask_processed = df["hash"].isin(processed_hashes)
    already_cnt   = mask_processed.sum()

    df = df[~mask_processed]        # 只保留尚未处理的行

    remain_cnt = len(df)

    print(
        f"Dataset total: {total_cnt} | "
        f"already processed: {already_cnt} | "
        f"to process this run: {remain_cnt}"
    )
    # ------------------------------------------------------------

    # ③ 主循环
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        hash_value = row["hash"]

        # 如果已经处理过就跳过 ------------------------------- NEW
        if hash_value in processed_hashes:
            continue

        instruction = row["instruction"]
        image_data  = row["image"]


        try:
            input_image = Image.open(BytesIO(image_data["bytes"])).convert('RGB')

            edited_image = editing_generate(
                model,
                vae_model,
                tokenizer,
                special_token_ids,
                vae_image_transform,
                vit_image_transform,
                input_image,
                instruction,
                num_timesteps,
                timestep_shift,
                cfg_text_scale,
                cfg_img_scale,
                cfg_interval,
                cfg_renorm_min,
                cfg_renorm_type,
                flow_solver,
                negative_prompt,
                mm_attn_num_layer_type=mm_attn_num_layer_type,
            )

            output_path = os.path.join(output_dir, f"{hash_value}.png")
            edited_image.save(output_path)
            # 可选：把刚保存的文件加入集合，防止同一轮里重复
            processed_hashes.add(hash_value)

        except Exception as e:
            print(f"Error processing image {hash_value}: {e}")


def gedit_inference(
    model, vae_model, tokenizer, 
    special_token_ids, vae_image_transform, vit_image_transform,
    data_path, output_dir,
    num_timesteps, timestep_shift,
    cfg_text_scale, cfg_img_scale, cfg_interval, cfg_renorm_min, cfg_renorm_type,
    flow_solver, negative_prompt,
    rank=0, world_size=1, mm_attn_num_layer_type="min"
):
    """
    Process images from the dataset using the editing model.
    """
    os.makedirs(output_dir, exist_ok=True)

    info_path = os.path.join(data_path, 'info.jsonl')
    with open(info_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = lines[rank::world_size]

    for line in tqdm(lines):
        data = json.loads(line)
        
        task_type = data['task_type']
        key = data['key']
        instruction_language = data['instruction_language']

        save_path_fullset_source_image = f"{output_dir}/fullset/{task_type}/{instruction_language}/{key}_SRCIMG.png"
        save_path_fullset = f"{output_dir}/fullset/{task_type}/{instruction_language}/{key}.png"
        os.makedirs(os.path.dirname(save_path_fullset_source_image), exist_ok=True)
        os.makedirs(os.path.dirname(save_path_fullset), exist_ok=True)

        if os.path.exists(save_path_fullset_source_image) and os.path.exists(save_path_fullset):
            print(f'sample {key} already generated, skipping...')
            continue

        instruction = data["instruction"]
        source_image_path = os.path.join(data_path, f'{key}_SRCIMG.png')
        # source_image_raw_path = os.path.join(data_path, f'{key}_SRCIMG_RAW.png')
        # image_data  = row["image"]
        # TODO: try if use source_image or source_image_raw
        input_image = Image.open(source_image_path)

        try:
            # input_image = Image.open(BytesIO(image_data["bytes"]))

            edited_image = editing_generate(
                model,
                vae_model,
                tokenizer,
                special_token_ids,
                vae_image_transform,
                vit_image_transform,
                input_image,
                instruction,
                num_timesteps,
                timestep_shift,
                cfg_text_scale,
                cfg_img_scale,
                cfg_interval,
                cfg_renorm_min,
                cfg_renorm_type,
                flow_solver,
                negative_prompt,
                mm_attn_num_layer_type=mm_attn_num_layer_type,
            )

            # output_path = os.path.join(output_dir, f"{hash_value}.png")
            # edited_image.save(output_path)
            # 可选：把刚保存的文件加入集合，防止同一轮里重复
            # processed_hashes.add(hash_value)

            input_image.save(save_path_fullset_source_image)
            edited_image.save(save_path_fullset)

        except Exception as e:
            raise
            print(f"Error processing image {key}: {e}")


def imgedit_inference(
    model, vae_model, tokenizer, 
    special_token_ids, vae_image_transform, vit_image_transform,
    data_path, output_dir,
    num_timesteps, timestep_shift,
    cfg_text_scale, cfg_img_scale, cfg_interval, cfg_renorm_min, cfg_renorm_type,
    flow_solver, negative_prompt,
    rank=0, world_size=1, mm_attn_num_layer_type="min",
):
    """
    Process images from the dataset using the editing model.
    """
    os.makedirs(output_dir, exist_ok=True)

    imgedit_prompt_path = os.path.join(data_path, "eval_prompts/basic_edit.json")
    with open(imgedit_prompt_path, "r") as f:
        data = json.load(f)
    data = dict(list(data.items())[rank::world_size])

    inference_list = []
    for key, value in tqdm(data.items()):
        prompt = value["prompt"]
        image_path = os.path.join(data_path, "data/Benchmark/singleturn", value["id"])
        inference_list.append([prompt, key, image_path])

    for prompt, key, image_path in tqdm(inference_list):
        cur_save_path = os.path.join(output_dir, f"{key}.png")
        if os.path.exists(cur_save_path):
            print(f"GPU {args.rank} {key=} '{prompt}' already exists, skipping...")
            continue

        input_image = Image.open(image_path).convert('RGB')
        edited_image = editing_generate(
            model,
            vae_model,
            tokenizer,
            special_token_ids,
            vae_image_transform,
            vit_image_transform,
            input_image,
            prompt,
            num_timesteps,
            timestep_shift,
            cfg_text_scale,
            cfg_img_scale,
            cfg_interval,
            cfg_renorm_min,
            cfg_renorm_type,
            flow_solver,
            negative_prompt,
            mm_attn_num_layer_type=mm_attn_num_layer_type,

        )

        edited_image.save(cur_save_path)


def intelligent_inference(
    model, vae_model, tokenizer, 
    special_token_ids, vae_image_transform, vit_image_transform,
    data_path, output_dir,
    num_timesteps, timestep_shift,
    cfg_text_scale, cfg_img_scale, cfg_interval, cfg_renorm_min, cfg_renorm_type,
    flow_solver, negative_prompt,
    rank=0, world_size=1, think=False, mm_attn_num_layer_type="min", max_length=384,**kwargs,
):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_parquet(data_path)
    df = df
    indexes = list(range(len(df)))[rank::world_size][::-1]

    results = []

    for idx in indexes:
        row = df.iloc[idx]
        img = Image.open(BytesIO(row['images'][0])).convert("RGB")
        prompt = row['caption']['q']
        if not think:
            edited_image = editing_generate(
                model,
                vae_model,
                tokenizer,
                special_token_ids,
                vae_image_transform,
                vit_image_transform,
                img,
                prompt,
                num_timesteps,
                timestep_shift,
                cfg_text_scale,
                cfg_img_scale,
                cfg_interval,
                cfg_renorm_min,
                cfg_renorm_type,
                flow_solver,
                negative_prompt,
                mm_attn_num_layer_type=mm_attn_num_layer_type,

            )
        else:
            edited_image,think_output = editing_generate_with_thinking(
                model,
                vae_model,
                tokenizer,
                special_token_ids,
                vae_image_transform,
                vit_image_transform,
                img,
                prompt,
                num_timesteps,
                timestep_shift,
                cfg_text_scale,
                cfg_img_scale,
                cfg_interval,
                cfg_renorm_min,
                cfg_renorm_type,
                flow_solver,
                negative_prompt,
                max_length=max_length,
                mm_attn_num_layer_type=mm_attn_num_layer_type,

                **kwargs,

            )

        result_row = row.copy()
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        outpath = os.path.join(output_dir, f"{idx}.png")
        edited_image.save(outpath)   # 保存为 PNG
        buffer = BytesIO()
        edited_image.save(buffer, format='PNG')
        result_row["model_out_image"] = buffer.getvalue()
        result_row["model_out_text"] = ""

        results.append(result_row)

    df_out = pd.DataFrame(results)
    # Write to local temp file first, then copy to HDFS to avoid "Device or resource busy" error
    final_path = os.path.join(output_dir, f"shard_{rank}.parquet")
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        tmp_path = tmp.name
    df_out.to_parquet(tmp_path)
    shutil.copy(tmp_path, final_path)
    os.unlink(tmp_path)

    # if think:
    #     with open(txt_output, "w", encoding="utf-8") as f:
    #         f.write(think_output)
    # if shard_id == 0:
    #     dfs = []
    #     for idx in range(0, total_shards):
    #         dfs.append(pd.read_parquet(os.path.join(output_dir, f"shard_{rank}.parquet")))
    #     df_out = pd.concat(dfs, ignore_index=True)
    #     df_out.to_parquet(os.path.join(output_dir, "results.parquet"), index=False)

    #     print(f'Results saved to: {os.path.join(output_dir, "results.parquet")}')


def rise_inference(
    model, vae_model, tokenizer, 
    special_token_ids, vae_image_transform, vit_image_transform,
    data_path, output_dir,
    num_timesteps, timestep_shift,
    cfg_text_scale, cfg_img_scale, cfg_interval, cfg_renorm_min, cfg_renorm_type,
    flow_solver, negative_prompt,
    rank=0, world_size=1, think=False, mm_attn_num_layer_type="min", max_length=384,**kwargs,
):
    """
    Process images from the dataset using the editing model.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(data_path, "datav2_total_w_subtask.json"), "r") as f:
        metadatas = json.load(f)
    total_metadatas = len(metadatas)
    indexes = list(range(total_metadatas))[rank::world_size]
    # rank = dist.get_rank()
    # world_size = dist.get_world_size()
    
    # 将待处理的元数据按 GPU 划分
    # prompts_per_gpu = (total_metadatas + world_size - 1) // world_size
    # start = rank * prompts_per_gpu
    # end = min(start + prompts_per_gpu, total_metadatas)
    # print(f"GPU {rank}: 处理 {end - start} 个提示词 (索引 {start} 到 {end - 1})")
    image_path = data_path

    for idx in indexes:
        metadata = metadatas[idx]
        images = []
        images.append(pil_img2rgb(Image.open(os.path.join(image_path, metadata['image']))))
        image = pil_img2rgb(Image.open(os.path.join(image_path, metadata['image'])))
        prompt = metadata['instruction']
        os.makedirs(os.path.join(output_dir, metadata['category']), exist_ok=True)
        outpath = os.path.join(output_dir, metadata['category'], f"{metadata['index']}.png")
        #print(f"GPU {rank} 处理提示词 {idx - start + 1}/{end - start}: '{prompt}'")

        #if os.path.exists(outpath):
        #    print(f"GPU {rank} 跳过 {prompt} 生成的图像")
        #    continue
        if not think:
            edited_image = editing_generate(
                model,
                vae_model,
                tokenizer,
                special_token_ids,
                vae_image_transform,
                vit_image_transform,
                image,
                prompt,
                num_timesteps,
                timestep_shift,
                cfg_text_scale,
                cfg_img_scale,
                cfg_interval,
                cfg_renorm_min,
                cfg_renorm_type,
                flow_solver,
                negative_prompt,
                mm_attn_num_layer_type=mm_attn_num_layer_type,

            )
        else:
            edited_image,think_output = editing_generate_with_thinking(
                model,
                vae_model,
                tokenizer,
                special_token_ids,
                vae_image_transform,
                vit_image_transform,
                image,
                prompt,
                num_timesteps,
                timestep_shift,
                cfg_text_scale,
                cfg_img_scale,
                cfg_interval,
                cfg_renorm_min,
                cfg_renorm_type,
                flow_solver,
                negative_prompt,
                max_length=max_length,
                mm_attn_num_layer_type=mm_attn_num_layer_type,

                **kwargs,
            )
        edited_image = edited_image.crop(edited_image.getbbox())
        edited_image.save(outpath)
        txt_output = os.path.join(output_dir, metadata['category'], f"{metadata['index']}.txt")

        if think:
            with open(txt_output, "w", encoding="utf-8") as f:
                f.write(think_output)


def kris_inference(
    model, vae_model, tokenizer, 
    special_token_ids, vae_image_transform, vit_image_transform,
    data_path, output_dir,
    num_timesteps, timestep_shift,
    cfg_text_scale, cfg_img_scale, cfg_interval, cfg_renorm_min, cfg_renorm_type,
    flow_solver, negative_prompt, 
    rank=0, world_size=1, think=False, mm_attn_num_layer_type="min", **kwargs
):
    os.makedirs(output_dir, exist_ok=True)
    
    #metadata_file = os.path.join(data_path, "sampled_final_data.json")
    #metadata_file = os.path.join(data_path, "final_300_data.json")
    
    if data_path.endswith(".json"):
        metadata_file = data_path
        data_path = os.path.dirname(data_path)
    else:
        metadata_file = os.path.join(data_path, "final_300_data.json")
    with open(metadata_file, "r") as f:
        metadatas = json.load(f)

    # Get the directory of data_path
    
    total_metadatas = len(metadatas)

    world_size = args.world_size
    rank = args.rank
    
    # 将待处理的元数据按 GPU 划分
    prompts_per_gpu = (total_metadatas + world_size - 1) // world_size
    start = rank * prompts_per_gpu
    end = min(start + prompts_per_gpu, total_metadatas)
    print(f"GPU {rank}: 处理 {end - start} 个提示词 (索引 {start} 到 {end - 1})")
    image_path = os.path.join(data_path, "KRIS_Bench")

    print("image_path is",image_path)
    rank = args.rank
    world_size = args.world_size

    # 遍历当前 GPU 负责的元数据，生成图像并保存
    for idx in range(start, end):
        metadata = metadatas[idx]
        images = []
        if isinstance(metadata['ori_img'], str):
            images.append(pil_img2rgb(Image.open(os.path.join(image_path, metadata['type'], metadata['ori_img']))))
        else:
            for img_path in metadata['ori_img']:
                images.append(pil_img2rgb(Image.open(os.path.join(image_path, metadata['type'], img_path))))
        prompt = metadata['ins_en']
        os.makedirs(os.path.join(output_dir, metadata['type']), exist_ok=True)
        outpath = os.path.join(output_dir, metadata['type'], f"{metadata['id']}.png")
        print(f"GPU {rank} 处理提示词 {idx - start + 1}/{end - start}: '{prompt}'")

        if os.path.exists(outpath):
            print(f"GPU {rank} 跳过 {prompt} 生成的图像")
            continue

        if not think:
            edited_image = editing_generate(
                model,
                vae_model,
                tokenizer,
                special_token_ids,
                vae_image_transform,
                vit_image_transform,
                images,
                prompt,
                num_timesteps,
                timestep_shift,
                cfg_text_scale,
                cfg_img_scale,
                cfg_interval,
                cfg_renorm_min,
                cfg_renorm_type,
                flow_solver,
                negative_prompt,
                mm_attn_num_layer_type=mm_attn_num_layer_type,

            )
        else:
            edited_image,think_output = editing_generate_with_thinking(
                model,
                vae_model,
                tokenizer,
                special_token_ids,
                vae_image_transform,
                vit_image_transform,
                images,
                prompt,
                num_timesteps,
                timestep_shift,
                cfg_text_scale,
                cfg_img_scale,
                cfg_interval,
                cfg_renorm_min,
                cfg_renorm_type,
                flow_solver,
                negative_prompt,
                mm_attn_num_layer_type=mm_attn_num_layer_type,

                **kwargs,
            )
        edited_image = edited_image.crop(edited_image.getbbox())
        edited_image.save(outpath)
        txt_output = os.path.join(output_dir, metadata['type'], f"{metadata['id']}.txt")

        if think:
            with open(txt_output, "w", encoding="utf-8") as f:
                f.write(think_output)

# TODO: need to set output image size
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using CausalFusion model.")
    parser.add_argument("--benchmark", type=str, required=True, choices=["emuedit", "intelligent_bench", "gedit_bench", "imgedit_bench","rise_bench","kris_bench"])
    parser.add_argument("--data_path", type=str, default="./eval_data/emu_edit_test/data.parquet", 
                    help="Path to parquet file with test data")
    parser.add_argument("--rank", type=int, default=0, help="ID of the current shard (0-based)")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of shards")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--load_from", type=str, required=True, help="Path to the checkpoint directory to resume from.")
    parser.add_argument("--vlm_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--vgen_model_path", type=str, default="./pretrained_weights/Wan2.2-TI2V-5B-NaViT")
    parser.add_argument("--vae_path", type=str, default="./pretrained_weights/Wan2.2-TI2V-5B-NaViT/Wan2.2_VAE.pth")
    parser.add_argument("--pre_t5_context_path", type=str, default="./pretrained_weights/Wan2.2-TI2V-5B-NaViT/aphotoof_t5_context.pt")
    parser.add_argument("--llm_qk_norm", action="store_true")
    parser.add_argument("--mm_attn_qk_norm", action="store_true")
    parser.add_argument("--standalone_num_vlm_layers", type=int, required=True)
    parser.add_argument("--tie_word_embeddings", action="store_true")
    parser.add_argument("--flow_solver", type=str, default="unipc", choices=["unipc", "naive", "dpm++"])
    parser.add_argument("--mm_attn_num_layer_type", type=str, default="min", choices=["min", "max"])

    parser.add_argument("--vae_transform_sizes", nargs='+', type=int, default=(512, 256, 16), help="in the form of (max_size, min_size, stride)")
    parser.add_argument("--vit_transform_sizes", nargs='+', type=int, default=(532, 224, 28), help="in the form of (max_size, min_size, stride)")
    parser.add_argument("--max_pixels", type=int, default=1806336)
    parser.add_argument("--vit_image_mean", nargs='+', type=float, default=[0.48145466, 0.4578275, 0.40821073])
    parser.add_argument("--vit_image_std", nargs='+', type=float, default=[0.26862954, 0.26130258, 0.27577711])
    parser.add_argument("--vae_image_mean", nargs='+', type=float, default=[0.5, 0.5, 0.5])
    parser.add_argument("--vae_image_std", nargs='+', type=float, default=[0.5, 0.5, 0.5])
    parser.add_argument("--task", type=str, default="ti2v-5B")
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--timestep_shift", type=int, default=4)
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--cfg_img_scale", type=float, default=2.0)
    parser.add_argument('--cfg_interval', type=float, nargs='+', default=[0.0, 1.0])
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument("--cfg_renorm_type", type=str, default="noop", choices=["text_channel", "global", "channel", "noop"])
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--use_magic_negative_prompt", action="store_true")
    parser.add_argument("--use_vgen_for_mm_attn", action="store_true")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--temperature", type=int, default=0.3)
    parser.add_argument("--do_sample", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed, deterministic=False)

    mm_attn_num_layer_type = args.mm_attn_num_layer_type

    
    vae_image_transform = ImageTransform(
        max_image_size=args.vae_transform_sizes[0],
        min_image_size=args.vae_transform_sizes[1],
        image_stride=args.vae_transform_sizes[2],
        max_pixels=args.max_pixels, 
        image_mean=args.vae_image_mean,
        image_std=args.vae_image_std
    )
    vit_image_transform = ImageTransform(
        max_image_size=args.vit_transform_sizes[0],
        min_image_size=args.vit_transform_sizes[1],
        image_stride=args.vit_transform_sizes[2],
        max_pixels=args.max_pixels, 
        image_mean=args.vit_image_mean,
        image_std=args.vit_image_std
    )
    # vae_model, model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    vae_model, model, tokenizer, special_token_ids = load_model_and_tokenizer(args)

    if args.use_magic_negative_prompt:
        # use en because t2i uses en
       # negative_prompt = "inconsistent, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    else:
        negative_prompt = ""

    if args.benchmark == "emuedit":
        emuedit_inference(model, vae_model, tokenizer, 
                          special_token_ids, vae_image_transform, vit_image_transform,
                          args.data_path, args.output_dir,
                          args.num_timesteps, args.timestep_shift,
                          args.cfg_text_scale, args.cfg_img_scale, args.cfg_interval, args.cfg_renorm_min, args.cfg_renorm_type,
                          args.flow_solver, negative_prompt,
                          rank=args.rank, world_size=args.world_size, mm_attn_num_layer_type=mm_attn_num_layer_type)
    elif args.benchmark == "gedit_bench":
        gedit_inference(model, vae_model, tokenizer, 
                          special_token_ids, vae_image_transform, vit_image_transform,
                          args.data_path, args.output_dir,
                          args.num_timesteps, args.timestep_shift,
                          args.cfg_text_scale, args.cfg_img_scale, args.cfg_interval, args.cfg_renorm_min, args.cfg_renorm_type,
                          args.flow_solver, negative_prompt,
                          rank=args.rank, world_size=args.world_size, mm_attn_num_layer_type=mm_attn_num_layer_type)
    elif args.benchmark == "imgedit_bench":
        imgedit_inference(model, vae_model, tokenizer, 
                          special_token_ids, vae_image_transform, vit_image_transform,
                          args.data_path, args.output_dir,
                          args.num_timesteps, args.timestep_shift,
                          args.cfg_text_scale, args.cfg_img_scale, args.cfg_interval, args.cfg_renorm_min, args.cfg_renorm_type,
                          args.flow_solver, negative_prompt,
                          rank=args.rank, world_size=args.world_size, mm_attn_num_layer_type=mm_attn_num_layer_type)
    elif args.benchmark == "intelligent_bench":
        intelligent_inference(model, vae_model, tokenizer, 
                          special_token_ids, vae_image_transform, vit_image_transform,
                          args.data_path, args.output_dir,
                          args.num_timesteps, args.timestep_shift,
                          args.cfg_text_scale, args.cfg_img_scale, args.cfg_interval, args.cfg_renorm_min, args.cfg_renorm_type,
                          args.flow_solver, negative_prompt,
                          rank=args.rank, world_size=args.world_size, think=args.think, mm_attn_num_layer_type=mm_attn_num_layer_type, max_length=args.max_length, do_sample=args.do_sample, temperature=args.temperature)    
    elif args.benchmark == "rise_bench":
        rise_inference(model, vae_model, tokenizer, 
                        special_token_ids, vae_image_transform, vit_image_transform,
                        args.data_path, args.output_dir,
                        args.num_timesteps, args.timestep_shift,
                        args.cfg_text_scale, args.cfg_img_scale, args.cfg_interval, args.cfg_renorm_min, args.cfg_renorm_type,
                        args.flow_solver, negative_prompt,
                        rank=args.rank, world_size=args.world_size, think=args.think, max_length=args.max_length, do_sample=args.do_sample, temperature=args.temperature, mm_attn_num_layer_type=mm_attn_num_layer_type)
    elif args.benchmark == "kris_bench":
        kris_inference(model, vae_model, tokenizer, 
                        special_token_ids, vae_image_transform, vit_image_transform,
                        args.data_path, args.output_dir,
                        args.num_timesteps, args.timestep_shift,
                        args.cfg_text_scale, args.cfg_img_scale, args.cfg_interval, args.cfg_renorm_min, args.cfg_renorm_type,
                        args.flow_solver, negative_prompt,
                        rank=args.rank, world_size=args.world_size, think=args.think, max_length=args.max_length, do_sample=args.do_sample, temperature=args.temperature, mm_attn_num_layer_type=mm_attn_num_layer_type)
    else:
        raise NotImplementedError