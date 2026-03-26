# Copyright 2025 Zeyu Wang & Zilong Chen.
# SPDX-License-Identifier: Apache-2.0

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
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

from transformers import set_seed
from data.transforms import ImageTransform
from modeling.lightfusion.qwen25vl_navit_fusion import NaiveCache
from eval.utils import load_model_and_tokenizer
from data.transforms import ImageTransform


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
):
    past_und_key_values = NaiveCache(model.config.vlm_config.num_hidden_layers)
    past_uni_key_values = NaiveCache(model.vgen_model.num_layers - 1)

    # image context
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

    # cfg text
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

    # cfg image
    cfg_img_past_und_key_values = NaiveCache(model.config.vlm_config.num_hidden_layers)
    cfg_img_past_uni_key_values = NaiveCache(model.vgen_model.num_layers - 1)
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

    # text context
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

    # gen
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = model.visual_gen(
            **generation_input,
            past_und_key_values=past_und_key_values,
            past_uni_key_values=past_uni_key_values,
            flow_solver=flow_solver,
            num_timesteps=num_timesteps,
            timestep_shift=timestep_shift,
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


def gedit_inference(
    model, vae_model, tokenizer, 
    special_token_ids, vae_image_transform, vit_image_transform,
    data_path, output_dir,
    num_timesteps, timestep_shift,
    cfg_text_scale, cfg_img_scale, cfg_interval, cfg_renorm_min, cfg_renorm_type,
    flow_solver, negative_prompt,
    rank=0, world_size=1
):
    os.makedirs(output_dir, exist_ok=True)

    info_path = os.path.join(data_path, 'info.jsonl')
    with open(info_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = lines[rank::world_size]

    for idx, line in enumerate(tqdm(lines)):
        print(f"GPU {rank} processing sample {idx}/{len(lines)}")

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
        input_image = Image.open(source_image_path)

        try:
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
            )

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
    rank=0, world_size=1
):
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

    for idx, (prompt, key, image_path) in enumerate(tqdm(inference_list)):
        print(f"GPU {rank} processing sample {idx}/{len(inference_list)}")
        cur_save_path = os.path.join(output_dir, f"{key}.png")
        if os.path.exists(cur_save_path):
            print(f"GPU {rank=} {key=} '{prompt}' already exists, skipping...")
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
        )

        edited_image.save(cur_save_path)


def main(args):
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
    vae_model, model, tokenizer, special_token_ids = load_model_and_tokenizer(args)

    if args.use_magic_negative_prompt:
        negative_prompt = "Bright tones, overexposed, blurred details, subtitles, style, works, paintings, images, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, messy background, three legs, many people in the background, walking backwards"
    else:
        negative_prompt = ""

    if args.benchmark == "gedit_bench":
        gedit_inference(model, vae_model, tokenizer, 
                          special_token_ids, vae_image_transform, vit_image_transform,
                          args.data_path, args.output_dir,
                          args.num_timesteps, args.timestep_shift,
                          args.cfg_text_scale, args.cfg_img_scale, args.cfg_interval, args.cfg_renorm_min, args.cfg_renorm_type,
                          args.flow_solver, negative_prompt,
                          rank=args.rank, world_size=args.world_size)
    elif args.benchmark == "imgedit_bench":
        imgedit_inference(model, vae_model, tokenizer, 
                          special_token_ids, vae_image_transform, vit_image_transform,
                          args.data_path, args.output_dir,
                          args.num_timesteps, args.timestep_shift,
                          args.cfg_text_scale, args.cfg_img_scale, args.cfg_interval, args.cfg_renorm_min, args.cfg_renorm_type,
                          args.flow_solver, negative_prompt,
                          rank=args.rank, world_size=args.world_size)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using LightFusion model.")
    parser.add_argument("--benchmark", type=str, required=True, choices=["gedit_bench", "imgedit_bench"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--load_from", type=str, required=True, help="Path to the checkpoint directory to resume from.")
    parser.add_argument("--vlm_path", type=str, default="/path/to/hf/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--vgen_model_path", type=str, default="/path/to/weights/Wan2.2-TI2V-5B")
    parser.add_argument("--vae_path", type=str, default="/path/to/weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    parser.add_argument("--pre_t5_context_path", type=str, default="/path/to/weights/lightfusion/pre_t5_context.pt")
    parser.add_argument("--llm_qk_norm", action="store_true")
    parser.add_argument("--mm_attn_qk_norm", action="store_true")
    parser.add_argument("--standalone_num_vlm_layers", type=int, default=0)
    parser.add_argument("--tie_word_embeddings", action="store_true")
    parser.add_argument("--flow_solver", type=str, default="unipc", choices=["unipc", "naive", "dpm++"])

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
    parser.add_argument("--cfg_text_scale", type=float, default=3.0)
    parser.add_argument("--cfg_img_scale", type=float, default=1.0)
    parser.add_argument('--cfg_interval', type=int, nargs='+', default=[0, 1])
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument("--cfg_renorm_type", type=str, default="noop", choices=["text_channel", "global", "channel", "noop"])
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--use_magic_negative_prompt", action="store_true")
    parser.add_argument("--use_vgen_for_mm_attn", action="store_true")

    parser.add_argument("--rank", type=int, default=None, help="global rank")
    parser.add_argument("--world_size", type=int, default=None, help="world size")
    args = parser.parse_args()

    set_seed(args.seed, deterministic=False)
    main(args)