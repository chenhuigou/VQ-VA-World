# Copyright 2025 Zeyu Wang & Zilong Chen.
# SPDX-License-Identifier: Apache-2.0

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
import copy
import csv
import json
import argparse
import imageio
import random
from safetensors.torch import load_file

import torch
import torchvision
from torchvision.utils import make_grid
from transformers import set_seed
from eval.utils import load_model_and_tokenizer

from PIL import Image
from modeling.lightfusion.qwen25vl_navit_fusion import NaiveCache
import copy


def generate_image(args, model, vae_model, prompt, negative_prompt, tokenizer, special_token_ids, flow_solver, is_image=True):
    past_und_key_values = NaiveCache(model.config.vlm_config.num_hidden_layers)
    past_uni_key_values = NaiveCache(model.vgen_model.num_layers - 1)

    # text context
    generation_input, new_und_lens, new_uni_lens, new_rope = model.prepare_prompts(
        curr_und_kvlens=[0] * args.batch_size,
        curr_uni_kvlens=[0] * args.batch_size,
        curr_rope=[0] * args.batch_size,
        prompts=[prompt] * args.batch_size,
        tokenizer=tokenizer,
        special_token_ids=special_token_ids,
    )

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_und_key_values, past_uni_key_values = model.forward_cache_update_text(**generation_input, past_und_key_values=past_und_key_values, past_uni_key_values=past_uni_key_values)

    generation_input = model.prepare_vae_latent(
        curr_und_kvlens=new_und_lens,
        curr_uni_kvlens=new_uni_lens,
        curr_rope=new_rope,
        video_sizes=[args.sizes] * args.batch_size,
        special_token_ids=special_token_ids,
    )

    # cfg
    cfg_past_und_key_values = NaiveCache(model.config.vlm_config.num_hidden_layers)
    cfg_past_uni_key_values = NaiveCache(model.vgen_model.num_layers - 1)

    cfg_generation_input, cfg_new_und_lens, cfg_new_uni_lens, cfg_new_rope = model.prepare_prompts(
        curr_und_kvlens=[0] * args.batch_size,
        curr_uni_kvlens=[0] * args.batch_size,
        curr_rope=[0] * args.batch_size,
        prompts=[negative_prompt] * args.batch_size,
        tokenizer=tokenizer,
        special_token_ids=special_token_ids,
    )

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_past_und_key_values, cfg_past_uni_key_values = model.forward_cache_update_text(**cfg_generation_input, past_und_key_values=cfg_past_und_key_values, past_uni_key_values=cfg_past_uni_key_values)

    cfg_generation_input = model.prepare_vae_latent_cfg(
        curr_und_kvlens=cfg_new_und_lens,
        curr_uni_kvlens=cfg_new_uni_lens,
        curr_rope=cfg_new_rope,
        video_sizes=[args.sizes] * args.batch_size,
    )

    # gen
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = model.visual_gen(
            **generation_input,
            past_und_key_values=past_und_key_values,
            past_uni_key_values=past_uni_key_values,
            flow_solver=flow_solver,
            num_timesteps=args.num_timesteps,
            timestep_shift=args.timestep_shift,
            cfg_text_scale=args.cfg_scale,
            cfg_renorm_min=args.cfg_renorm_min,
            cfg_renorm_type=args.cfg_renorm_type,
            cfg_interval=args.cfg_interval,
            cfg_text_packed_und_query_indexes=cfg_generation_input["cfg_packed_und_query_indexes"],
            cfg_text_packed_uni_query_indexes=cfg_generation_input["cfg_packed_uni_query_indexes"],
            cfg_text_past_und_key_values=cfg_past_und_key_values,
            cfg_text_past_uni_key_values=cfg_past_uni_key_values,
            cfg_text_key_values_lens_und=cfg_generation_input["cfg_key_values_lens_und"],
            cfg_text_key_values_lens_uni=cfg_generation_input["cfg_key_values_lens_uni"],
            cfg_text_packed_und_key_value_indexes=cfg_generation_input["cfg_packed_und_key_value_indexes"],
            cfg_text_packed_uni_key_value_indexes=cfg_generation_input["cfg_packed_uni_key_value_indexes"],
            cfg_text_packed_position_ids=cfg_generation_input["cfg_packed_position_ids"],
        )

    # save image
    image_list = []
    for latent in unpacked_latent:
        image = vae_model.decode([latent])[0]
        image = image.squeeze(1)
        tmpimage = ((image * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        
        tmpimage = Image.fromarray(tmpimage)
        image_list.append(tmpimage)
    return image_list


def geneval_generate_image(args, model, vae_model, tokenizer, special_token_ids, flow_solver, negative_prompt):
    with open(args.metadata_file, "r", encoding="utf-8") as fp:
        metadatas = [json.loads(line) for line in fp]
    total_metadatas = len(metadatas)
    
    prompts_per_gpu = (total_metadatas + args.world_size - 1) // args.world_size
    start = args.rank * prompts_per_gpu
    end = min(start + prompts_per_gpu, total_metadatas)
    print(f"GPU {args.rank}: Processing {end - start} prompts (indices {start} to {end - 1})")

    for idx in range(start, end):
        metadata = metadatas[idx]
        outpath = os.path.join(args.output_dir, f"{idx:0>5}")
        os.makedirs(outpath, exist_ok=True)
        prompt = metadata['prompt']
        print(f"GPU {args.rank} processing prompt {idx - start + 1}/{end - start}: '{prompt}'")


        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        flag = True
        for sample_idx in range(args.num_images):
            if not os.path.exists(os.path.join(sample_path, f"{sample_idx:05}.png")):
                flag = False
                break
        if flag:
            print(f"GPU {args.rank} skipping generation for prompt: {prompt}")
            continue

        with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
            json.dump(metadata, fp)

        image_list = []
        for i in range(args.num_images // args.batch_size):
            tmp_image_list = generate_image(args, model, vae_model, prompt, negative_prompt, tokenizer, special_token_ids, flow_solver)
            image_list.extend(tmp_image_list)

        sample_count = 0
        for sample in image_list:
            sample = sample.crop(sample.getbbox())
            sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
            sample_count += 1


def dpgbench_generate_image(args, model, vae_model, tokenizer, special_token_ids, flow_solver, negative_prompt):
    all_prompt_filenames = [f"{args.metadata_file}/{filename}" for filename in os.listdir(args.metadata_file) if filename.endswith("txt")]
    all_prompts = []
    for filename in all_prompt_filenames:
        with open(filename, "r") as f:
            prompt = f.read().strip()
            all_prompts.append(prompt)
    num_samples = len(all_prompts)

    prompts_per_gpu = (num_samples + args.world_size - 1) // args.world_size
    start = args.rank * prompts_per_gpu
    end = min(start + prompts_per_gpu, num_samples)
    print(f"GPU {args.rank}: Processing {end - start} prompts (indices {start} to {end - 1})")

    for idx in range(start, end):
        prompt = all_prompts[idx]
        to_save_filename = all_prompt_filenames[idx].split('/')[-1].split('.')[0]
        outpath = os.path.join(args.output_dir, f"{to_save_filename}.png")
        print(f"GPU {args.rank} processing prompt {idx - start + 1}/{end - start}: '{prompt}'")

        if os.path.exists(outpath):
            print(f"GPU {args.rank} skipping generation for prompt: {prompt}")
            continue

        image_list = []
        assert args.num_images == 4
        assert args.batch_size == 1
        for i in range(args.num_images // args.batch_size):
            tmp_image_list = generate_image(args, model, vae_model, prompt, negative_prompt, tokenizer, special_token_ids, flow_solver)
            image_list.extend(tmp_image_list)

        # Create a new blank image for the 2x2 grid
        width, height = args.sizes[1], args.sizes[2]
        grid_width = width * 2
        grid_height = height * 2
        combined_image = Image.new('RGB', (grid_width, grid_height))
        image_list = [image.crop(image.getbbox()) for image in image_list]

        # Paste images into the grid
        combined_image.paste(image_list[0], (0, 0))
        combined_image.paste(image_list[1], (width, 0))
        combined_image.paste(image_list[2], (0, height))
        combined_image.paste(image_list[3], (width, height))

        combined_image.save(outpath)


def main(args):
    flow_solver = args.flow_solver
    os.makedirs(args.output_dir, exist_ok=True)
    vae_model, model, tokenizer, special_token_ids = load_model_and_tokenizer(args)
    if args.use_magic_negative_prompt:
        negative_prompt = "Bright tones, overexposed, blurred details, subtitles, style, works, paintings, images, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, messy background, three legs, many people in the background, walking backwards"
    else:
        negative_prompt = ""

    if args.benchmark == "geneval":
        geneval_generate_image(args, model, vae_model, tokenizer, special_token_ids, flow_solver, negative_prompt)
    elif args.benchmark == "dpgbench":
        dpgbench_generate_image(args, model, vae_model, tokenizer, special_token_ids, flow_solver, negative_prompt)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using LightFusion model.")
    parser.add_argument("--benchmark", type=str, required=True, choices=["geneval", "dpgbench"])
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument("--metadata_file", type=str, required=True, help="JSONL file containing lines of metadata for each prompt")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
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

    parser.add_argument("--sizes", nargs='+', type=int, default=(1, 480, 832))
    parser.add_argument("--task", type=str, default="ti2v-5B")
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--timestep_shift", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
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