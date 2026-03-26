# Copyright 2025 Zeyu Wang & Zilong Chen.
# SPDX-License-Identifier: Apache-2.0

import os
import gradio as gr
import numpy as np
import torch
import random
import argparse
import copy
from PIL import Image

from transformers import set_seed
from data.transforms import ImageTransform
from modeling.lightfusion.qwen25vl_navit_fusion import NaiveCache
from eval.utils import load_model_and_tokenizer

MAX_SEED = 65536


def generate_image_t2i(
    model,
    vae_model,
    tokenizer,
    special_token_ids,
    prompt,
    negative_prompt,
    flow_solver,
    num_timesteps,
    timestep_shift,
    cfg_text_scale,
    cfg_interval,
    cfg_renorm_min,
    cfg_renorm_type,
    video_sizes,
    batch_size=1,
):
    """
    Text-to-Image generation function.
    """
    past_und_key_values = NaiveCache(model.config.vlm_config.num_hidden_layers)
    past_uni_key_values = NaiveCache(model.vgen_model.num_layers - 1)

    generation_input, new_und_lens, new_uni_lens, new_rope = model.prepare_prompts(
        curr_und_kvlens=[0] * batch_size,
        curr_uni_kvlens=[0] * batch_size,
        curr_rope=[0] * batch_size,
        prompts=[prompt] * batch_size,
        tokenizer=tokenizer,
        special_token_ids=special_token_ids,
    )

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_und_key_values, past_uni_key_values = model.forward_cache_update_text(
            **generation_input, 
            past_und_key_values=past_und_key_values, 
            past_uni_key_values=past_uni_key_values
        )

    generation_input = model.prepare_vae_latent(
        curr_und_kvlens=new_und_lens,
        curr_uni_kvlens=new_uni_lens,
        curr_rope=new_rope,
        video_sizes=[video_sizes] * batch_size,
        special_token_ids=special_token_ids,
    )

    # Prepare negative prompt for CFG
    cfg_past_und_key_values = NaiveCache(model.config.vlm_config.num_hidden_layers)
    cfg_past_uni_key_values = NaiveCache(model.vgen_model.num_layers - 1)

    cfg_generation_input, cfg_new_und_lens, cfg_new_uni_lens, cfg_new_rope = model.prepare_prompts(
        curr_und_kvlens=[0] * batch_size,
        curr_uni_kvlens=[0] * batch_size,
        curr_rope=[0] * batch_size,
        prompts=[negative_prompt] * batch_size,
        tokenizer=tokenizer,
        special_token_ids=special_token_ids,
    )

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_past_und_key_values, cfg_past_uni_key_values = model.forward_cache_update_text(
            **cfg_generation_input, 
            past_und_key_values=cfg_past_und_key_values, 
            past_uni_key_values=cfg_past_uni_key_values
        )

    cfg_generation_input = model.prepare_vae_latent_cfg(
        curr_und_kvlens=cfg_new_und_lens,
        curr_uni_kvlens=cfg_new_uni_lens,
        curr_rope=cfg_new_rope,
        video_sizes=[video_sizes] * batch_size,
    )

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = model.visual_gen(
            **generation_input,
            past_und_key_values=past_und_key_values,
            past_uni_key_values=past_uni_key_values,
            flow_solver=flow_solver,
            num_timesteps=num_timesteps,
            timestep_shift=timestep_shift,
            cfg_text_scale=cfg_text_scale,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            cfg_interval=cfg_interval,
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

    # Decode and convert to PIL images
    image_list = []
    for latent in unpacked_latent:
        image = vae_model.decode([latent])[0]
        image = image.squeeze(1)
        tmpimage = ((image * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        tmpimage = Image.fromarray(tmpimage)
        image_list.append(tmpimage)
    
    return image_list[0] if len(image_list) == 1 else image_list


def generate_image_i2i(
    model,
    vae_model,
    tokenizer,
    special_token_ids,
    vae_image_transform,
    vit_image_transform,
    image,
    prompt,
    negative_prompt,
    flow_solver,
    num_timesteps,
    timestep_shift,
    cfg_text_scale,
    cfg_img_scale,
    cfg_interval,
    cfg_renorm_min,
    cfg_renorm_type,
):
    """
    Image-to-Image editing function.
    """
    past_und_key_values = NaiveCache(model.config.vlm_config.num_hidden_layers)
    past_uni_key_values = NaiveCache(model.vgen_model.num_layers - 1)

    # Process input image with VAE encoder
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
        past_und_key_values, past_uni_key_values = model.forward_cache_update_vae(
            **generation_input, 
            vae_model=vae_model, 
            past_und_key_values=past_und_key_values, 
            past_uni_key_values=past_uni_key_values
        )

    # Process input image with ViT encoder
    generation_input, new_und_lens, new_uni_lens, new_rope = model.prepare_vit_images(
        curr_und_kvlens=new_und_lens,
        curr_uni_kvlens=new_uni_lens,
        curr_rope=new_rope,
        images=[image],
        transforms=vit_image_transform,
        special_token_ids=special_token_ids,
    )

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_und_key_values, past_uni_key_values = model.forward_cache_update_vit(
            **generation_input, 
            past_und_key_values=past_und_key_values, 
            past_uni_key_values=past_uni_key_values
        )
    
    # Prepare CFG for text (negative prompt)
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
        cfg_text_past_und_key_values, cfg_text_past_uni_key_values = model.forward_cache_update_text(
            **cfg_text_generation_input, 
            past_und_key_values=cfg_text_past_und_key_values, 
            past_uni_key_values=cfg_text_past_uni_key_values
        )

    cfg_text_generation_input = model.prepare_vae_latent_cfg(
        curr_und_kvlens=cfg_text_new_und_lens,
        curr_uni_kvlens=cfg_text_new_uni_lens,
        curr_rope=cfg_text_new_rope,
        video_sizes=[video_sizes],
    )

    # Prepare CFG for image (text-only condition)
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
        cfg_img_past_und_key_values, cfg_img_past_uni_key_values = model.forward_cache_update_text(
            **cfg_img_generation_input, 
            past_und_key_values=cfg_img_past_und_key_values, 
            past_uni_key_values=cfg_img_past_uni_key_values
        )

    cfg_img_generation_input = model.prepare_vae_latent_cfg(
        curr_und_kvlens=cfg_img_new_und_lens,
        curr_uni_kvlens=cfg_img_new_uni_lens,
        curr_rope=cfg_img_new_rope,
        video_sizes=[video_sizes],
    )

    # Process prompt
    generation_input, new_und_lens, new_uni_lens, new_rope = model.prepare_prompts(
        curr_und_kvlens=new_und_lens,
        curr_uni_kvlens=new_uni_lens,
        curr_rope=new_rope,
        prompts=[prompt],
        tokenizer=tokenizer,
        special_token_ids=special_token_ids,
    )

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_und_key_values, past_uni_key_values = model.forward_cache_update_text(
            **generation_input, 
            past_und_key_values=past_und_key_values, 
            past_uni_key_values=past_uni_key_values
        )

    generation_input = model.prepare_vae_latent(
        curr_und_kvlens=new_und_lens,
        curr_uni_kvlens=new_uni_lens,
        curr_rope=new_rope,
        video_sizes=[video_sizes],
        special_token_ids=special_token_ids,
    )

    # Generate edited image
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

    # Decode to image
    image = vae_model.decode([unpacked_latent[0]])[0]
    image = image.squeeze(1)
    image = ((image * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    
    return image


def infer(prompt, image=None, seed=42, randomize_seed=False, 
          cfg_text_scale=4.0, cfg_img_scale=2.0, num_timesteps=50,
          cfg_interval_start=0.0, cfg_interval_end=1.0,
          cfg_renorm_min=0.0, cfg_renorm_type="noop",
          progress=gr.Progress(track_tqdm=True)):
    """
    Unified inference function for both T2I and image editing.
    
    Args:
        prompt (str): Text prompt for generation or editing instruction
        image (PIL.Image.Image, optional): Input image for editing. If None, performs T2I generation
        seed (int): Random seed for reproducibility
        randomize_seed (bool): Whether to randomize the seed
        cfg_text_scale (float): Text CFG guidance scale
        cfg_img_scale (float): Image CFG guidance scale (for editing)
        num_timesteps (int): Number of diffusion timesteps
        cfg_interval_start (float): CFG interval start
        cfg_interval_end (float): CFG interval end
        cfg_renorm_min (float): CFG renormalization minimum
        cfg_renorm_type (str): CFG renormalization type
        progress (gr.Progress): Progress tracker
    
    Returns:
        tuple: (generated_image, seed_used, button_update)
    """
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    set_seed(seed, deterministic=False)
    
    # Prepare cfg_interval as a list
    cfg_interval = [cfg_interval_start, cfg_interval_end]
    
    # Check if image is provided for editing or T2I generation
    if image is not None:
        # Image Editing Mode
        print(f"Running Image Editing with prompt: {prompt}")
        image = image.convert("RGB")
        generated_image = generate_image_i2i(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            special_token_ids=special_token_ids,
            vae_image_transform=vae_image_transform,
            vit_image_transform=vit_image_transform,
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            flow_solver=args.flow_solver,
            num_timesteps=num_timesteps,
            timestep_shift=args.timestep_shift,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
        )
    else:
        # Text-to-Image Generation Mode
        print(f"Running T2I Generation with prompt: {prompt}")
        generated_image = generate_image_t2i(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            special_token_ids=special_token_ids,
            prompt=prompt,
            negative_prompt=negative_prompt,
            flow_solver=args.flow_solver,
            num_timesteps=num_timesteps,
            timestep_shift=args.timestep_shift,
            cfg_text_scale=cfg_text_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            video_sizes=args.sizes,
            batch_size=args.batch_size,
        )
    
    return generated_image, seed, gr.Button(visible=True)

def infer_example(input_image, prompt):
    """Wrapper for examples without progress tracking"""
    image, seed, _ = infer(prompt, input_image)
    return image, seed

css = """
#col-container {
    margin: 0 auto;
    max-width: 960px;
}
.mode-indicator {
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
    text-align: center;
    font-weight: bold;
}
.t2i-mode {
    background-color: #e3f2fd;
    color: #1976d2;
}
.edit-mode {
    background-color: #fff3e0;
    color: #f57c00;
}
"""

with gr.Blocks() as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# LightBagel: Unified Text-to-Image & Image Editing

This demo supports two modes:
- **Text-to-Image (T2I)**: Leave the image upload empty and enter a text prompt
- **Image Editing**: Upload an image and provide editing instructions

[Paper]() | [Project Page]() | [Arxiv]() | [Code]()
        """)
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Image (Optional - leave empty for T2I generation)", 
                    type="pil"
                )
                
                mode_indicator = gr.HTML(
                    '<div class="mode-indicator t2i-mode">Mode: Text-to-Image Generation</div>',
                    visible=True
                )
                
                with gr.Row():
                    prompt = gr.Textbox(
                        label="Prompt",
                        show_label=False,
                        max_lines=3,
                        placeholder="Enter your prompt (T2I: 'A cat sitting on a chair' | Editing: 'Remove the glasses')",
                        container=False,
                    )
                    run_button = gr.Button("Run", scale=0, variant="primary")
                
                with gr.Accordion("Advanced Settings", open=False):
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=42,
                    )
                    
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    
                    with gr.Row():
                        cfg_text_scale = gr.Slider(
                            label="Text CFG Scale",
                            minimum=1.0,
                            maximum=15.0,
                            step=0.1,
                            value=4.0,
                        )
                        
                        cfg_img_scale = gr.Slider(
                            label="Image CFG Scale (for editing)",
                            minimum=1.0,
                            maximum=10.0,
                            step=0.1,
                            value=2.0,
                        )
                    
                    with gr.Row():
                        num_timesteps = gr.Slider(
                            label="Timesteps",
                            minimum=1,
                            maximum=100,
                            value=50,
                            step=1
                        )
                        
                        cfg_renorm_min = gr.Slider(
                            label="CFG Renorm Min",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=0.0,
                        )
                    
                    with gr.Row():
                        cfg_interval_start = gr.Slider(
                            label="CFG Interval Start",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=0.0,
                        )
                        
                        cfg_interval_end = gr.Slider(
                            label="CFG Interval End",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=1.0,
                        )
                    
                    cfg_renorm_type = gr.Dropdown(
                        label="CFG Renorm Type",
                        choices=["noop", "text_channel", "global", "channel"],
                        value="noop",
                    )
                    
            with gr.Column():
                result = gr.Image(label="Result", show_label=False, interactive=False)
                reuse_button = gr.Button("Reuse this image for editing", visible=False)
        
        # Examples section
        with gr.Row():
            gr.Markdown("### Examples")
        
        with gr.Tabs():
            with gr.Tab("Text-to-Image Examples"):
                examples_t2i = gr.Examples(
                    examples=[
                        [None, "A majestic lion standing on a cliff at sunset"],
                        [None, "A futuristic city with flying cars and neon lights"],
                        [None, "A cozy coffee shop interior with warm lighting"],
                    ],
                    inputs=[input_image, prompt],
                    outputs=[result, seed],
                    fn=infer_example,
                    cache_examples=False
                )
            
            with gr.Tab("Image Editing Examples"):
                examples_edit = gr.Examples(
                    examples=[
                        ["examples/image1.png", "Turn the flowers into sunflowers"],
                        ["examples/image2.png", "Make the sky more dramatic with sunset colors"],
                        ["examples/image3.png", "Add snow to the scene"],
                    ],
                    inputs=[input_image, prompt],
                    outputs=[result, seed],
                    fn=infer_example,
                    cache_examples=False
                )
        
        # Update mode indicator based on image upload
        def update_mode(image):
            if image is None:
                return '<div class="mode-indicator t2i-mode">Mode: Text-to-Image Generation</div>'
            else:
                return '<div class="mode-indicator edit-mode">Mode: Image Editing</div>'
        
        input_image.change(
            fn=update_mode,
            inputs=[input_image],
            outputs=[mode_indicator]
        )
        
        # Main inference trigger
        gr.on(
            triggers=[run_button.click, prompt.submit],
            fn=infer,
            inputs=[prompt, input_image, seed, randomize_seed, 
                   cfg_text_scale, cfg_img_scale, num_timesteps,
                   cfg_interval_start, cfg_interval_end,
                   cfg_renorm_min, cfg_renorm_type],
            outputs=[result, seed, reuse_button]
        )
        
        # Reuse button functionality
        reuse_button.click(
            fn=lambda image: image,
            inputs=[result],
            outputs=[input_image]
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using LightFusion model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)

    # Model paths
    parser.add_argument("--load_from", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--vlm_path", type=str, default="/path/to/hf/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--vgen_model_path", type=str, default="/path/to/weights/Wan2.2-TI2V-5B")
    parser.add_argument("--vae_path", type=str, default="/path/to/weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    parser.add_argument("--pre_t5_context_path", type=str, default="/path/to/weights/lightfusion/pre_t5_context.pt")
    
    # Model config
    parser.add_argument("--llm_qk_norm", action="store_true")
    parser.add_argument("--standalone_num_vlm_layers", type=int, required=True)
    parser.add_argument("--tie_word_embeddings", action="store_true")
    parser.add_argument("--flow_solver", type=str, default="unipc", choices=["unipc", "naive", "dpm++"])
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--use_vgen_for_cross_attn", action="store_true")

    # Image transforms
    parser.add_argument("--vae_transform_sizes", nargs='+', type=int, default=(512, 256, 16))
    parser.add_argument("--vit_transform_sizes", nargs='+', type=int, default=(532, 224, 28))
    parser.add_argument("--max_pixels", type=int, default=1806336)
    parser.add_argument("--vit_image_mean", nargs='+', type=float, default=[0.48145466, 0.4578275, 0.40821073])
    parser.add_argument("--vit_image_std", nargs='+', type=float, default=[0.26862954, 0.26130258, 0.27577711])
    parser.add_argument("--vae_image_mean", nargs='+', type=float, default=[0.5, 0.5, 0.5])
    parser.add_argument("--vae_image_std", nargs='+', type=float, default=[0.5, 0.5, 0.5])
    
    # Generation config
    parser.add_argument("--sizes", nargs='+', type=int, default=(1, 480, 832))
    parser.add_argument("--task", type=str, default="ti2v-5B")
    parser.add_argument("--timestep_shift", type=int, default=4)
    parser.add_argument("--use_magic_negative_prompt", action="store_true")
    
    args = parser.parse_args()

    set_seed(args.seed, deterministic=False)
    
    # Initialize transforms
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
    
    # Load model
    print("Loading model...")
    vae_model, model, tokenizer, special_token_ids = load_model_and_tokenizer(args)
    print("Model loaded successfully!")

    # Set negative prompt
    if args.use_magic_negative_prompt:
        negative_prompt = "Bright tones, overexposed, blurred details, subtitles, style, works, paintings, images, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, messy background, three legs, many people in the background, walking backwards"
    else:
        negative_prompt = ""

    # Launch demo
    demo.launch(server_name="[::]", server_port=9388, share=True, root_path="/pMV7CGB9vvIK9EwnsRFc8cKMQybFUQepA")
