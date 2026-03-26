# Copyright 2025 Zeyu Wang & Zilong Chen.
# SPDX-License-Identifier: Apache-2.0

import os
import yaml
from safetensors.torch import load_file

import torch
from data.data_utils import pil_img2rgb
from modeling.lightfusion import (
    LightFusionConfig, LightFusion, Qwen2_5_VL_Fusion_Config, Qwen2_5_VL_Fusion_ForConditionalGeneration
)

from modeling.qwen2_5_vl import Qwen2_5_VLProcessor
from modeling.wan22_modules import WanModel, Wan2_2_VAE
from modeling.wan22_modules.configs import WAN_CONFIGS
from train.fsdp_utils import FSDPCheckpoint
from train.train_utils import create_logger
from data.transforms import ImageTransform

# TODO: check if need to reduce memory usage
def load_model_and_tokenizer(args):
    vae_model = Wan2_2_VAE(vae_pth=args.vae_path)
    vgen_model = WanModel.from_pretrained(args.vgen_model_path, block_type="navit")
    vgen_model.convert_conv2d_to_linear()
    vgen_model.cuda().eval().requires_grad_(False)
    vgen_config = WAN_CONFIGS[getattr(args, "task", "ti2v-5B")]
    vgen_config.z_dim = vae_model.model.z_dim

    vlm_config = Qwen2_5_VL_Fusion_Config.from_pretrained(args.vlm_path)
    vlm_config.text_config.layer_module = "Qwen2_5_VLDecoderLayer"
    vlm_config.text_config.qk_norm = args.llm_qk_norm
    vlm_config.text_config.tie_word_embeddings = args.tie_word_embeddings
    vlm_config.text_config.vgen_hidden_size = vgen_model.dim
    vlm_config.text_config.standalone_num_vlm_layers = args.standalone_num_vlm_layers
    vlm_config.text_config.vgen_num_hidden_layers = vgen_model.num_layers
    if args.use_vgen_for_mm_attn:
        vlm_config.text_config.mm_attn_hidden_size = vgen_model.dim
        vlm_config.text_config.mm_attn_num_attention_heads = vgen_model.num_heads
        vlm_config.text_config.mm_attn_num_key_value_heads = vgen_model.num_heads
    else:
        vlm_config.text_config.mm_attn_hidden_size = vlm_config.hidden_size
        vlm_config.text_config.mm_attn_num_attention_heads = vlm_config.num_attention_heads
        vlm_config.text_config.mm_attn_num_key_value_heads = vlm_config.num_key_value_heads
    vlm_config.text_config.mm_attn_qk_norm = args.mm_attn_qk_norm
    vlm_config.text_config.mm_attn_rms_norm_eps = vlm_config.rms_norm_eps
    vlm_config.text_config.mm_attn_num_layer_type = args.mm_attn_num_layer_type
    vlm_config.vision_config.torch_dtype = "bfloat16"
    vision_language_model = Qwen2_5_VL_Fusion_ForConditionalGeneration.from_pretrained(args.vlm_path, config=vlm_config, torch_dtype=torch.bfloat16)
    vision_language_model.zero_init_mm_attn()
    vision_language_model.cuda().eval()

    config = LightFusionConfig(
        vlm_config=vlm_config,
        vgen_config=vgen_config,
        pre_t5_context_path=args.pre_t5_context_path,
    )
    model = LightFusion(vision_language_model, vgen_model, config)
    model.cuda().eval()

    processor = Qwen2_5_VLProcessor.from_pretrained(args.vlm_path)
    tokenizer = processor.tokenizer
    special_token_ids = dict(
        bos_token_id=tokenizer.convert_tokens_to_ids('<|im_start|>'), 
        eos_token_id=tokenizer.convert_tokens_to_ids('<|im_end|>'), 
        sov_token_id=tokenizer.convert_tokens_to_ids('<|vision_start|>'), 
        eov_token_id=tokenizer.convert_tokens_to_ids('<|vision_end|>'), 
    )

    if args.use_ema:
        ema_state_dict_path = os.path.join(args.load_from, f"ema.safetensors")
        ema_state_dict = load_file(ema_state_dict_path, device="cpu")
        msg = model.load_state_dict(ema_state_dict, strict=False)
    else:
        model_state_dict_path = os.path.join(args.load_from, f"model.safetensors")
        model_state_dict = load_file(model_state_dict_path, device="cpu")
        msg = model.load_state_dict(model_state_dict, strict=False)
    print(f"load state_dict message: {msg}")
    model.cuda().eval()

    return vae_model, model, tokenizer, special_token_ids