# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 Zeyu Wang & Zilong Chen.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple, Optional

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from data.data_utils import (
    qwen2_5_vl_patchify, get_qwen2_5_vl_mrope_index
)
from .qwen25vl_navit_fusion import NaiveCache

from ..wan22_modules import wan_sinusoidal_embedding_1d
from modeling.wan22_modules.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from modeling.wan22_modules.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class LightFusionConfig(PretrainedConfig):
    def __init__(
        self,
        vlm_config=None,
        vgen_config=None,
        visual_und=True,
        visual_gen=True,
        pre_t5_context_path=None,
        timestep_shift=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vlm_config = vlm_config
        self.vgen_config = vgen_config
        self.visual_und = visual_und
        self.visual_gen = visual_gen
        self.pre_t5_context_path = pre_t5_context_path
        self.timestep_shift = timestep_shift
        if visual_gen:
            assert pre_t5_context_path is not None


class LightFusion(PreTrainedModel):
    config_class = LightFusionConfig
    base_model_prefix = 'lightfusion'

    def __init__(self, vision_language_model, vgen_model, config: LightFusionConfig):
        super().__init__(config)    
        self.vision_language_model = vision_language_model
        self.vgen_model = vgen_model
        self.hidden_size = config.vlm_config.hidden_size
        self.num_heads = config.vlm_config.num_attention_heads
        self.vgen_num_heads = self.vgen_model.num_heads
        self.cross_attn_num_attention_heads = getattr(config.vlm_config.text_config, "mm_attn_num_attention_heads", config.vlm_config.num_attention_heads)

        self.vae_stride = config.vgen_config.vae_stride
        self.latent_patch_size = vgen_model.patch_size
        self.latent_downsample = [stride * patch_size for stride, patch_size in zip(self.vae_stride, self.latent_patch_size)]
        self.latent_channel = vgen_model.in_dim
        self.vgen_num_train_timesteps = config.vgen_config.num_train_timesteps
        self.vgen_out_dim = vgen_model.out_dim
        self.vgen_model_cross_att_context = nn.Parameter(
            torch.randn(self.vgen_model.text_len, self.vgen_model.text_dim) * 0.002,
            requires_grad=True
        )
        self.timestep_shift = config.timestep_shift

        self.config = config
        self._init_weights(config.pre_t5_context_path)

    def _init_weights(self, pre_t5_context_path):
        # FIXME need to change this!
        if pre_t5_context_path is None:
            pre_t5_context_path = "/path/to/weights/lightfusion/pre_t5_context.pt"
        pre_extract_t5_context = torch.load(pre_t5_context_path)
        context_len = pre_extract_t5_context.shape[0]
        self.vgen_model_cross_att_context.data[:context_len, :].copy_(pre_extract_t5_context.float())


    def forward(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        sample_lens_und: List[int] = None,
        split_lens_gen: List[int] = None,
        packed_m_position_ids: torch.LongTensor = None,
        nested_attention_masks: List[torch.Tensor]=None,
        # for visual understanding
        und_ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_token_indexes_und: Optional[torch.LongTensor] = None,
        packed_text_indexes_und: Optional[torch.LongTensor] = None,
        vit_image_grid_thws: Optional[torch.IntTensor] = None,
        # for visual generation
        padded_image_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            sample_lens_und: A list of N ints, length of und tokens in each sample in packed_sequence.
            split_lens_gen: A list of N ints, length of gen tokens in each image in packed_sequence.
            packed_m_position_ids: 2-D long tensor, packed multimodal position ids
            nested_attention_masks: A list of N 2-D float tensor,  where 0.0 means attention and 
                -inf means ignore.

            und_ce_loss_indexes: 1-D int tensor, where to compute ce loss in und sequence.
            packed_label_ids: 1-D int tensor, packed label token ids.
            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_token_indexes: 1-D int tensor, packed vit token indexes in sequence.
            packed_vit_token_indexes_und: 1-D int tensor, packed vit token indexes in und sequence.
            packed_text_indexes_und: 1-D int tensor, packed text token indexes in und sequence.
            vit_image_grid_thws: The temporal, height and width of feature shape of each image.

            padded_image_latent: padded image latent from VAE encoder.
            patchified_vae_latent_shapes: A list of (t, h, w) tuples, patchfied latent shapes of each image.
            packed_vae_token_indexes: 1-D int tensor, padded image token indexes in sequence.
            packed_timesteps: 1-D float tensor, flow timesteps. 0 indicates use clean image.
        """

        packed_text_embedding = self.vision_language_model.model.language_model.embed_tokens(packed_text_ids)
        packed_sequence_und = packed_text_embedding

        extra_inputs = {}
        extra_inputs.update(nested_attention_masks=nested_attention_masks)

        if self.config.visual_und and vit_image_grid_thws is not None:
            packed_vit_token_embed = self.vision_language_model.model.visual(packed_vit_tokens, grid_thw=vit_image_grid_thws)
            assert packed_vit_token_embed.shape[0] == packed_vit_token_indexes.shape[0], "vit number of tokens and indexes not matching!"

            packed_sequence_und = packed_text_embedding.new_zeros(
                size=(len(packed_text_indexes_und)+len(packed_vit_token_indexes_und), self.hidden_size)
            )
            packed_sequence_und[packed_text_indexes_und] = packed_text_embedding
            packed_sequence_und[packed_vit_token_indexes_und] = packed_vit_token_embed

        if self.config.visual_gen:
            packed_latent = []
            p, q, r = self.latent_patch_size
            curr_vae_image_index = 0
            for curr_vae_index, (f, h, w) in enumerate(patchified_vae_latent_shapes):
                latent = padded_image_latent[curr_vae_image_index][:, :f*p, :h*q, :w*r].reshape(self.latent_channel, f, p, h, q, w, r)
                curr_vae_image_index += 1
                latent = torch.einsum("cfphqwr->fhwpqrc", latent).reshape(f * h * w, p * q * r * self.latent_channel)
                packed_latent.append(latent)
            assert curr_vae_image_index == len(patchified_vae_latent_shapes), "vae latent shapes not match!"
            packed_latent_clean = torch.cat(packed_latent, dim=0)

            noise = torch.randn_like(packed_latent_clean)
            packed_timesteps = torch.sigmoid(packed_timesteps)
            packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)

            packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
            packed_sequence_gen = self.vgen_model.patch_embedding(packed_latent)

            grid_sizes = torch.stack(
                [torch.tensor((f, h, w), dtype=torch.long) for f, h, w in patchified_vae_latent_shapes])

            with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
                vgen_e = self.vgen_model.time_embedding(
                    wan_sinusoidal_embedding_1d(self.vgen_model.freq_dim,
                                                packed_timesteps * self.vgen_num_train_timesteps).float())
                vgen_e0 = self.vgen_model.time_projection(vgen_e).unflatten(1, (6, self.vgen_model.dim))
                assert vgen_e.dtype == torch.float32 and vgen_e0.dtype == torch.float32, 'time_embedding dtype error'

            vgen_model_cross_att_context = self.vgen_model.text_embedding(self.vgen_model_cross_att_context)
            vgen_kwargs = dict(
                vgen_e=vgen_e0,
                vgen_grid_sizes=grid_sizes,
                vgen_freqs=self.vgen_model.freqs.to(self.device),
                vgen_context=vgen_model_cross_att_context,
            )

        packed_und_token_indexes = packed_text_indexes
        if packed_vit_token_indexes is not None:
            packed_und_token_indexes=torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            packed_und_token_indexes = torch.sort(packed_und_token_indexes)[0]
        extra_inputs.update(
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_vae_token_indexes,
        )
        extra_inputs.update(vgen_kwargs)
        last_hidden_state_und, last_hidden_state_gen = self.vision_language_model(
            vgen_model=self.vgen_model,
            packed_sequence_und=packed_sequence_und,
            packed_sequence_gen=packed_sequence_gen,
            sample_lens=sample_lens,
            split_lens_gen=split_lens_gen,
            sample_lens_und=sample_lens_und,
            packed_position_ids=packed_m_position_ids,
            **extra_inputs,
        )

        mse = None
        if self.config.visual_gen:
            has_mse = packed_timesteps > 0
            packed_mse_preds = self.vgen_model.head(last_hidden_state_gen, vgen_e)
            target = noise - packed_latent_clean
            mse = (packed_mse_preds[has_mse] - target[has_mse]) ** 2

        ce = None
        if und_ce_loss_indexes is not None:
            packed_ce_preds = self.language_model.lm_head(last_hidden_state_und[und_ce_loss_indexes])
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        return dict(mse=mse, ce=ce)


    def prepare_prompts(self, curr_und_kvlens, curr_uni_kvlens, curr_rope, prompts, tokenizer, special_token_ids):
        packed_text_ids = list()
        packed_m_position_ids = [[], [], []]
        text_token_lens = list()
        packed_text_indexes_in_query = list()
        packed_und_text_indexes, packed_uni_text_indexes = list(), list()
        packed_und_key_value_indexes, packed_uni_key_value_indexes = list(), list()

        query_curr = und_curr = uni_curr = 0
        new_und_lens, new_uni_lens, new_rope = list(), list(), list()
        for prompt, curr_und_kvlen, curr_uni_kvlen, curr_position_id in zip(prompts, curr_und_kvlens, curr_uni_kvlens, curr_rope):
            packed_und_key_value_indexes.extend(range(und_curr, und_curr + curr_und_kvlen))
            packed_uni_key_value_indexes.extend(range(uni_curr, uni_curr + curr_uni_kvlen))
            und_curr += curr_und_kvlen
            uni_curr += curr_uni_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [special_token_ids['bos_token_id']] + text_ids + [special_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            for packed_position_ids in packed_m_position_ids:
                packed_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            
            packed_text_indexes_in_query.extend(range(query_curr, query_curr + len(text_ids)))
            packed_und_text_indexes.extend(range(und_curr, und_curr + len(text_ids)))
            packed_uni_text_indexes.extend(range(uni_curr, uni_curr + len(text_ids)))

            new_und_lens.append(curr_und_kvlen + len(text_ids))
            new_uni_lens.append(curr_uni_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            query_curr += len(text_ids)
            und_curr += len(text_ids)
            uni_curr += len(text_ids)

        packed_m_position_ids = [torch.tensor(packed_position_ids, dtype=torch.long) for packed_position_ids in packed_m_position_ids]
        packed_m_position_ids = torch.stack(packed_m_position_ids, dim=0)
        packed_m_position_ids = packed_m_position_ids.unsqueeze(1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_position_ids": packed_m_position_ids,
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_indexes_in_query": torch.tensor(packed_text_indexes_in_query, dtype=torch.int),
            "packed_und_text_indexes": torch.tensor(packed_und_text_indexes, dtype=torch.long),
            "packed_uni_text_indexes": torch.tensor(packed_uni_text_indexes, dtype=torch.long),
            "key_values_lens_und": torch.tensor(curr_und_kvlens, dtype=torch.int),
            "key_values_lens_uni": torch.tensor(curr_uni_kvlens, dtype=torch.int),
            "packed_und_key_value_indexes": torch.tensor(packed_und_key_value_indexes, dtype=torch.long),
            "packed_uni_key_value_indexes": torch.tensor(packed_uni_key_value_indexes, dtype=torch.long),
        }
        for k, v in generation_input.items():
            if isinstance(v, torch.Tensor):
                generation_input[k] = v.to(self.device)

        return generation_input, new_und_lens, new_uni_lens, new_rope


    @torch.no_grad
    def forward_cache_update_text(
        self,
        packed_text_ids: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes_in_query: torch.LongTensor,
        packed_und_text_indexes: torch.LongTensor,
        packed_uni_text_indexes: torch.LongTensor,
        past_und_key_values: NaiveCache,
        past_uni_key_values: NaiveCache,
        key_values_lens_und: torch.IntTensor,
        key_values_lens_uni: torch.IntTensor,
        packed_und_key_value_indexes: torch.LongTensor,
        packed_uni_key_value_indexes: torch.LongTensor,
    ):
        packed_text_embedding = self.vision_language_model.model.language_model.embed_tokens(packed_text_ids)

        output = self.vision_language_model.forward_inference(
            vgen_model=self.vgen_model,
            packed_query_sequence_und=packed_text_embedding,
            packed_query_sequence_gen=None,
            query_lens_und=text_token_lens,
            query_lens_gen=None,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_position_ids,
            packed_und_indexes_in_query=packed_text_indexes_in_query,
            packed_gen_indexes_in_query=None,
            vgen_e=None,
            vgen_grid_sizes=None,
            vgen_freqs=None,
            vgen_context=None,
            packed_und_query_indexes=packed_und_text_indexes,
            packed_uni_query_indexes=packed_uni_text_indexes,
            past_und_key_values=past_und_key_values,
            past_uni_key_values=past_uni_key_values,
            key_values_lens_und=key_values_lens_und,
            key_values_lens_uni=key_values_lens_uni,
            packed_und_key_value_indexes=packed_und_key_value_indexes,
            packed_uni_key_value_indexes=packed_uni_key_value_indexes,
            update_past_key_values=True,
            is_causal=True,
            mode="und",
        )
        past_und_key_values = output.past_und_key_values
        past_uni_key_values = output.past_uni_key_values

        return past_und_key_values, past_uni_key_values


    def prepare_vit_images(self, curr_und_kvlens, curr_uni_kvlens, curr_rope, images, transforms, special_token_ids):
        packed_vit_token_indexes = list()
        vit_image_grid_thws, packed_vit_tokens = list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_m_position_ids = list(), [[], [], []]
        packed_und_indexes_in_query = list()
        packed_und_query_indexes, packed_uni_query_indexes = list(), list()
        packed_und_key_value_indexes, packed_uni_key_value_indexes = list(), list()

        query_curr = und_curr = uni_curr = 0
        new_und_lens, new_uni_lens, new_rope = list(), list(), list()
        for image, curr_und_kvlen, curr_uni_kvlen, curr_position_id in zip(images, curr_und_kvlens, curr_uni_kvlens, curr_rope):
            packed_und_key_value_indexes.extend(range(und_curr, und_curr + curr_und_kvlen))
            packed_uni_key_value_indexes.extend(range(uni_curr, uni_curr + curr_uni_kvlen))
            und_curr += curr_und_kvlen
            uni_curr += curr_uni_kvlen

            packed_text_ids.append(special_token_ids['sov_token_id'])
            packed_text_indexes.append(query_curr)
            packed_und_indexes_in_query.append(query_curr)
            packed_und_query_indexes.append(und_curr)
            packed_uni_query_indexes.append(uni_curr)
            query_curr += 1
            und_curr += 1
            uni_curr += 1

            image_tensor = transforms(image)
            vit_tokens = qwen2_5_vl_patchify(image_tensor.unsqueeze(0), 
                self.config.vlm_config.vision_config.temporal_patch_size, 
                self.config.vlm_config.vision_config.patch_size, 
                self.config.vlm_config.vision_config.spatial_merge_size
            )
            
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0] // (self.config.vlm_config.vision_config.spatial_merge_size ** 2)
            vit_image_grid_thws.append(
                [1,
                 image_tensor.size(1) // self.config.vlm_config.vision_config.patch_size,
                 image_tensor.size(2) // self.config.vlm_config.vision_config.patch_size]
            )
            packed_vit_token_indexes.extend(range(query_curr, query_curr + num_img_tokens))
            packed_und_indexes_in_query.extend(range(query_curr, query_curr + num_img_tokens))
            packed_und_query_indexes.extend(range(und_curr, und_curr + num_img_tokens))
            packed_uni_query_indexes.extend(range(uni_curr, uni_curr + num_img_tokens))
            query_curr += num_img_tokens
            und_curr += num_img_tokens
            uni_curr += num_img_tokens

            packed_text_ids.append(special_token_ids['eov_token_id'])
            packed_text_indexes.append(query_curr)
            packed_und_indexes_in_query.append(query_curr)
            packed_und_query_indexes.append(und_curr)
            packed_uni_query_indexes.append(uni_curr)
            query_curr += 1
            und_curr += 1
            uni_curr += 1
            
            for packed_position_ids in packed_m_position_ids:
                packed_position_ids.append(curr_position_id)
            t_index, h_index, w_index = get_qwen2_5_vl_mrope_index(
                st_idx=curr_position_id+1,
                spatial_merge_size=self.config.vlm_config.vision_config.spatial_merge_size,
                tokens_per_second=self.config.vlm_config.vision_config.tokens_per_second,
                image_grid_thw=[1, 
                                image_tensor.size(1) // self.config.vlm_config.vision_config.patch_size,
                                image_tensor.size(2) // self.config.vlm_config.vision_config.patch_size]
            )
            packed_m_position_ids[0].extend(t_index)
            packed_m_position_ids[1].extend(h_index)
            packed_m_position_ids[2].extend(w_index)
            max_pos_id = max(max(t_index), max(h_index), max(w_index))
            for packed_position_ids in packed_m_position_ids:
                packed_position_ids.append(max_pos_id + 1)

            packed_m_position_ids = [torch.tensor(packed_position_ids, dtype=torch.long) for packed_position_ids in packed_m_position_ids]
            packed_m_position_ids = torch.stack(packed_m_position_ids, dim=0)
            packed_m_position_ids = packed_m_position_ids.unsqueeze(1)
        
            packed_seqlens.append(num_img_tokens + 2)
            new_und_lens.append(curr_und_kvlen + num_img_tokens + 2)
            new_uni_lens.append(curr_uni_kvlen + num_img_tokens + 2)
            new_rope.append(max_pos_id + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_image_grid_thws": torch.tensor(vit_image_grid_thws, dtype=torch.long),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": packed_m_position_ids,
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_und_indexes_in_query": torch.tensor(packed_und_indexes_in_query, dtype=torch.long),
            "packed_und_query_indexes": torch.tensor(packed_und_query_indexes, dtype=torch.long),
            "packed_uni_query_indexes": torch.tensor(packed_uni_query_indexes, dtype=torch.long),
            "key_values_lens_und": torch.tensor(curr_und_kvlens, dtype=torch.int),
            "key_values_lens_uni": torch.tensor(curr_uni_kvlens, dtype=torch.int),
            "packed_und_key_value_indexes": torch.tensor(packed_und_key_value_indexes, dtype=torch.long),
            "packed_uni_key_value_indexes": torch.tensor(packed_uni_key_value_indexes, dtype=torch.long),
        }

        for k, v in generation_input.items():
            if isinstance(v, torch.Tensor):
                generation_input[k] = v.to(self.device)

        return generation_input, new_und_lens, new_uni_lens, new_rope
        

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        vit_image_grid_thws: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_und_indexes_in_query: torch.LongTensor,
        packed_und_query_indexes: torch.LongTensor,
        packed_uni_query_indexes: torch.LongTensor,
        past_und_key_values: NaiveCache,
        past_uni_key_values: NaiveCache,
        key_values_lens_und: torch.IntTensor,
        key_values_lens_uni: torch.IntTensor,
        packed_und_key_value_indexes: torch.LongTensor,
        packed_uni_key_value_indexes: torch.LongTensor,
    ):
        packed_text_embedding = self.vision_language_model.model.language_model.embed_tokens(packed_text_ids)
        packed_query_sequence_und = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_query_sequence_und[packed_text_indexes] = packed_text_embedding

        packed_vit_token_embed = self.vision_language_model.model.visual(packed_vit_tokens, grid_thw=vit_image_grid_thws)
        assert packed_vit_token_embed.shape[0] == packed_vit_token_indexes.shape[0], "vit number of tokens and indexes not matching!"
        packed_query_sequence_und[packed_vit_token_indexes] = packed_vit_token_embed

        output = self.vision_language_model.forward_inference(
            vgen_model=self.vgen_model,
            packed_query_sequence_und=packed_query_sequence_und,
            packed_query_sequence_gen=None,
            query_lens_und=packed_seqlens,
            query_lens_gen=None,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_und_indexes_in_query=packed_und_indexes_in_query,
            packed_gen_indexes_in_query=None,
            vgen_e=None,
            vgen_grid_sizes=None,
            vgen_freqs=None,
            vgen_context=None,
            packed_und_query_indexes=packed_und_query_indexes,
            packed_uni_query_indexes=packed_uni_query_indexes,
            past_und_key_values=past_und_key_values,
            past_uni_key_values=past_uni_key_values,
            key_values_lens_und=key_values_lens_und,
            key_values_lens_uni=key_values_lens_uni,
            packed_und_key_value_indexes=packed_und_key_value_indexes,
            packed_uni_key_value_indexes=packed_uni_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            mode="und",
        )
        past_und_key_values = output.past_und_key_values
        past_uni_key_values = output.past_uni_key_values

        return past_und_key_values, past_uni_key_values


    def prepare_vae_images(self, curr_und_kvlens, curr_uni_kvlens, curr_rope, images, transforms, special_token_ids):
        patchified_vae_latent_shapes = list()
        packed_gen_indexes_in_query = list()
        packed_text_ids, packed_und_indexes_in_query = list(), list()
        packed_seqlens, packed_seqlens_und, packed_seqlens_gen, packed_m_position_ids = list(), list(), list(), [[], [], []]

        packed_und_query_indexes, packed_uni_query_indexes = list(), list()
        packed_und_key_value_indexes, packed_uni_key_value_indexes = list(), list()

        query_curr = und_curr = uni_curr = 0
        vae_image_tensors = list()
        new_und_lens, new_uni_lens, new_rope = list(), list(), list()
        for image, curr_und_kvlen, curr_uni_kvlen, curr_position_id in zip(images, curr_und_kvlens, curr_uni_kvlens, curr_rope):
            packed_und_key_value_indexes.extend(range(und_curr, und_curr + curr_und_kvlen))
            packed_uni_key_value_indexes.extend(range(uni_curr, uni_curr + curr_uni_kvlen))
            und_curr += curr_und_kvlen
            uni_curr += curr_uni_kvlen

            packed_text_ids.append(special_token_ids['sov_token_id'])
            packed_und_indexes_in_query.append(query_curr)
            packed_und_query_indexes.append(und_curr)
            packed_uni_query_indexes.append(uni_curr)
            query_curr += 1
            und_curr += 1
            uni_curr += 1

            image_tensor = transforms(image).unsqueeze(1)
            vae_image_tensors.append(image_tensor)
            F, H, W = image_tensor.shape[1:]
            f = ((F - 1) // self.vae_stride[0] + 1) // self.latent_patch_size[0]
            assert f == 1, f"image frame_number must be one, but got {f}"
            h = H // self.latent_downsample[1]
            w = W // self.latent_downsample[2]
            patchified_vae_latent_shapes.append((f, h, w))

            num_img_tokens = f * w * h
            packed_gen_indexes_in_query.extend(range(query_curr, query_curr + num_img_tokens))
            packed_uni_query_indexes.extend(range(uni_curr, uni_curr + num_img_tokens))
            query_curr += num_img_tokens
            uni_curr += num_img_tokens

            packed_text_ids.append(special_token_ids['eov_token_id'])
            packed_und_indexes_in_query.append(query_curr)
            packed_und_query_indexes.append(und_curr)
            packed_uni_query_indexes.append(uni_curr)
            query_curr += 1
            und_curr += 1
            uni_curr += 1

            for packed_position_ids in packed_m_position_ids:
                packed_position_ids.append(curr_position_id)
            t_index, h_index, w_index = get_qwen2_5_vl_mrope_index(
                st_idx=curr_position_id+1,
                spatial_merge_size=1,
                tokens_per_second=self.config.vlm_config.vision_config.tokens_per_second,
                image_grid_thw=[f, h, w]
            )
            packed_m_position_ids[0].extend(t_index)
            packed_m_position_ids[1].extend(h_index)
            packed_m_position_ids[2].extend(w_index)
            max_pos_id = max(max(t_index), max(h_index), max(w_index))
            for packed_position_ids in packed_m_position_ids:
                packed_position_ids.append(max_pos_id + 1)

            packed_m_position_ids = [torch.tensor(packed_position_ids, dtype=torch.long) for packed_position_ids in packed_m_position_ids]
            packed_m_position_ids = torch.stack(packed_m_position_ids, dim=0)
            packed_m_position_ids = packed_m_position_ids.unsqueeze(1)

            packed_seqlens.append(num_img_tokens + 2)
            packed_seqlens_und.append(2)
            packed_seqlens_gen.append(num_img_tokens)

            new_und_lens.append(curr_und_kvlen + 2)
            new_uni_lens.append(curr_uni_kvlen + num_img_tokens + 2)
            new_rope.append(max_pos_id + 2)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2], :image_tensor.shape[3]] = image_tensor

        timesteps = [0] * len(patchified_vae_latent_shapes)
        packed_timesteps = [torch.ones(math.prod(latent_shape)) * t for t, latent_shape in zip(timesteps, patchified_vae_latent_shapes)]
        packed_timesteps = torch.cat(packed_timesteps, dim=0)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_und_indexes_in_query": torch.tensor(packed_und_indexes_in_query, dtype=torch.long),
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_timesteps": packed_timesteps,
            "packed_gen_indexes_in_query": torch.tensor(packed_gen_indexes_in_query, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_seqlens_und": torch.tensor(packed_seqlens_und, dtype=torch.int),
            "packed_seqlens_gen": torch.tensor(packed_seqlens_gen, dtype=torch.int),
            "packed_position_ids": packed_m_position_ids,
            "packed_und_query_indexes": torch.tensor(packed_und_query_indexes, dtype=torch.long),
            "packed_uni_query_indexes": torch.tensor(packed_uni_query_indexes, dtype=torch.long),
            "key_values_lens_und": torch.tensor(curr_und_kvlens, dtype=torch.int),
            "key_values_lens_uni": torch.tensor(curr_uni_kvlens, dtype=torch.int),
            "packed_und_key_value_indexes": torch.tensor(packed_und_key_value_indexes, dtype=torch.long),
            "packed_uni_key_value_indexes": torch.tensor(packed_uni_key_value_indexes, dtype=torch.long),
        }

        for k, v in generation_input.items():
            if isinstance(v, torch.Tensor):
                generation_input[k] = v.to(self.device)

        return generation_input, new_und_lens, new_uni_lens, new_rope


    @torch.no_grad
    def forward_cache_update_vae(
        self,
        vae_model,
        packed_text_ids: torch.LongTensor,
        packed_und_indexes_in_query: torch.LongTensor,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List,
        packed_timesteps: torch.Tensor,
        packed_gen_indexes_in_query: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_seqlens_und: torch.IntTensor,
        packed_seqlens_gen: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_und_query_indexes: torch.LongTensor,
        packed_uni_query_indexes: torch.LongTensor,
        past_und_key_values: NaiveCache,
        past_uni_key_values: NaiveCache,
        key_values_lens_und: torch.IntTensor,
        key_values_lens_uni: torch.IntTensor,
        packed_und_key_value_indexes: torch.Tensor,
        packed_uni_key_value_indexes: torch.Tensor,
    ):
        packed_text_embedding = self.vision_language_model.model.language_model.embed_tokens(packed_text_ids)
        packed_query_sequence_und = packed_text_embedding

        padded_latent = vae_model.model.encode(padded_images, vae_model.scale)

        p, q, r = self.latent_patch_size
        packed_latent = list()
        for latent, (f, h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :f * p, :h * q, :w * r].reshape(self.latent_channel, f, p, h, q, w, r)
            latent = torch.einsum("cfphqwr->fhwpqrc", latent).reshape(f * h * w, p * q * r * self.latent_channel)
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        packed_query_sequence_gen = self.vgen_model.patch_embedding(packed_latent)

        grid_sizes = torch.stack(
            [torch.tensor((f, h, w), dtype=torch.long) for f, h, w in patchified_vae_latent_shapes])

        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            vgen_e = self.vgen_model.time_embedding(
                wan_sinusoidal_embedding_1d(self.vgen_model.freq_dim, packed_timesteps).float())
            vgen_e0 = self.vgen_model.time_projection(vgen_e).unflatten(1, (6, self.vgen_model.dim))
            assert vgen_e.dtype == torch.float32 and vgen_e0.dtype == torch.float32, 'time_embedding dtype error'

        vgen_model_cross_att_context = self.vgen_model.text_embedding(self.vgen_model_cross_att_context)
        vgen_kwargs = dict(
            vgen_e=vgen_e0,
            vgen_grid_sizes=grid_sizes,
            vgen_freqs=self.vgen_model.freqs.to(self.device),
            vgen_context=vgen_model_cross_att_context,
        )

        output = self.vision_language_model.forward_inference(
            vgen_model=self.vgen_model,
            packed_query_sequence_und=packed_query_sequence_und,
            packed_query_sequence_gen=packed_query_sequence_gen,
            query_lens_und=packed_seqlens_und,
            query_lens_gen=packed_seqlens_gen,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_und_indexes_in_query=packed_und_indexes_in_query,
            packed_gen_indexes_in_query=packed_gen_indexes_in_query,
            **vgen_kwargs,
            packed_und_query_indexes=packed_und_query_indexes,
            packed_uni_query_indexes=packed_uni_query_indexes,
            past_und_key_values=past_und_key_values,
            past_uni_key_values=past_uni_key_values,
            key_values_lens_und=key_values_lens_und,
            key_values_lens_uni=key_values_lens_uni,
            packed_und_key_value_indexes=packed_und_key_value_indexes,
            packed_uni_key_value_indexes=packed_uni_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            mode="gen",
        )
        past_und_key_values = output.past_und_key_values
        past_uni_key_values = output.past_uni_key_values

        return past_und_key_values, past_uni_key_values


    def prepare_vae_latent(self, curr_und_kvlens, curr_uni_kvlens, curr_rope, video_sizes, special_token_ids, generator: torch.Generator = None):
        packed_text_ids, packed_und_indexes_in_query = list(), list()
        init_noises, patchified_vae_latent_shapes = list(), list()
        packed_gen_indexes_in_query = list()
        packed_seqlens, packed_seqlens_und, packed_seqlens_gen, packed_m_position_ids = list(), list(), list(), [[], [], []]

        packed_und_query_indexes, packed_uni_query_indexes = list(), list()
        packed_und_key_value_indexes, packed_uni_key_value_indexes = list(), list()

        query_curr = und_curr = uni_curr = 0
        for (F, H, W), curr_und_kvlen, curr_uni_kvlen, curr_position_id in zip(video_sizes, curr_und_kvlens, curr_uni_kvlens, curr_rope):
            packed_und_key_value_indexes.extend(range(und_curr, und_curr + curr_und_kvlen))
            packed_uni_key_value_indexes.extend(range(uni_curr, uni_curr + curr_uni_kvlen))
            und_curr += curr_und_kvlen
            uni_curr += curr_uni_kvlen

            packed_text_ids.append(special_token_ids['sov_token_id'])
            packed_und_indexes_in_query.append(query_curr)
            packed_und_query_indexes.append(und_curr)
            packed_uni_query_indexes.append(uni_curr)
            query_curr += 1
            und_curr += 1
            uni_curr += 1

            f, h, w = (F - 1) // self.vae_stride[0] + 1, H // self.vae_stride[1], W // self.vae_stride[2]
            num_img_tokens = (f * h * w) // math.prod(self.latent_patch_size)
            # we apply 3d conv patch_embedding afterwards
            init_noises.append(
                torch.randn(self.latent_channel, f, h, w , dtype=torch.float32, device=self.device, generator=generator)
            )
            patchified_vae_latent_shapes.append((f // self.latent_patch_size[0], h // self.latent_patch_size[1], w // self.latent_patch_size[2]))

            packed_gen_indexes_in_query.extend(range(query_curr, query_curr + num_img_tokens))
            packed_uni_query_indexes.extend(range(uni_curr, uni_curr + num_img_tokens))
            query_curr += num_img_tokens
            uni_curr += num_img_tokens

            packed_text_ids.append(special_token_ids['eov_token_id'])
            packed_und_indexes_in_query.append(query_curr)
            packed_und_query_indexes.append(und_curr)
            packed_uni_query_indexes.append(uni_curr)
            query_curr += 1
            und_curr += 1
            uni_curr += 1

            for packed_position_ids in packed_m_position_ids:
                packed_position_ids.append(curr_position_id)
            t_index, h_index, w_index = get_qwen2_5_vl_mrope_index(
                st_idx=curr_position_id+1,
                spatial_merge_size=1,
                tokens_per_second=self.config.vlm_config.vision_config.tokens_per_second,
                image_grid_thw=[f // self.latent_patch_size[0], h // self.latent_patch_size[1], w // self.latent_patch_size[2]]
            )
            packed_m_position_ids[0].extend(t_index)
            packed_m_position_ids[1].extend(h_index)
            packed_m_position_ids[2].extend(w_index)
            max_pos_id = max(max(t_index), max(h_index), max(w_index))
            for packed_position_ids in packed_m_position_ids:
                packed_position_ids.append(max_pos_id + 1)

            packed_m_position_ids = [torch.tensor(packed_position_ids, dtype=torch.long) for packed_position_ids in packed_m_position_ids]
            packed_m_position_ids = torch.stack(packed_m_position_ids, dim=0)
            packed_m_position_ids = packed_m_position_ids.unsqueeze(1)

            packed_seqlens.append(num_img_tokens + 2)
            packed_seqlens_und.append(2)
            packed_seqlens_gen.append(num_img_tokens)

        latent_shapes = [item.shape for item in init_noises]
        max_latent_shape = [max(item) for item in list(zip(*latent_shapes))]
        padded_init_noises = init_noises[0].new_zeros(size=(len(video_sizes), *max_latent_shape))
        for i, noise in enumerate(init_noises):
            padded_init_noises[i, :, :noise.shape[1], :noise.shape[2], :noise.shape[3]] = noise

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_und_indexes_in_query": torch.tensor(packed_und_indexes_in_query, dtype=torch.long),
            "padded_init_noises": padded_init_noises,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_gen_indexes_in_query": torch.tensor(packed_gen_indexes_in_query, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_seqlens_und": torch.tensor(packed_seqlens_und, dtype=torch.int),
            "packed_seqlens_gen": torch.tensor(packed_seqlens_gen, dtype=torch.int),
            "packed_position_ids": packed_m_position_ids,
            "packed_und_query_indexes": torch.tensor(packed_und_query_indexes, dtype=torch.long),
            "packed_uni_query_indexes": torch.tensor(packed_uni_query_indexes, dtype=torch.long),
            "key_values_lens_und": torch.tensor(curr_und_kvlens, dtype=torch.int),
            "key_values_lens_uni": torch.tensor(curr_uni_kvlens, dtype=torch.int),
            "packed_und_key_value_indexes": torch.tensor(packed_und_key_value_indexes, dtype=torch.long),
            "packed_uni_key_value_indexes": torch.tensor(packed_uni_key_value_indexes, dtype=torch.long),
        }

        for k, v in generation_input.items():
            if isinstance(v, torch.Tensor):
                generation_input[k] = v.to(self.device)

        return generation_input


    def prepare_vae_latent_cfg(self, curr_und_kvlens, curr_uni_kvlens, curr_rope, video_sizes):
        packed_m_position_ids = [[], [], []]
        packed_und_query_indexes, packed_uni_query_indexes = list(), list()
        packed_und_key_value_indexes, packed_uni_key_value_indexes = list(), list()

        und_curr = uni_curr = 0
        for (F, H, W), curr_und_kvlen, curr_uni_kvlen, curr_position_id in zip(video_sizes, curr_und_kvlens, curr_uni_kvlens, curr_rope):
            packed_und_key_value_indexes.extend(range(und_curr, und_curr + curr_und_kvlen))
            packed_uni_key_value_indexes.extend(range(uni_curr, uni_curr + curr_uni_kvlen))
            und_curr += curr_und_kvlen
            uni_curr += curr_uni_kvlen

            packed_und_query_indexes.append(und_curr)
            packed_uni_query_indexes.append(uni_curr)
            und_curr += 1
            uni_curr += 1

            f, h, w = (F - 1) // self.vae_stride[0] + 1, H // self.vae_stride[1], W // self.vae_stride[2]
            num_img_tokens = (f * h * w) // math.prod(self.latent_patch_size)
            packed_uni_query_indexes.extend(range(uni_curr, uni_curr + num_img_tokens))
            uni_curr += num_img_tokens

            packed_und_query_indexes.append(und_curr)
            packed_uni_query_indexes.append(uni_curr)
            und_curr += 1
            uni_curr += 1

            for packed_position_ids in packed_m_position_ids:
                packed_position_ids.append(curr_position_id)
            t_index, h_index, w_index = get_qwen2_5_vl_mrope_index(
                st_idx=curr_position_id+1,
                spatial_merge_size=1,
                tokens_per_second=self.config.vlm_config.vision_config.tokens_per_second,
                image_grid_thw=[f // self.latent_patch_size[0], h // self.latent_patch_size[1], w // self.latent_patch_size[2]]
            )
            packed_m_position_ids[0].extend(t_index)
            packed_m_position_ids[1].extend(h_index)
            packed_m_position_ids[2].extend(w_index)
            max_pos_id = max(max(t_index), max(h_index), max(w_index))
            for packed_position_ids in packed_m_position_ids:
                packed_position_ids.append(max_pos_id + 1)

            packed_m_position_ids = [torch.tensor(packed_position_ids, dtype=torch.long) for packed_position_ids in packed_m_position_ids]
            packed_m_position_ids = torch.stack(packed_m_position_ids, dim=0)
            packed_m_position_ids = packed_m_position_ids.unsqueeze(1)


        generation_input = {
            "cfg_packed_und_query_indexes": torch.tensor(packed_und_query_indexes, dtype=torch.long),
            "cfg_packed_uni_query_indexes": torch.tensor(packed_uni_query_indexes, dtype=torch.long),
            "cfg_key_values_lens_und": torch.tensor(curr_und_kvlens, dtype=torch.int),
            "cfg_key_values_lens_uni": torch.tensor(curr_uni_kvlens, dtype=torch.int),
            "cfg_packed_und_key_value_indexes": torch.tensor(packed_und_key_value_indexes, dtype=torch.long),
            "cfg_packed_uni_key_value_indexes": torch.tensor(packed_uni_key_value_indexes, dtype=torch.long),
            "cfg_packed_position_ids": packed_m_position_ids,
        }

        for k, v in generation_input.items():
            if isinstance(v, torch.Tensor):
                generation_input[k] = v.to(self.device)

        return generation_input


    @torch.no_grad
    def visual_gen(
        self,
        packed_text_ids: torch.LongTensor,
        packed_und_indexes_in_query: torch.LongTensor,
        padded_init_noises: torch.Tensor,
        patchified_vae_latent_shapes: List[Tuple[int, int, int]],
        packed_gen_indexes_in_query: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_seqlens_und: torch.IntTensor,
        packed_seqlens_gen: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_und_query_indexes: torch.Tensor,
        packed_uni_query_indexes: torch.Tensor,
        past_und_key_values: NaiveCache,
        past_uni_key_values: NaiveCache,
        key_values_lens_und: torch.Tensor,
        key_values_lens_uni: torch.Tensor,
        packed_und_key_value_indexes: torch.Tensor,
        packed_uni_key_value_indexes: torch.Tensor,
        # hyper-params
        flow_solver: str = "unipc",
        num_timesteps: int = 50,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: int = 1.0,
        cfg_text_packed_und_query_indexes: Optional[torch.Tensor] = None,
        cfg_text_packed_uni_query_indexes: Optional[torch.Tensor] = None,
        cfg_text_past_und_key_values: Optional[NaiveCache] = None,
        cfg_text_past_uni_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens_und: Optional[torch.Tensor] = None,
        cfg_text_key_values_lens_uni: Optional[torch.Tensor] = None,
        cfg_text_packed_und_key_value_indexes: Optional[torch.Tensor] = None,
        cfg_text_packed_uni_key_value_indexes: Optional[torch.Tensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: int = 1.0,
        cfg_img_packed_und_query_indexes: Optional[torch.Tensor] = None,
        cfg_img_packed_uni_query_indexes: Optional[torch.Tensor] = None,
        cfg_img_past_und_key_values: Optional[NaiveCache] = None,
        cfg_img_past_uni_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens_und: Optional[torch.Tensor] = None,
        cfg_img_key_values_lens_uni: Optional[torch.Tensor] = None,
        cfg_img_packed_und_key_value_indexes: Optional[torch.Tensor] = None,
        cfg_img_packed_uni_key_value_indexes: Optional[torch.Tensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        generator: torch.Generator = None,
    ):
        x_t = padded_init_noises

        if flow_solver == "naive":
            timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
            timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
            dts =  timesteps[:-1] - timesteps[1:]
            timesteps = timesteps[:-1]
            timesteps = (timesteps * self.vgen_num_train_timesteps).int()
        elif flow_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.vgen_num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                num_timesteps, device=self.device, shift=timestep_shift)
            timesteps = sample_scheduler.timesteps
        elif flow_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.vgen_num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(num_timesteps, timestep_shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        for i, t in enumerate(timesteps):
            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if (t / self.vgen_num_train_timesteps) > cfg_interval[0] and (t / self.vgen_num_train_timesteps) <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self._forward_flow_wan(
                x_t=x_t,
                packed_text_ids=packed_text_ids,
                timestep=timestep,
                patchified_vae_latent_shapes=patchified_vae_latent_shapes,
                packed_und_indexes_in_query=packed_und_indexes_in_query,
                packed_gen_indexes_in_query=packed_gen_indexes_in_query,
                packed_seqlens=packed_seqlens,
                packed_seqlens_und=packed_seqlens_und,
                packed_seqlens_gen=packed_seqlens_gen,
                packed_position_ids=packed_position_ids,
                packed_und_query_indexes=packed_und_query_indexes,
                packed_uni_query_indexes=packed_uni_query_indexes,
                past_und_key_values=past_und_key_values,
                past_uni_key_values=past_uni_key_values,
                key_values_lens_und=key_values_lens_und,
                key_values_lens_uni=key_values_lens_uni,
                packed_und_key_value_indexes=packed_und_key_value_indexes,
                packed_uni_key_value_indexes=packed_uni_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_und_query_indexes=cfg_text_packed_und_query_indexes,
                cfg_text_packed_uni_query_indexes=cfg_text_packed_uni_query_indexes,
                cfg_text_past_und_key_values=cfg_text_past_und_key_values,
                cfg_text_past_uni_key_values=cfg_text_past_uni_key_values,
                cfg_text_key_values_lens_und=cfg_text_key_values_lens_und,
                cfg_text_key_values_lens_uni=cfg_text_key_values_lens_uni,
                cfg_text_packed_und_key_value_indexes=cfg_text_packed_und_key_value_indexes,
                cfg_text_packed_uni_key_value_indexes=cfg_text_packed_uni_key_value_indexes,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_und_query_indexes=cfg_img_packed_und_query_indexes,
                cfg_img_packed_uni_query_indexes=cfg_img_packed_uni_query_indexes,
                cfg_img_past_und_key_values=cfg_img_past_und_key_values,
                cfg_img_past_uni_key_values=cfg_img_past_uni_key_values,
                cfg_img_key_values_lens_und=cfg_img_key_values_lens_und,
                cfg_img_key_values_lens_uni=cfg_img_key_values_lens_uni,
                cfg_img_packed_und_key_value_indexes=cfg_img_packed_und_key_value_indexes,
                cfg_img_packed_uni_key_value_indexes=cfg_img_packed_uni_key_value_indexes,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
            )

            if flow_solver == "naive":
                x_t = x_t - v_t * dts[i] # velocity pointing from data to noise
            elif flow_solver == 'unipc' or flow_solver == 'dpm++':
                x_t = sample_scheduler.step(
                    v_t,
                    t,
                    x_t,
                    return_dict=False,
                    generator=generator)[0]
            else:
                raise NotImplementedError("Unsupported solver.")

        return x_t


    @torch.no_grad
    def _forward_flow_wan(
        self,
        x_t: torch.Tensor,
        packed_text_ids: torch.LongTensor,
        timestep: torch.Tensor,
        patchified_vae_latent_shapes: List[Tuple[int, int, int]],
        packed_und_indexes_in_query: torch.LongTensor,
        packed_gen_indexes_in_query: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_seqlens_und: torch.IntTensor,
        packed_seqlens_gen: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_und_query_indexes: torch.Tensor,
        packed_uni_query_indexes: torch.Tensor,
        past_und_key_values: NaiveCache,
        past_uni_key_values: NaiveCache,
        key_values_lens_und: torch.Tensor,
        key_values_lens_uni: torch.Tensor,
        packed_und_key_value_indexes: torch.Tensor,
        packed_uni_key_value_indexes: torch.Tensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: int = 1.0,
        cfg_text_packed_und_query_indexes: Optional[torch.Tensor] = None,
        cfg_text_packed_uni_query_indexes: Optional[torch.Tensor] = None,
        cfg_text_past_und_key_values: Optional[NaiveCache] = None,
        cfg_text_past_uni_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens_und: Optional[torch.Tensor] = None,
        cfg_text_key_values_lens_uni: Optional[torch.Tensor] = None,
        cfg_text_packed_und_key_value_indexes: Optional[torch.Tensor] = None,
        cfg_text_packed_uni_key_value_indexes: Optional[torch.Tensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: int = 1.0,
        cfg_img_packed_und_query_indexes: Optional[torch.Tensor] = None,
        cfg_img_packed_uni_query_indexes: Optional[torch.Tensor] = None,
        cfg_img_past_und_key_values: Optional[NaiveCache] = None,
        cfg_img_past_uni_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens_und: Optional[torch.Tensor] = None,
        cfg_img_key_values_lens_uni: Optional[torch.Tensor] = None,
        cfg_img_packed_und_key_value_indexes: Optional[torch.Tensor] = None,
        cfg_img_packed_uni_key_value_indexes: Optional[torch.Tensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
    ):
        packed_text_embedding = self.vision_language_model.model.language_model.embed_tokens(packed_text_ids)
        packed_query_sequence_und = packed_text_embedding

        if self.config.visual_gen:
            packed_latent = []
            p, q, r = self.latent_patch_size
            for latent, (f, h, w) in zip(x_t, patchified_vae_latent_shapes):
                latent = latent[:, :f*p, :h*q, :w*r].reshape(self.latent_channel, f, p, h, q, w, r)
                latent = torch.einsum("cfphqwr->fhwpqrc", latent).reshape(f * h * w, p * q * r * self.latent_channel)
                packed_latent.append(latent)
            packed_latent = torch.cat(packed_latent, dim=0)
            packed_query_sequence_gen = self.vgen_model.patch_embedding(packed_latent)

            grid_sizes = torch.stack(
                [torch.tensor((f, h, w), dtype=torch.long) for f, h, w in patchified_vae_latent_shapes])

            packed_timesteps = [t.new_ones(math.prod(latent_shape)) * t for t, latent_shape in zip(timestep, patchified_vae_latent_shapes)]
            packed_timesteps = torch.cat(packed_timesteps, dim=0)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
                vgen_e = self.vgen_model.time_embedding(
                    wan_sinusoidal_embedding_1d(self.vgen_model.freq_dim, packed_timesteps).float())
                vgen_e0 = self.vgen_model.time_projection(vgen_e).unflatten(1, (6, self.vgen_model.dim))
                assert vgen_e.dtype == torch.float32 and vgen_e0.dtype == torch.float32, 'wan model time_embedding dtype error'

            vgen_model_cross_att_context = self.vgen_model.text_embedding(self.vgen_model_cross_att_context)
            vgen_kwargs = dict(
                vgen_e=vgen_e0,
                vgen_grid_sizes=grid_sizes,
                vgen_freqs=self.vgen_model.freqs.to(self.device),
                vgen_context=vgen_model_cross_att_context,
            )

        output = self.vision_language_model.forward_inference(
            vgen_model=self.vgen_model,
            packed_query_sequence_und=packed_query_sequence_und,
            packed_query_sequence_gen=packed_query_sequence_gen,
            query_lens_und=packed_seqlens_und,
            query_lens_gen=packed_seqlens_gen,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_und_indexes_in_query=packed_und_indexes_in_query,
            packed_gen_indexes_in_query=packed_gen_indexes_in_query,
            **vgen_kwargs,
            packed_und_query_indexes=packed_und_query_indexes,
            packed_uni_query_indexes=packed_uni_query_indexes,
            past_und_key_values=past_und_key_values,
            past_uni_key_values=past_uni_key_values,
            key_values_lens_und=key_values_lens_und,
            key_values_lens_uni=key_values_lens_uni,
            packed_und_key_value_indexes=packed_und_key_value_indexes,
            packed_uni_key_value_indexes=packed_uni_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            mode="gen",
        )
        v_t = self.vgen_model.head(output.packed_query_sequence_gen, vgen_e)

        if cfg_text_scale > 1.0:
            cfg_text_output = self.vision_language_model.forward_inference(
                vgen_model=self.vgen_model,
                packed_query_sequence_und=packed_query_sequence_und,
                packed_query_sequence_gen=packed_query_sequence_gen,
                query_lens_und=packed_seqlens_und,
                query_lens_gen=packed_seqlens_gen,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_und_indexes_in_query=packed_und_indexes_in_query,
                packed_gen_indexes_in_query=packed_gen_indexes_in_query,
                **vgen_kwargs,
                packed_und_query_indexes=cfg_text_packed_und_query_indexes,
                packed_uni_query_indexes=cfg_text_packed_uni_query_indexes,
                past_und_key_values=cfg_text_past_und_key_values,
                past_uni_key_values=cfg_text_past_uni_key_values,
                key_values_lens_und=cfg_text_key_values_lens_und,
                key_values_lens_uni=cfg_text_key_values_lens_uni,
                packed_und_key_value_indexes=cfg_text_packed_und_key_value_indexes,
                packed_uni_key_value_indexes=cfg_text_packed_uni_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                mode="gen",
            )
            cfg_text_v_t = self.vgen_model.head(cfg_text_output.packed_query_sequence_gen, vgen_e)

        if cfg_img_scale > 1.0:
            cfg_img_output = self.vision_language_model.forward_inference(
                vgen_model=self.vgen_model,
                packed_query_sequence_und=packed_query_sequence_und,
                packed_query_sequence_gen=packed_query_sequence_gen,
                query_lens_und=packed_seqlens_und,
                query_lens_gen=packed_seqlens_gen,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_und_indexes_in_query=packed_und_indexes_in_query,
                packed_gen_indexes_in_query=packed_gen_indexes_in_query,
                **vgen_kwargs,
                packed_und_query_indexes=cfg_img_packed_und_query_indexes,
                packed_uni_query_indexes=cfg_img_packed_uni_query_indexes,
                past_und_key_values=cfg_img_past_und_key_values,
                past_uni_key_values=cfg_img_past_uni_key_values,
                key_values_lens_und=cfg_img_key_values_lens_und,
                key_values_lens_uni=cfg_img_key_values_lens_uni,
                packed_und_key_value_indexes=cfg_img_packed_und_key_value_indexes,
                packed_uni_key_value_indexes=cfg_img_packed_uni_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                mode="gen",
            )
            cfg_img_v_t = self.vgen_model.head(cfg_img_output.packed_query_sequence_gen, vgen_e)

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                
                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                elif cfg_renorm_type == "noop":
                    norm_v_t = torch.ones_like(v_t)
                    norm_v_t_ = torch.ones_like(v_t_)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale
        else:
            # No CFG
            pass


        unpacked_latent = v_t.split(packed_seqlens_gen.tolist())
        p, q, r = self.latent_patch_size
        vae_latent_shapes = [(f * p, h * q, w * r) for (f, h, w) in patchified_vae_latent_shapes]
        max_latent_shapes = [max(item) for item in list(zip(*vae_latent_shapes))]
        padded_v_t = v_t.new_zeros(size=(len(unpacked_latent), self.vgen_out_dim, *max_latent_shapes))

        for idx, (latent, (f, h, w)) in enumerate(zip(unpacked_latent, patchified_vae_latent_shapes)):
            latent = latent.reshape(f, h, w, p, q, r, self.vgen_out_dim)
            latent = torch.einsum("fhwpqrc->cfphqwr", latent)
            latent = latent.reshape(self.vgen_out_dim, f * p, h * q, w * r)
            padded_v_t[idx, :, :f * p, :h * q, :w * r] = latent

        return padded_v_t


    def prepare_start_tokens(self, curr_und_kvlens, curr_uni_kvlens, curr_rope, special_token_ids, tokenizer):
        packed_text_ids = list()
        packed_m_position_ids = [[], [], []]
        packed_und_key_value_indexes, packed_uni_key_value_indexes = list(), list()

        query_curr = und_curr = uni_curr = 0
        for curr_und_kvlen, curr_uni_kvlen, curr_position_id in zip(curr_und_kvlens, curr_uni_kvlens, curr_rope):
            packed_und_key_value_indexes.extend(range(und_curr, und_curr + curr_und_kvlen))
            packed_uni_key_value_indexes.extend(range(uni_curr, uni_curr + curr_uni_kvlen))
            und_curr += curr_und_kvlen
            uni_curr += curr_uni_kvlen

            text_ids = tokenizer.encode("\n")
            packed_text_ids.extend(text_ids)

            for packed_position_ids in packed_m_position_ids:
                packed_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))

            query_curr += len(text_ids)
            und_curr += len(text_ids)
            uni_curr += len(text_ids)

        packed_m_position_ids = [torch.tensor(packed_position_ids, dtype=torch.long) for packed_position_ids in packed_m_position_ids]
        packed_m_position_ids = torch.stack(packed_m_position_ids, dim=0)
        packed_m_position_ids = packed_m_position_ids.unsqueeze(1)

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_query_position_ids": packed_m_position_ids,
            "key_values_lens_und": torch.tensor(curr_und_kvlens, dtype=torch.int),
            "key_values_lens_uni": torch.tensor(curr_uni_kvlens, dtype=torch.int),
            "packed_und_key_value_indexes": torch.tensor(packed_und_key_value_indexes, dtype=torch.long),
            "packed_uni_key_value_indexes": torch.tensor(packed_uni_key_value_indexes, dtype=torch.long),
        }
        for k, v in generation_input.items():
            if isinstance(v, torch.Tensor):
                generation_input[k] = v.to(self.device)

        return generation_input


    @torch.no_grad
    def generate_text(
        self,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        past_und_key_values: NaiveCache,
        past_uni_key_values: NaiveCache,
        key_values_lens_und: torch.IntTensor,
        key_values_lens_uni: torch.IntTensor,
        packed_und_key_value_indexes: torch.LongTensor,
        packed_uni_key_value_indexes: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.vision_language_model.model.language_model.embed_tokens(curr_tokens)
            # because each time there is only one tokens
            query_lens = torch.ones_like(curr_tokens)
            packed_und_indexes_in_query = torch.zeros_like(curr_tokens)
            packed_und_query_indexes = torch.cumsum(key_values_lens_und, dim=0) + torch.arange(
                0, len(key_values_lens_und), 
                device=key_values_lens_und.device, 
                dtype=key_values_lens_und.dtype
            )
            packed_uni_query_indexes = torch.cumsum(key_values_lens_uni, dim=0) + torch.arange(
                0, len(key_values_lens_uni), 
                device=key_values_lens_uni.device, 
                dtype=key_values_lens_uni.dtype
            )

            und_uppacked = list(packed_und_key_value_indexes.split(key_values_lens_und.tolist(), dim=0))
            for i in range(len(und_uppacked)):
                und_uppacked[i] += i
            packed_und_key_value_indexes = torch.cat(und_uppacked, dim=0)

            uni_uppacked = list(packed_uni_key_value_indexes.split(key_values_lens_uni.tolist(), dim=0))
            for i in range(len(uni_uppacked)):
                uni_uppacked[i] += i
            packed_uni_key_value_indexes = torch.cat(uni_uppacked, dim=0)

            output = self.vision_language_model.forward_inference(
                vgen_model=self.vgen_model,
                packed_query_sequence_und=packed_text_embedding,
                packed_query_sequence_gen=None,
                query_lens_und=query_lens,
                query_lens_gen=None,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_und_indexes_in_query=packed_und_indexes_in_query,
                packed_gen_indexes_in_query=None,
                vgen_e=None,
                vgen_grid_sizes=None,
                vgen_freqs=None,
                vgen_context=None,
                packed_und_query_indexes=packed_und_query_indexes,
                packed_uni_query_indexes=packed_uni_query_indexes,
                past_und_key_values=past_und_key_values,
                past_uni_key_values=past_uni_key_values,
                key_values_lens_und=key_values_lens_und,
                key_values_lens_uni=key_values_lens_uni,
                packed_und_key_value_indexes=packed_und_key_value_indexes,
                packed_uni_key_value_indexes=packed_uni_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                mode="und",
            )

            past_und_key_values = output.past_und_key_values
            past_uni_key_values = output.past_uni_key_values
            packed_query_sequence_und = output.packed_query_sequence_und
            pred_logits = self.vision_language_model.lm_head(packed_query_sequence_und)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            und_uppacked = list(packed_und_key_value_indexes.split(key_values_lens_und.tolist(), dim=0))
            for i in range(len(und_uppacked)):
                und_uppacked[i] = torch.cat(
                    [und_uppacked[i], torch.tensor([und_uppacked[i][-1] + 1], device=self.device)], dim=0
                )
            packed_und_key_value_indexes = torch.cat(und_uppacked, dim=0)

            uni_uppacked = list(packed_uni_key_value_indexes.split(key_values_lens_uni.tolist(), dim=0))
            for i in range(len(uni_uppacked)):
                uni_uppacked[i] = torch.cat(
                    [uni_uppacked[i], torch.tensor([uni_uppacked[i][-1] + 1], device=self.device)], dim=0
                )
            packed_uni_key_value_indexes = torch.cat(uni_uppacked, dim=0)

            key_values_lens_und = key_values_lens_und + 1
            key_values_lens_uni = key_values_lens_uni + 1

            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # only support batch=1
                break

        return torch.stack(generated_sequence, dim=0)


    @torch.no_grad
    def chat(
        self,
        tokenizer,
        special_token_ids,
        image_transform,
        images,
        system_prompt,
        user_prompt,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        cross_attn_num_layer_type:  str = "min",
    ):
        # prefill
        past_und_key_values = NaiveCache(self.config.vlm_config.num_hidden_layers)
        if cross_attn_num_layer_type == "min":
            past_uni_key_values = NaiveCache(min(self.config.vlm_config.num_hidden_layers, self.config.vgen_config.num_layers) - 1)
        elif cross_attn_num_layer_type == "max":
            past_uni_key_values = NaiveCache(max(self.config.vlm_config.num_hidden_layers, self.config.vgen_config.num_layers) - 1)
        else:
            raise NotImplementedError
        new_und_lens = [0]
        new_uni_lens = [0]
        new_rope = [0]

        generation_input, new_und_lens, new_uni_lens, new_rope = self.prepare_prompts(
            curr_und_kvlens=new_und_lens,
            curr_uni_kvlens=new_uni_lens,
            curr_rope=new_rope,
            prompts=[system_prompt],
            tokenizer=tokenizer,
            special_token_ids=special_token_ids,
            disable_bos_eos=True,
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_und_key_values, past_uni_key_values = self.forward_cache_update_text(**generation_input, past_und_key_values=past_und_key_values, past_uni_key_values=past_uni_key_values)

        if images is not None:
            for image in images:
                generation_input, new_und_lens, new_uni_lens, new_rope = self.prepare_vit_images(
                    curr_und_kvlens=new_und_lens,
                    curr_uni_kvlens=new_uni_lens,
                    curr_rope=new_rope, 
                    images=[image], 
                    transforms=image_transform,
                    special_token_ids=special_token_ids,
                )
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    past_und_key_values, past_uni_key_values = self.forward_cache_update_vit(**generation_input, past_und_key_values=past_und_key_values, past_uni_key_values=past_uni_key_values)

        generation_input, new_und_lens, new_uni_lens, new_rope = self.prepare_prompts(
            curr_und_kvlens=new_und_lens,
            curr_uni_kvlens=new_uni_lens,
            curr_rope=new_rope,
            prompts=[user_prompt],
            tokenizer=tokenizer,
            special_token_ids=special_token_ids,
            disable_bos_eos=True,
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_und_key_values, past_uni_key_values = self.forward_cache_update_text(**generation_input, past_und_key_values=past_und_key_values, past_uni_key_values=past_uni_key_values)

        generation_input = self.prepare_start_tokens(new_und_lens, new_uni_lens, new_rope, special_token_ids, tokenizer)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_und_key_values=past_und_key_values,
                past_uni_key_values=past_uni_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=special_token_ids['eos_token_id'],
                **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        output = output.lstrip('\n')

        return output