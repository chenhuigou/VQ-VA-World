# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# Copyright 2025 Zeyu Wang & Zilong Chen.
# SPDX-License-Identifier: Apache-2.0

import re
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention
from flash_attn import flash_attn_varlen_func

from transformers.generation import GenerationMixin
# from .masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import ModelOutput

from modeling.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLAttention,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2_5_VLRotaryEmbedding,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLPreTrainedModel,
    apply_multimodal_rotary_pos_emb
)
from modeling.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLTextConfig, Qwen2_5_VLVisionConfig

from transformers.utils import logging
logger = logging.get_logger(__name__)

class NaiveCache:
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_lens(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        else:
            return 0


@dataclass
class BaseNavitOutputWithPast(ModelOutput):
    # packed_query_sequence: torch.FloatTensor = None
    packed_query_sequence_und: torch.FloatTensor = None
    packed_query_sequence_gen: torch.FloatTensor = None
    past_und_key_values: Optional[NaiveCache] = None
    past_uni_key_values: Optional[NaiveCache] = None


class PackedAttention(Qwen2_5_VLAttention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        if self.config.qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        nested_attention_masks: List[torch.Tensor],
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ):
        packed_query_states = self.q_proj(packed_sequence).view(-1, self.num_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_sequence).view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states = packed_query_states.permute(1, 0, 2).unsqueeze(0)
        packed_key_states = packed_key_states.permute(1, 0, 2).unsqueeze(0)
        packed_query_states, packed_key_states = apply_multimodal_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, self.rope_scaling["mrope_section"], unsqueeze_dim=1
        )
        packed_query_states = packed_query_states.squeeze(0).permute(1, 0, 2)
        packed_key_states = packed_key_states.squeeze(0).permute(1, 0, 2)

        packed_key_states = packed_key_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
        packed_key_states = packed_key_states.reshape(-1, self.num_heads, self.head_dim)
        packed_value_states = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
        packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

        unpacked_query_states = packed_query_states.transpose(0, 1).split(sample_lens, dim=1)
        unpacked_key_states = packed_key_states.transpose(0, 1).split(sample_lens, dim=1)
        unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)

        upacked_attn_output = []
        for query_states, key_states, value_states, attention_mask in zip(
            unpacked_query_states, unpacked_key_states, unpacked_value_states, nested_attention_masks
        ):
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                attn_output = scaled_dot_product_attention(
                    query_states.to(torch.bfloat16).unsqueeze(0), 
                    key_states.to(torch.bfloat16).unsqueeze(0), 
                    value_states.to(torch.bfloat16).unsqueeze(0),
                    attention_mask.to(torch.bfloat16).unsqueeze(0),
                )
            upacked_attn_output.append(attn_output.squeeze(0))
        packed_attn_output = torch.cat(upacked_attn_output, dim=1)

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.hidden_size)
        packed_attn_output = self.o_proj(packed_attn_output)

        return packed_attn_output

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
    ):
        packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states = packed_query_states.permute(1, 0, 2).unsqueeze(0)
        packed_key_states = packed_key_states.permute(1, 0, 2).unsqueeze(0)
        packed_query_states, packed_key_states = apply_multimodal_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, self.rope_scaling["mrope_section"], unsqueeze_dim=1
        )
        packed_query_states = packed_query_states.squeeze(0).permute(1, 0, 2)
        packed_key_states = packed_key_states.squeeze(0).permute(1, 0, 2)

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_value_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

        packed_attn_output = flash_attn_varlen_func(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max(query_lens).item(),
            max_seqlen_k=max(key_values_lens).item(),
            causal=is_causal,
        )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        packed_attn_output = self.o_proj(packed_attn_output)

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values


class Qwen2_5_VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5_VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
        #     logger.warning_once(
        #         f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
        #         "unexpected results may be encountered."
        #     )
        self.self_attn = PackedAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        nested_attention_masks: List[torch.Tensor],
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:

        residual = packed_sequence
        packed_sequence = self.input_layernorm(packed_sequence)

        # Self Attention
        packed_sequence = self.self_attn(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            nested_attention_masks=nested_attention_masks,
            packed_position_embeddings=packed_position_embeddings,
        )
        packed_sequence = residual + packed_sequence

        # Fully Connected
        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)
        packed_sequence = self.mlp(packed_sequence)
        packed_sequence = residual + packed_sequence

        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
    ):

        residual = packed_query_sequence
        packed_query_sequence = self.input_layernorm(packed_query_sequence)

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
        packed_query_sequence = self.mlp(packed_query_sequence)
        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


Decoder_layer_dict = {
    "Qwen2_5_VLDecoderLayer": Qwen2_5_VLDecoderLayer,
}


class QwenMMAttention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None,):
        super().__init__()
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.llm_hidden_size = config.hidden_size
        self.vgen_hidden_size = config.vgen_hidden_size
        self.hidden_size = config.mm_attn_hidden_size
        self.num_heads = config.mm_attn_num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.mm_attn_num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_scaling = config.rope_scaling

        self.q_und_proj = nn.Linear(self.llm_hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_und_proj = nn.Linear(self.llm_hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_und_proj = nn.Linear(self.llm_hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_und_proj = nn.Linear(self.num_heads * self.head_dim, self.llm_hidden_size, bias=False)

        self.q_gen_proj = nn.Linear(self.vgen_hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_gen_proj = nn.Linear(self.vgen_hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_gen_proj = nn.Linear(self.vgen_hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_gen_proj = nn.Linear(self.num_heads * self.head_dim, self.vgen_hidden_size, bias=False)

        if config.mm_attn_qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.mm_attn_rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.mm_attn_rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)


    def forward_train(
        self,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        sample_lens: List[int],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor],
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        packed_query_states_und = self.q_und_proj(packed_sequence_und).view(-1, self.num_heads, self.head_dim)
        packed_key_states_und = self.k_und_proj(packed_sequence_und).view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states_und = self.v_und_proj(packed_sequence_und).view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states_gen = self.q_gen_proj(packed_sequence_gen).view(-1, self.num_heads, self.head_dim)
        packed_key_states_gen = self.k_gen_proj(packed_sequence_gen).view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states_gen = self.v_gen_proj(packed_sequence_gen).view(-1, self.num_key_value_heads, self.head_dim)

        sequence_length = len(packed_und_token_indexes) + len(packed_gen_token_indexes)
        packed_query_states = packed_query_states_und.new_zeros(
            size=(sequence_length, self.num_heads, self.head_dim)
        )
        packed_query_states[packed_und_token_indexes] = packed_query_states_und
        packed_query_states[packed_gen_token_indexes] = packed_query_states_gen
        packed_key_states = packed_key_states_und.new_zeros(
            size=(sequence_length, self.num_key_value_heads, self.head_dim)
        )
        packed_key_states[packed_und_token_indexes] = packed_key_states_und
        packed_key_states[packed_gen_token_indexes] = packed_key_states_gen
        packed_value_states = packed_value_states_und.new_zeros(
            size=(sequence_length, self.num_key_value_heads, self.head_dim)
        )
        packed_value_states[packed_und_token_indexes] = packed_value_states_und
        packed_value_states[packed_gen_token_indexes] = packed_value_states_gen

        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states = packed_query_states.permute(1, 0, 2).unsqueeze(0)
        packed_key_states = packed_key_states.permute(1, 0, 2).unsqueeze(0)
        packed_query_states, packed_key_states = apply_multimodal_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, self.rope_scaling["mrope_section"], unsqueeze_dim=1
        )
        packed_query_states = packed_query_states.squeeze(0).permute(1, 0, 2)
        packed_key_states = packed_key_states.squeeze(0).permute(1, 0, 2)

        packed_key_states = packed_key_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
        packed_key_states = packed_key_states.reshape(-1, self.num_heads, self.head_dim)
        packed_value_states = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
        packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

        unpacked_query_states = packed_query_states.transpose(0, 1).split(sample_lens, dim=1)
        unpacked_key_states = packed_key_states.transpose(0, 1).split(sample_lens, dim=1)
        unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)

        upacked_attn_output = []
        for query_states, key_states, value_states, attention_mask in zip(
            unpacked_query_states, unpacked_key_states, unpacked_value_states, nested_attention_masks
        ):
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                attn_output = scaled_dot_product_attention(
                    query_states.to(torch.bfloat16).unsqueeze(0), 
                    key_states.to(torch.bfloat16).unsqueeze(0), 
                    value_states.to(torch.bfloat16).unsqueeze(0),
                    attention_mask.to(torch.bfloat16).unsqueeze(0),
                )
            upacked_attn_output.append(attn_output.squeeze(0))
        packed_attn_output = torch.cat(upacked_attn_output, dim=1)
            # assert torch.allclose(packed_attn_output, flex_packed_attn_output, rtol=1e-2, atol=1e-2)

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.hidden_size)
        packed_attn_output_und = self.o_und_proj(packed_attn_output[packed_und_token_indexes])
        packed_attn_output_gen = self.o_gen_proj(packed_attn_output[packed_gen_token_indexes])
        return packed_attn_output_und, packed_attn_output_gen

    def forward_inference(
        self,
        packed_query_sequence_und: torch.Tensor,
        packed_query_sequence_gen: torch.Tensor,
        query_lens: List[int],
        packed_und_indexes_in_query: torch.LongTensor,
        packed_gen_indexes_in_query: torch.LongTensor,
        packed_query_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        ################
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        ################
        is_causal=True,
        mode="und",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode == "und":
            assert packed_query_sequence_gen is None and packed_gen_indexes_in_query is None, \
                "expect packed_query_sequence_gen and packed_gen_indexes_in_query to be None in und mode!"

            packed_query_states_und = self.q_und_proj(packed_query_sequence_und).view(-1, self.num_heads, self.head_dim)
            packed_key_states_und = self.k_und_proj(packed_query_sequence_und).view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states_und = self.v_und_proj(packed_query_sequence_und).view(-1, self.num_key_value_heads, self.head_dim)

            sequence_length = len(packed_und_indexes_in_query)
            sequence_length2 = sum(query_lens)
            assert sequence_length == sequence_length2, "total sequence_length not matching!"

            packed_query_states = packed_query_states_und
            packed_key_states = packed_key_states_und
            packed_value_states = packed_value_states_und
        elif mode == "gen":
            packed_query_states_und = self.q_und_proj(packed_query_sequence_und).view(-1, self.num_heads, self.head_dim)
            packed_key_states_und = self.k_und_proj(packed_query_sequence_und).view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states_und = self.v_und_proj(packed_query_sequence_und).view(-1, self.num_key_value_heads, self.head_dim)

            packed_query_states_gen = self.q_gen_proj(packed_query_sequence_gen).view(-1, self.num_heads, self.head_dim)
            packed_key_states_gen = self.k_gen_proj(packed_query_sequence_gen).view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states_gen = self.v_gen_proj(packed_query_sequence_gen).view(-1, self.num_key_value_heads, self.head_dim)

            sequence_length = len(packed_und_indexes_in_query) + len(packed_gen_indexes_in_query)
            sequence_length2 = sum(query_lens)
            assert sequence_length == sequence_length2, "total sequence_length not matching!"
            
            packed_query_states = packed_query_states_und.new_zeros(
                size=(sequence_length, self.num_heads, self.head_dim)
            )
            packed_query_states[packed_und_indexes_in_query] = packed_query_states_und
            packed_query_states[packed_gen_indexes_in_query] = packed_query_states_gen
            packed_key_states = packed_key_states_und.new_zeros(
                size=(sequence_length, self.num_key_value_heads, self.head_dim)
            )
            packed_key_states[packed_und_indexes_in_query] = packed_key_states_und
            packed_key_states[packed_gen_indexes_in_query] = packed_key_states_gen
            packed_value_states = packed_value_states_und.new_zeros(
                size=(sequence_length, self.num_key_value_heads, self.head_dim)
            )
            packed_value_states[packed_und_indexes_in_query] = packed_value_states_und
            packed_value_states[packed_gen_indexes_in_query] = packed_value_states_gen
        else:
            raise ValueError(f"wrong inference mode {mode}!")

        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states = packed_query_states.permute(1, 0, 2).unsqueeze(0)
        packed_key_states = packed_key_states.permute(1, 0, 2).unsqueeze(0)
        packed_query_states, packed_key_states = apply_multimodal_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, self.rope_scaling["mrope_section"], unsqueeze_dim=1
        )
        packed_query_states = packed_query_states.squeeze(0).permute(1, 0, 2)
        packed_key_states = packed_key_states.squeeze(0).permute(1, 0, 2)

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_value_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

        packed_attn_output = flash_attn_varlen_func(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max(query_lens).item(),
            max_seqlen_k=max(key_values_lens).item(),
            causal=is_causal,
        )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        if mode == "und":
            packed_attn_output_und = self.o_und_proj(packed_attn_output[packed_und_indexes_in_query])
            packed_attn_output_gen = None
        elif mode == "gen":
            packed_attn_output_und = self.o_und_proj(packed_attn_output[packed_und_indexes_in_query])
            packed_attn_output_gen = self.o_gen_proj(packed_attn_output[packed_gen_indexes_in_query])
        else:
            raise ValueError(f"wrong inference mode {mode}!")

        return packed_attn_output_und, packed_attn_output_gen, past_key_values


class QwenMMAttentionLayer(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None,):
        super().__init__()

        self.mm_attn = QwenMMAttention(config, layer_idx=layer_idx)
        self.llm_norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.vgen_norm = Qwen2RMSNorm(config.vgen_hidden_size, eps=config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)
        
    def forward_train(
        self,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        sample_lens: List[int],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor],
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        residual_und = packed_sequence_und
        residual_gen = packed_sequence_gen
        packed_sequence_und = self.llm_norm(packed_sequence_und)
        packed_sequence_gen = self.vgen_norm(packed_sequence_gen)
        packed_sequence_und, packed_sequence_gen = self.mm_attn(
            packed_sequence_und=packed_sequence_und,
            packed_sequence_gen=packed_sequence_gen,
            sample_lens=sample_lens,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
            nested_attention_masks=nested_attention_masks,
            packed_position_embeddings=packed_position_embeddings,
        )
        packed_sequence_und = residual_und + packed_sequence_und
        packed_sequence_gen = residual_gen + packed_sequence_gen

        return packed_sequence_und, packed_sequence_gen
    
    def forward_inference(
        self,
        packed_query_sequence_und: torch.Tensor,
        packed_query_sequence_gen: torch.Tensor,
        query_lens: List[int],
        packed_und_indexes_in_query: torch.LongTensor,
        packed_gen_indexes_in_query: torch.LongTensor,
        packed_query_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        residual_und = packed_query_sequence_und
        packed_query_sequence_und = self.llm_norm(packed_query_sequence_und)
        if mode == "gen":
            residual_gen = packed_query_sequence_gen
            packed_query_sequence_gen = self.vgen_norm(packed_query_sequence_gen)
        packed_query_sequence_und, packed_query_sequence_gen, past_key_values = self.mm_attn(
            packed_query_sequence_und=packed_query_sequence_und,
            packed_query_sequence_gen=packed_query_sequence_gen,
            query_lens=query_lens,
            packed_und_indexes_in_query=packed_und_indexes_in_query,
            packed_gen_indexes_in_query=packed_gen_indexes_in_query,
            packed_query_position_embeddings=packed_query_position_embeddings,
            ################
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            ################
            is_causal=is_causal,
            mode=mode,
        )
        packed_query_sequence_und = residual_und + packed_query_sequence_und
        if mode == "gen":
            packed_query_sequence_gen = residual_gen + packed_query_sequence_gen

        return packed_query_sequence_und, packed_query_sequence_gen, past_key_values
    

class Qwen2_5_VLTextModel(Qwen2_5_VLPreTrainedModel):
    config_class = Qwen2_5_VLTextConfig

    def __init__(self, config: Qwen2_5_VLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        layer_module = Decoder_layer_dict[config.layer_module]
        self.layers = nn.ModuleList(
            [layer_module(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.standalone_num_vlm_layers = config.standalone_num_vlm_layers
        assert self.standalone_num_vlm_layers <= config.num_hidden_layers, "standalone_num_vlm_layers must be less than or equal to num_hidden_layers!"

        reg_pattern = re.compile(r'^\d+$')
        if config.mm_attn_num_layer_type == "min":
            self.num_mm_attn_layers = min(config.num_hidden_layers, config.vgen_num_hidden_layers) - 1
            self.mm_attn_layers_idxs = [idx for idx in range(self.num_mm_attn_layers)]
        elif config.mm_attn_num_layer_type == "max":
            self.num_mm_attn_layers = max(config.num_hidden_layers, config.vgen_num_hidden_layers) - 1
            self.mm_attn_layers_idxs = [idx for idx in range(self.num_mm_attn_layers)]
        # elif reg_pattern.search(config.mm_attn_num_layer_type):
        #     num_skip_layer = int(config.mm_attn_num_layer_type)
        #     self.mm_attn_layers_idxs = [idx for idx in range(0, max(config.num_hidden_layers, config.vgen_num_hidden_layers) - 1, num_skip_layer)]
        #     self.num_mm_attn_layers = len(self.mm_attn_layers_idxs)
        else:
            raise NotImplementedError(f"got wrong cross_attn_num_layer_type {config.mm_attn_num_layer_type}")
        self.mm_attn_layers = nn.ModuleList(
            [QwenMMAttentionLayer(config, layer_idx) for layer_idx in range(self.num_mm_attn_layers)]
        )

        # self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
        # self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # we do need to use gradient_checkpointing
        # self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)
        
    def forward_train(
        self,
        vgen_model: nn.Module,
        # packed_sequence: torch.Tensor,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        sample_lens: List[int],
        # unpacked_sample_lens_gen: List[int],
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
        vgen_e: torch.Tensor,
        vgen_grid_sizes: torch.Tensor,
        vgen_freqs: torch.Tensor,
        vgen_context: torch.Tensor,
        ########################
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens_gen: List[int] = None,
        sample_lens_und: List[int] = None,
        ########################
    ) -> torch.Tensor:

        # create position embeddings to be shared across the decoder layers
        # only uses packed_sequence_und.device and .dtype
        # cos, sin = self.rotary_emb(packed_sequence_und, packed_position_ids.unsqueeze(0))
        # cos = cos.squeeze(0)
        # sin = sin.squeeze(0)
        # packed_position_embeddings = (cos, sin)

        # packed_cos, packed_sin = packed_position_embeddings
        # packed_position_embeddings_und = (packed_cos[packed_und_token_indexes], packed_sin[packed_und_token_indexes])
        cos, sin = self.rotary_emb(packed_sequence_und, packed_position_ids)
        packed_position_embeddings = (cos, sin)
        packed_position_embeddings_und = (cos[:, :, packed_und_token_indexes, :], sin[:, :, packed_und_token_indexes, :])

        use_flex = not (nested_attention_masks is not None)

        if not use_flex:
            cumsum_sample_lens = np.cumsum([0] + sample_lens)
            und_nested_attention_masks = []
        
            for idx, (cumsum_sample_len_min, cumsum_sample_len_max)  in \
                enumerate(zip(cumsum_sample_lens[:-1], cumsum_sample_lens[1:])):
                und_token_index = packed_und_token_indexes[
                    torch.logical_and(
                        packed_und_token_indexes >= cumsum_sample_len_min,
                        packed_und_token_indexes < cumsum_sample_len_max
                    )
                ]
                min_token_index = torch.min(und_token_index)
                und_token_index_in_sample = und_token_index - min_token_index # to index attention mask
                und_token_index_in_sample = torch.sort(und_token_index_in_sample)[0]

                und_attention_mask = nested_attention_masks[idx][und_token_index_in_sample][:, und_token_index_in_sample]
                und_nested_attention_masks.append(und_attention_mask)
        else:
            und_nested_attention_masks = None

        for idx in range(self.standalone_num_vlm_layers):
            decoder_layer = self.layers[idx]
            packed_sequence_und = decoder_layer(
                packed_sequence=packed_sequence_und,
                nested_attention_masks=und_nested_attention_masks,
                sample_lens=sample_lens_und,
                packed_position_embeddings=packed_position_embeddings_und,
            )
            
        for idx in range(max(len(self.layers) - self.standalone_num_vlm_layers, vgen_model.num_layers)):
            if idx < len(self.layers) - self.standalone_num_vlm_layers:
                decoder_layer = self.layers[idx + self.standalone_num_vlm_layers]
                packed_sequence_und = decoder_layer(
                    packed_sequence=packed_sequence_und,
                    nested_attention_masks=und_nested_attention_masks,
                    sample_lens=sample_lens_und,
                    packed_position_embeddings=packed_position_embeddings_und,
                )

            if idx < vgen_model.num_layers:
                vgen_block = vgen_model.blocks[idx]
                packed_sequence_gen = vgen_block(
                    packed_sequence_gen=packed_sequence_gen,
                    sample_lens=split_lens_gen,
                    vgen_e=vgen_e,
                    vgen_grid_sizes=vgen_grid_sizes,
                    vgen_freqs=vgen_freqs,
                    vgen_context=vgen_context,
                )

            # if idx < min(len(self.layers), vgen_model.num_layers) - 1:
            # if idx < self.num_mm_attn_layers:
            if idx in self.mm_attn_layers_idxs:
                mm_attn_idx = self.mm_attn_layers_idxs.index(idx)
                mm_attn_layer = self.mm_attn_layers[mm_attn_idx]
                packed_sequence_und_, packed_sequence_gen_ = mm_attn_layer(
                    packed_sequence_und=packed_sequence_und,
                    packed_sequence_gen=packed_sequence_gen,
                    sample_lens=sample_lens,
                            packed_und_token_indexes=packed_und_token_indexes,
                    packed_gen_token_indexes=packed_gen_token_indexes,
                    nested_attention_masks=nested_attention_masks,
                    packed_position_embeddings=packed_position_embeddings,
                )
                # TODO: check this implementation
                # if idx + self.standalone_num_vlm_layers < len(self.layers) - 1:
                packed_sequence_und = packed_sequence_und_
                # if idx < vgen_model.num_layers - 1:
                packed_sequence_gen = packed_sequence_gen_

        packed_sequence_und = self.norm(packed_sequence_und)
        return packed_sequence_und, packed_sequence_gen
    

    def forward_inference(
        self,
        vgen_model: nn.Module,
        packed_query_sequence_und: torch.Tensor,
        packed_query_sequence_gen: torch.Tensor,
        query_lens_und: torch.Tensor, # TODO: change this to list[int]
        query_lens_gen: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_und_indexes_in_query: torch.LongTensor,
        packed_gen_indexes_in_query: torch.LongTensor,
        vgen_e: torch.Tensor,
        vgen_grid_sizes: torch.Tensor,
        vgen_freqs: torch.Tensor,
        vgen_context: torch.Tensor,
        ###########################
        packed_und_query_indexes: torch.Tensor,
        packed_uni_query_indexes: torch.Tensor,
        past_und_key_values: Optional[NaiveCache] = None,
        past_uni_key_values: Optional[NaiveCache] = None,
        key_values_lens_und: Optional[torch.Tensor] = None,
        key_values_lens_uni: Optional[torch.Tensor] = None,
        packed_und_key_value_indexes: Optional[torch.Tensor] = None,
        packed_uni_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        ###########################
        is_causal=True,
        mode="und",
    ):

        # create position embeddings to be shared across the decoder layers
        # cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
        # only uses packed_sequence_und.device and .dtype
        try:
            cos, sin = self.rotary_emb(packed_query_sequence_und, packed_query_position_ids)
        except:
            import pdb
            pdb.set_trace()
        packed_query_position_embeddings = (cos, sin)
        packed_query_position_embeddings_und = (cos[:, :, packed_und_indexes_in_query, :], sin[:, :, packed_und_indexes_in_query, :])

        for idx in range(self.standalone_num_vlm_layers):
            decoder_layer = self.layers[idx]
            packed_query_sequence_und, past_und_key_values = decoder_layer(
                packed_query_sequence=packed_query_sequence_und,
                query_lens=query_lens_und,
                packed_query_position_embeddings=packed_query_position_embeddings_und,
                packed_query_indexes=packed_und_query_indexes,
                past_key_values=past_und_key_values,
                key_values_lens=key_values_lens_und,
                packed_key_value_indexes=packed_und_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
            )
        
        for idx in range(max(len(self.layers) - self.standalone_num_vlm_layers, vgen_model.num_layers)):
            if idx < len(self.layers) - self.standalone_num_vlm_layers:
                decoder_layer = self.layers[idx + self.standalone_num_vlm_layers]
                packed_query_sequence_und, past_und_key_values = decoder_layer(
                    packed_query_sequence=packed_query_sequence_und,
                    query_lens=query_lens_und,
                    packed_query_position_embeddings=packed_query_position_embeddings_und,
                    packed_query_indexes=packed_und_query_indexes,
                    past_key_values=past_und_key_values,
                    key_values_lens=key_values_lens_und,
                    packed_key_value_indexes=packed_und_key_value_indexes,
                    update_past_key_values=update_past_key_values,
                    is_causal=is_causal,
                )

            if mode == "gen" and idx < vgen_model.num_layers:
                vgen_block = vgen_model.blocks[idx]
                packed_query_sequence_gen = vgen_block(
                    packed_sequence_gen=packed_query_sequence_gen,
                    sample_lens=query_lens_gen,
                    vgen_e=vgen_e,
                    vgen_grid_sizes=vgen_grid_sizes,
                    vgen_freqs=vgen_freqs,
                    vgen_context=vgen_context,
                )

            # if idx < min(len(self.layers), vgen_model.num_layers) - 1:
            # if idx < self.num_mm_attn_layers:
            if idx in self.mm_attn_layers_idxs:
                mm_attn_idx = self.mm_attn_layers_idxs.index(idx)
                mm_attn_layer = self.mm_attn_layers[mm_attn_idx]
                packed_query_sequence_und_, packed_query_sequence_gen_, past_uni_key_values = mm_attn_layer(
                    packed_query_sequence_und=packed_query_sequence_und,
                    packed_query_sequence_gen=packed_query_sequence_gen,
                    query_lens=query_lens,
                    packed_und_indexes_in_query=packed_und_indexes_in_query,
                    packed_gen_indexes_in_query=packed_gen_indexes_in_query,
                    packed_query_position_embeddings=packed_query_position_embeddings,
                    ##############
                    packed_query_indexes=packed_uni_query_indexes,
                    past_key_values=past_uni_key_values,
                    key_values_lens=key_values_lens_uni,
                    packed_key_value_indexes=packed_uni_key_value_indexes,
                    update_past_key_values=update_past_key_values,
                    ##############
                    is_causal=is_causal,
                    mode=mode,
                )
                # TODO: check this implementation
                # if idx + self.standalone_num_vlm_layers < len(self.layers) - 1:
                packed_query_sequence_und = packed_query_sequence_und_
                # if idx < vgen_model.num_layers - 1:
                packed_query_sequence_gen = packed_query_sequence_gen_

        packed_query_sequence_und = self.norm(packed_query_sequence_und)
        return BaseNavitOutputWithPast(
            packed_query_sequence_und=packed_query_sequence_und,
            packed_query_sequence_gen=packed_query_sequence_gen,
            past_und_key_values=past_und_key_values,
            past_uni_key_values=past_uni_key_values,
        )


class Qwen2_5_VLModel(Qwen2_5_VLPreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}
    config_class = Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.language_model = Qwen2_5_VLTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    # because self.visual is called else where.
    def forward(self, *args, **kwargs):
        return self.language_model.forward(*args, **kwargs)


    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)


class Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # because vit features are extracted in cdt_unified_navit
        self.model = Qwen2_5_VLModel(config)
        # self.model = Qwen2_5_VLTextModel._from_config(config.text_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def zero_init_mm_attn(self):
        for name, param in self.named_parameters():
            und_pattern = re.compile(r'mm_attn_layers\.\d+\.mm_attn.o_und_proj')
            gen_pattern = re.compile(r'mm_attn_layers\.\d+\.mm_attn.o_gen_proj')
            if und_pattern.search(name) or gen_pattern.search(name):
                param.data.zero_()

    def zero_freeze_mm_attn_und_branch(self):
        for name, param in self.named_parameters():
            und_pattern = re.compile(r'mm_attn_layers\.\d+\.mm_attn.o_und_proj')
            if und_pattern.search(name):
                param.data.zero_()
                param.requires_grad = False

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules available throught conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        vgen_model: nn.Module,
        # packed_sequence: torch.Tensor,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        sample_lens: List[int],
        # unpacked_sample_lens_gen: List[int],
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
        vgen_e: torch.Tensor,
        vgen_grid_sizes: torch.Tensor,
        vgen_freqs: torch.Tensor,
        vgen_context: torch.Tensor,
        ########################
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens_gen: List[int] = None,
        sample_lens_und: List[int] = None,
        ########################
    ) -> torch.Tensor:

        outputs = self.model(
            vgen_model=vgen_model,
            # packed_sequence=packed_sequence,
            packed_sequence_und=packed_sequence_und,
            packed_sequence_gen=packed_sequence_gen,
            sample_lens=sample_lens,
            # unpacked_sample_lens_gen=unpacked_sample_lens_gen,
            packed_position_ids=packed_position_ids,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
            vgen_e=vgen_e,
            vgen_grid_sizes=vgen_grid_sizes,
            vgen_freqs=vgen_freqs,
            vgen_context=vgen_context,
            ########################
            nested_attention_masks=nested_attention_masks,
            split_lens_gen=split_lens_gen,
            sample_lens_und=sample_lens_und,
            ########################
        )
        return outputs

    def forward_inference(
        self,
        vgen_model: nn.Module,
        packed_query_sequence_und: torch.Tensor,
        packed_query_sequence_gen: torch.Tensor,
        query_lens_und: torch.Tensor,
        query_lens_gen: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_und_indexes_in_query: torch.LongTensor,
        packed_gen_indexes_in_query: torch.LongTensor,
        vgen_e: torch.Tensor,
        vgen_grid_sizes: torch.Tensor,
        vgen_freqs: torch.Tensor,
        vgen_context: torch.Tensor,
        ###########################
        packed_und_query_indexes: torch.Tensor,
        packed_uni_query_indexes: torch.Tensor,
        past_und_key_values: Optional[NaiveCache] = None,
        past_uni_key_values: Optional[NaiveCache] = None,
        key_values_lens_und: Optional[torch.Tensor] = None,
        key_values_lens_uni: Optional[torch.Tensor] = None,
        packed_und_key_value_indexes: Optional[torch.Tensor] = None,
        packed_uni_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        ###########################
        is_causal=True,
        mode="und",
    ):

        outputs = self.model(
            vgen_model=vgen_model,
            packed_query_sequence_und=packed_query_sequence_und,
            packed_query_sequence_gen=packed_query_sequence_gen,
            query_lens_und=query_lens_und,
            query_lens_gen=query_lens_gen,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_und_indexes_in_query=packed_und_indexes_in_query,
            packed_gen_indexes_in_query=packed_gen_indexes_in_query,
            vgen_e=vgen_e,
            vgen_grid_sizes=vgen_grid_sizes,
            vgen_freqs=vgen_freqs,
            vgen_context=vgen_context,
            ###########################
            packed_und_query_indexes=packed_und_query_indexes,
            packed_uni_query_indexes=packed_uni_query_indexes,
            past_und_key_values=past_und_key_values,
            past_uni_key_values=past_uni_key_values,
            key_values_lens_und=key_values_lens_und,
            key_values_lens_uni=key_values_lens_uni,
            packed_und_key_value_indexes=packed_und_key_value_indexes,
            packed_uni_key_value_indexes=packed_uni_key_value_indexes,
            update_past_key_values=update_past_key_values,
            ###########################
            is_causal=is_causal,
            mode=mode,
        )

        return outputs