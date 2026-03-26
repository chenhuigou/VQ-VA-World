# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 Zeyu Wang & Zilong Chen.
# SPDX-License-Identifier: Apache-2.0

import random
import json
import numpy as np
import torch

from .data_utils import (
    qwen2_5_vl_patchify, 
    get_qwen2_5_vl_mrope_index, 
    prepare_attention_mask_per_sample, 
    len2weight
)
from .dataset_info import DATASET_INFO, DATASET_REGISTRY
from .transforms import ImageTransform
from .video_utils import FrameSampler


class DataConfig:
    def __init__(
        self, 
        grouped_datasets, 
        text_cond_dropout_prob=0.1,
        vit_cond_dropout_prob=0.4,
        vae_cond_dropout_prob=0.1,
        vae_image_downsample=[4, 16, 16],
        max_latent_size=32,
        vit_temporal_patch_size=2,
        vit_patch_size=14,
        vit_spatial_merge_size=2,
        vit_tokens_per_second=2,
        max_num_patch_per_side=70,
    ):
        self.grouped_datasets = grouped_datasets
        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.vit_cond_dropout_prob = vit_cond_dropout_prob
        self.vit_temporal_patch_size = vit_temporal_patch_size
        self.vit_patch_size = vit_patch_size
        self.vit_spatial_merge_size = vit_spatial_merge_size
        self.vit_tokens_per_second = vit_tokens_per_second
        self.max_num_patch_per_side = max_num_patch_per_side
        self.vae_cond_dropout_prob = vae_cond_dropout_prob
        self.vae_image_downsample = vae_image_downsample
        self.max_latent_size = max_latent_size


class PackedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        data_config, 
        tokenizer, 
        special_tokens,
        local_rank, 
        world_size, 
        num_workers,
        expected_num_tokens=32768, 
        max_num_tokens_per_sample=16384,
        max_num_tokens=36864,
        prefer_buffer_before=16384,
        max_buffer_size=50,
        data_status=None,
    ):
        super().__init__()
        self.expected_num_tokens = expected_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.prefer_buffer_before = prefer_buffer_before
        self.max_num_tokens = max_num_tokens
        self.max_buffer_size = max_buffer_size
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        for k, v in special_tokens.items():
            setattr(self, k, v)

        grouped_datasets, is_mandatory, grouped_weights = self.build_datasets(
            data_config.grouped_datasets, data_status
        )
        self.grouped_datasets = grouped_datasets
        self.dataset_iters = [iter(dataset) for dataset in grouped_datasets]
        self.is_mandatory = is_mandatory
        self.grouped_weights = grouped_weights
        self.data_config = data_config

    def build_datasets(self, datasets_metainfo, data_status):
        datasets = []
        is_mandatory = []
        grouped_weights = []
        for grouped_dataset_name, dataset_args in datasets_metainfo.items():
            is_mandatory.append(dataset_args.pop('is_mandatory', False))
            grouped_weights.append(dataset_args.pop('weight', 0.0))

            if 'frame_sampler_args' in dataset_args.keys():
                frame_sampler = FrameSampler(**dataset_args.pop('frame_sampler_args'))
                dataset_args['frame_sampler'] = frame_sampler
            if 'image_transform_args' in dataset_args.keys():
                transform = ImageTransform(**dataset_args.pop('image_transform_args'))
                dataset_args['transform'] = transform
            if 'vit_image_transform_args' in dataset_args.keys():
                vit_transform = ImageTransform(**dataset_args.pop('vit_image_transform_args'))
                dataset_args['vit_transform'] = vit_transform

            assert 'dataset_names' in dataset_args.keys()
            dataset_names = dataset_args.pop('dataset_names')
            dataset_args['data_dir_list'] = []
            for item in dataset_names:
                if self.local_rank == 0:
                    print(f'Preparing Dataset {grouped_dataset_name}/{item}')
                meta_info = DATASET_INFO[grouped_dataset_name][item]
                dataset_args['data_dir_list'].append(meta_info['data_dir'])

                if "parquet_info_path" in meta_info.keys():
                    if 'parquet_info' not in dataset_args.keys():
                        dataset_args['parquet_info'] = {}
                    with open(meta_info['parquet_info_path'], 'r') as f:
                        parquet_info = json.load(f)
                    dataset_args['parquet_info'].update(parquet_info)

                if 'json_dir' in meta_info.keys():
                    # parquet/tar with json
                    if 'json_dir_list' not in dataset_args.keys():
                        dataset_args['json_dir_list'] = [meta_info['json_dir']]
                    else:
                        dataset_args['json_dir_list'].append(meta_info['json_dir'])

                if 'jsonl_path' in meta_info.keys():
                    # jsonl with jpeg
                    if 'jsonl_path_list' not in dataset_args.keys():
                        dataset_args['jsonl_path_list'] = [meta_info['jsonl_path']]
                    else:
                        dataset_args['jsonl_path_list'].append(meta_info['jsonl_path'])

                if "rewrite_prompt_dir" in meta_info.keys():
                    if 'rewrite_prompt_dir' not in dataset_args.keys():
                        dataset_args['rewrite_prompt_dir'] = [meta_info['rewrite_prompt_dir']]
                    else:
                        dataset_args['rewrite_prompt_dir'].append(meta_info['rewrite_prompt_dir'])

            resume_data_status = dataset_args.pop('resume_data_status', True)
            if data_status is not None and grouped_dataset_name in data_status.keys() and resume_data_status:
                data_status_per_group = data_status[grouped_dataset_name]
            else:
                data_status_per_group = None
            dataset = DATASET_REGISTRY[grouped_dataset_name](
                dataset_name=grouped_dataset_name,
                tokenizer=self.tokenizer,
                local_rank=self.local_rank,
                world_size=self.world_size,
                num_workers=self.num_workers,
                data_status=data_status_per_group,
                **dataset_args
            )
            datasets.append(dataset)

        return datasets, is_mandatory, grouped_weights

    def set_epoch(self, seed):
        for dataset in self.grouped_datasets:
            dataset.set_epoch(seed)

    def set_sequence_status(self):
        sequence_status = dict(
            curr                        = 0,
            curr_und                    = 0,
            sample_lens                 = list(),
            sample_lens_und             = list(),
            packed_m_position_ids       = [[], [], []],
            nested_attention_masks      = list(),
            split_lens_gen              = list(),
            packed_text_ids             = list(), 
            packed_text_indexes         = list(),
            packed_text_indexes_und     = list(),
            packed_label_ids            = list(),
            und_ce_loss_indexes         = list(),
            ce_loss_weights             = list(),
            vae_image_tensors           = list(),
            vae_latent_shapes           = list(), 
            packed_vae_token_indexes    = list(), 
            packed_timesteps            = list(), 
            packed_vit_tokens           = list(), 
            vit_image_grid_thws         = list(),
            packed_vit_token_indexes    = list(),
            packed_vit_token_indexes_und= list(),
        )
        return sequence_status

    def to_tensor(self, sequence_status):
        data = dict(
            packed_text_ids=torch.tensor(sequence_status['packed_text_ids']),
            packed_text_indexes=torch.tensor(sequence_status['packed_text_indexes']),
            packed_m_position_ids=torch.tensor(sequence_status['packed_m_position_ids']).unsqueeze(1),
        )
        data['nested_attention_masks'] = sequence_status['nested_attention_masks']
        data['sample_lens_und'] = sequence_status['sample_lens_und']
        data['split_lens_gen'] = sequence_status['split_lens_gen']
        data['sample_lens'] = sequence_status['sample_lens']

        # if the model has a convnet vae (e.g., as visual tokenizer)
        if len(sequence_status['vae_image_tensors']) > 0:
            image_tensors = sequence_status.pop('vae_image_tensors')
            image_sizes = [item.shape for item in image_tensors]
            max_image_size = [max(item) for item in list(zip(*image_sizes))]
            padded_images = torch.zeros(size=(len(image_tensors), *max_image_size))
            for i, image_tensor in enumerate(image_tensors):
                padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2], :image_tensor.shape[3]] = image_tensor

            data['padded_images'] = padded_images

            data['patchified_vae_latent_shapes'] = sequence_status['vae_latent_shapes']
            data['packed_vae_token_indexes'] = torch.tensor(sequence_status['packed_vae_token_indexes'])

        # if the model has a vit (e.g., as visual tokenizer)
        if len(sequence_status['packed_vit_tokens']) > 0:
            data['packed_vit_tokens'] = torch.cat(sequence_status['packed_vit_tokens'], dim=0)
            data['packed_vit_token_indexes'] = torch.tensor(sequence_status['packed_vit_token_indexes'])
            data['vit_image_grid_thws'] = torch.tensor(sequence_status['vit_image_grid_thws'])
            data['packed_text_indexes_und'] = torch.tensor(sequence_status['packed_text_indexes_und'])
            data['packed_vit_token_indexes_und'] = torch.tensor(sequence_status['packed_vit_token_indexes_und'])

        # if the model is required to perform visual generation
        if len(sequence_status['packed_timesteps']) > 0:
            data['packed_timesteps'] = torch.tensor(sequence_status['packed_timesteps'])

        # if the model is required to perform text generation
        if len(sequence_status['packed_label_ids']) > 0:
            data['packed_label_ids'] = torch.tensor(sequence_status['packed_label_ids'])
            data['und_ce_loss_indexes'] = torch.tensor(sequence_status['und_ce_loss_indexes'])
            data['ce_loss_weights'] = torch.tensor(sequence_status['ce_loss_weights'])

        return data

    def __iter__(self):
        total_weights = sum(self.grouped_weights)
        assert total_weights > 0.0
        group_cumprobs = [sum(self.grouped_weights[:i + 1]) / total_weights 
                          for i in range(len(self.grouped_weights))]
        sequence_status = self.set_sequence_status()
        batch_data_indexes = []

        buffer = []
        while True:
            # Ensure at least one sample from each group
            if sequence_status['curr'] == 0:
                for group_index, group_iter in enumerate(self.dataset_iters):
                    if self.is_mandatory[group_index]:
                        while True:
                            sample = next(group_iter)
                            # if a sample is too long, skip it
                            num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan'])
                            if num_tokens < self.max_num_tokens_per_sample:
                                sequence_status = self.pack_sequence(sample, sequence_status)
                                batch_data_indexes.append(sample['data_indexes'])
                                break
                            else:
                                print(f"skip a sample with length {num_tokens}")
                                continue

            if sequence_status['curr'] < self.prefer_buffer_before and len(buffer) > 0:
                sample = buffer.pop(0)
                sample_from_buffer = True
            else:
                # sample normally across all groups
                n = random.random()
                group_index = 0
                for i, cumprob in enumerate(group_cumprobs):
                    if n < cumprob:
                        group_index = i
                        break
                sample = next(self.dataset_iters[group_index])
                sample_from_buffer = False

            # if a sample is too long, skip it
            num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan'])
            if num_tokens > self.max_num_tokens_per_sample:
                print(f"skip a sample with length {num_tokens}")
                continue

            if sequence_status['curr'] + num_tokens > self.max_num_tokens:
                if len(buffer) < self.max_buffer_size and not sample_from_buffer:
                    buffer.append(sample)
                else:
                    print(f"Yielding data with length {sum(sequence_status['sample_lens'])}")
                    data = self.to_tensor(sequence_status)
                    data['batch_data_indexes'] = batch_data_indexes
                    yield data
                    sequence_status = self.set_sequence_status()
                    batch_data_indexes = []
                continue

            sequence_status = self.pack_sequence(sample, sequence_status)
            batch_data_indexes.append(sample['data_indexes'])

            if sequence_status['curr'] >= self.expected_num_tokens:
                data = self.to_tensor(sequence_status)
                data['batch_data_indexes'] = batch_data_indexes
                yield data
                sequence_status = self.set_sequence_status()
                batch_data_indexes = []

    def pack_sequence(self, sample, sequence_status):
        image_tensor_list = sample['image_tensor_list']
        text_ids_list = sample['text_ids_list']
        sequence_plan = sample['sequence_plan']

        split_lens, attn_modes = list(), list()
        curr = sequence_status['curr']
        curr_und = sequence_status['curr_und']
        curr_rope_id = 0
        sample_lens, sample_lens_und = 0, 0
        split_lens_und = list()
        split_lens_gen = list()

        for item in sequence_plan:
            curr_split_len = 0
            curr_split_len_und = 0
            curr_split_len_gen = 0

            if item['type'] == 'text':
                text_ids = text_ids_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.text_cond_dropout_prob:
                    continue

                shifted_text_ids = [self.bos_token_id] + text_ids
                sequence_status['packed_text_ids'].extend(shifted_text_ids)
                sequence_status['packed_text_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                sequence_status['packed_text_indexes_und'].extend(range(curr_und, curr_und + len(shifted_text_ids)))
                if item['loss'] == 1:
                    sequence_status['und_ce_loss_indexes'].extend(range(curr_und, curr_und + len(shifted_text_ids)))
                    sequence_status['ce_loss_weights'].extend([len2weight(len(shifted_text_ids))] * len(shifted_text_ids))
                    sequence_status['packed_label_ids'].extend(text_ids + [self.eos_token_id])
                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)
                curr_split_len_und += len(shifted_text_ids)
                curr_und += len(shifted_text_ids)

                # add a <|im_end|> token
                sequence_status['packed_text_ids'].append(self.eos_token_id)
                sequence_status['packed_text_indexes'].append(curr)
                sequence_status['packed_text_indexes_und'].append(curr_und)
                if item['special_token_loss'] == 1: # <|im_end|> may have loss
                    sequence_status['und_ce_loss_indexes'].append(curr_und)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1
                curr_split_len_und += 1
                curr_und += 1

                # update sequence status
                attn_modes.append("causal")
                for packed_position_ids in sequence_status['packed_m_position_ids']:
                    packed_position_ids.extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len

            elif item['type'] == 'vit_image':
                image_tensor = image_tensor_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vit_cond_dropout_prob:  
                    continue

                # add a <|startofimage|> token
                sequence_status['packed_text_ids'].append(self.sov_token_id)
                sequence_status['packed_text_indexes'].append(curr)
                sequence_status['packed_text_indexes_und'].append(curr_und)
                curr += 1
                curr_split_len += 1
                curr_split_len_und += 1
                curr_und += 1

                vit_tokens = qwen2_5_vl_patchify(image_tensor.unsqueeze(0),
                    self.data_config.vit_temporal_patch_size, 
                    self.data_config.vit_patch_size, 
                    self.data_config.vit_spatial_merge_size
                )
                num_img_tokens = vit_tokens.shape[0] // (self.data_config.vit_spatial_merge_size ** 2)
                sequence_status['vit_image_grid_thws'].append(
                    [1,
                    image_tensor.size(1) // self.data_config.vit_patch_size,
                    image_tensor.size(2) // self.data_config.vit_patch_size]
                )
                sequence_status['packed_vit_token_indexes'].extend(range(curr, curr + num_img_tokens))
                sequence_status['packed_vit_token_indexes_und'].extend(range(curr_und, curr_und + num_img_tokens))
                curr += num_img_tokens
                curr_split_len += num_img_tokens
                curr_split_len_und += num_img_tokens
                curr_und += num_img_tokens

                sequence_status['packed_vit_tokens'].append(vit_tokens)

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.eov_token_id)
                sequence_status['packed_text_indexes'].append(curr)
                sequence_status['packed_text_indexes_und'].append(curr_und)
                if item['special_token_loss'] == 1: # <|endofimage|> may have loss
                    sequence_status['und_ce_loss_indexes'].append(curr_und)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1
                curr_split_len_und += 1
                curr_und += 1

                # update sequence status
                attn_modes.append("full")

                for packed_position_ids in sequence_status['packed_m_position_ids']:
                    packed_position_ids.append(curr_rope_id)
                t_index, h_index, w_index = get_qwen2_5_vl_mrope_index(
                    st_idx=curr_rope_id+1,
                    spatial_merge_size=self.data_config.vit_spatial_merge_size,
                    tokens_per_second=self.data_config.vit_tokens_per_second,
                    image_grid_thw=[1, 
                                    image_tensor.size(1) // self.data_config.vit_patch_size,
                                    image_tensor.size(2) // self.data_config.vit_patch_size]
                )
                sequence_status['packed_m_position_ids'][0].extend(t_index)
                sequence_status['packed_m_position_ids'][1].extend(h_index)
                sequence_status['packed_m_position_ids'][2].extend(w_index)
                max_pos_id = max(max(t_index), max(h_index), max(w_index))
                for packed_position_ids in sequence_status['packed_m_position_ids']:
                    packed_position_ids.append(max_pos_id + 1)

                curr_rope_id = max_pos_id + 2

            elif item['type'] == 'vae_image':
                # now both image and video are of shape CTHW
                image_tensor = image_tensor_list.pop(0).unsqueeze(1)
                F, H, W = image_tensor.shape[1:]
                f = 1 + (F - 1) // self.data_config.vae_image_downsample[0]
                h = H // self.data_config.vae_image_downsample[1]
                w = W // self.data_config.vae_image_downsample[2]

                if item['enable_cfg'] == 1 and random.random() < self.data_config.vae_cond_dropout_prob:
                    continue

                # add a <|startofimage|> token
                sequence_status['packed_text_ids'].append(self.sov_token_id)
                sequence_status['packed_text_indexes'].append(curr)
                sequence_status['packed_text_indexes_und'].append(curr_und)
                curr += 1
                curr_split_len += 1
                curr_split_len_und += 1
                curr_und +=1

                # preprocess image
                curr_vae_index = len(sequence_status['vae_image_tensors'])
                sequence_status['vae_image_tensors'].append(image_tensor)

                sequence_status['vae_latent_shapes'].append((f, h, w))

                num_img_tokens = f * w * h
                sequence_status['packed_vae_token_indexes'].extend(range(curr, curr + num_img_tokens))
                if item['loss'] == 1:
                    timestep = np.random.randn()
                else:
                    timestep = float('-inf')

                sequence_status['packed_timesteps'].extend([timestep] * num_img_tokens)
                curr += num_img_tokens
                curr_split_len += num_img_tokens
                curr_split_len_gen += num_img_tokens

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.eov_token_id)
                sequence_status['packed_text_indexes'].append(curr)
                sequence_status['packed_text_indexes_und'].append(curr_und)
                # <|endofimage|> may have loss
                if item['special_token_loss'] == 1:
                    sequence_status['und_ce_loss_indexes'].append(curr_und)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1
                curr_split_len_und += 1
                curr_und += 1

                # update sequence status
                if item['loss'] == 1:
                    attn_modes.append("noise")
                else:
                    attn_modes.append("full")

                for packed_position_ids in sequence_status['packed_m_position_ids']:
                    packed_position_ids.append(curr_rope_id)
                t_index, h_index, w_index = get_qwen2_5_vl_mrope_index(
                    st_idx=curr_rope_id+1,
                    spatial_merge_size=1,
                    tokens_per_second=self.data_config.vit_tokens_per_second,
                    image_grid_thw=[f, h, w]
                )
                sequence_status['packed_m_position_ids'][0].extend(t_index)
                sequence_status['packed_m_position_ids'][1].extend(h_index)
                sequence_status['packed_m_position_ids'][2].extend(w_index)
                max_pos_id = max(max(t_index), max(h_index), max(w_index))
                for packed_position_ids in sequence_status['packed_m_position_ids']:
                    packed_position_ids.append(max_pos_id + 1)

                if item['loss'] == 0:
                    curr_rope_id = max_pos_id + 2

            split_lens.append(curr_split_len)
            sample_lens += curr_split_len

            sample_lens_und += curr_split_len_und
            split_lens_und.append(curr_split_len_und)

            # if using gen mode
            if curr_split_len_gen != 0:
                split_lens_gen.append(curr_split_len_gen)
        
        sequence_status['curr'] = curr
        sequence_status['curr_und'] = curr_und
        sequence_status['sample_lens'].append(sample_lens)
        # prepare attention mask
        sequence_status['nested_attention_masks'].append(
            prepare_attention_mask_per_sample(split_lens, attn_modes)
        )
        sequence_status['sample_lens_und'].append(sample_lens_und)
        sequence_status['split_lens_gen'].extend(split_lens_gen)

        return sequence_status


class SimpleCustomBatch:
    def __init__(self, batch):
        data = batch[0]
        self.batch_data_indexes = data['batch_data_indexes']
        self.sample_lens = data["sample_lens"]
        self.packed_text_ids = data["packed_text_ids"]
        self.packed_text_indexes = data["packed_text_indexes"]
        self.packed_m_position_ids = data["packed_m_position_ids"]

        self.sample_lens_und = data['sample_lens_und']
        self.split_lens_gen = data['split_lens_gen']
        self.nested_attention_masks = data["nested_attention_masks"]

        if "padded_images" in data.keys():
            self.padded_images = data["padded_images"]
            self.patchified_vae_latent_shapes = data["patchified_vae_latent_shapes"]
            self.packed_vae_token_indexes = data["packed_vae_token_indexes"]

        if "packed_vit_tokens" in data.keys():
            self.packed_vit_tokens = data["packed_vit_tokens"]
            self.packed_vit_token_indexes = data["packed_vit_token_indexes"]
            self.vit_image_grid_thws = data["vit_image_grid_thws"]
            self.packed_text_indexes_und = data["packed_text_indexes_und"]
            self.packed_vit_token_indexes_und = data["packed_vit_token_indexes_und"]

        if "packed_timesteps" in data.keys():
            self.packed_timesteps = data["packed_timesteps"]

        if "packed_label_ids" in data.keys():
            self.packed_label_ids = data["packed_label_ids"]
            self.und_ce_loss_indexes = data["und_ce_loss_indexes"]
            self.ce_loss_weights = data["ce_loss_weights"]

    def pin_memory(self):
        self.packed_text_ids = self.packed_text_ids.pin_memory()
        self.packed_text_indexes = self.packed_text_indexes.pin_memory()
        self.packed_m_position_ids = self.packed_m_position_ids.pin_memory()

        self.nested_attention_masks = [item.pin_memory() for item in self.nested_attention_masks]

        if hasattr(self, 'padded_images'):
            self.padded_images = self.padded_images.pin_memory()
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.pin_memory()

        if hasattr(self, 'packed_timesteps'):
            self.packed_timesteps = self.packed_timesteps.pin_memory()

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.pin_memory()
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.pin_memory()
            self.vit_image_grid_thws = self.vit_image_grid_thws.pin_memory()
            self.packed_text_indexes_und = self.packed_text_indexes_und.pin_memory()
            self.packed_vit_token_indexes_und = self.packed_vit_token_indexes_und.pin_memory()

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.pin_memory()
            self.und_ce_loss_indexes = self.und_ce_loss_indexes.pin_memory()
            self.ce_loss_weights = self.ce_loss_weights.pin_memory()

        return self

    def cuda(self, device):
        self.packed_text_ids = self.packed_text_ids.to(device)
        self.packed_text_indexes = self.packed_text_indexes.to(device)
        self.packed_m_position_ids = self.packed_m_position_ids.to(device)

        self.nested_attention_masks = [item.to(device) for item in self.nested_attention_masks]

        if hasattr(self, 'padded_images'):
            self.padded_images = self.padded_images.to(device)
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.to(device)

        if hasattr(self, 'packed_timesteps'):
            self.packed_timesteps = self.packed_timesteps.to(device)

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.to(device)
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.to(device)
            self.vit_image_grid_thws = self.vit_image_grid_thws.to(device)
            self.packed_text_indexes_und = self.packed_text_indexes_und.to(device)
            self.packed_vit_token_indexes_und = self.packed_vit_token_indexes_und.to(device)

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.to(device)
            self.und_ce_loss_indexes = self.und_ce_loss_indexes.to(device)
            self.ce_loss_weights = self.ce_loss_weights.to(device)

        return self

    def to_dict(self):
        data = dict(
            sample_lens = self.sample_lens,
            packed_text_ids = self.packed_text_ids,
            packed_text_indexes = self.packed_text_indexes,
            packed_m_position_ids = self.packed_m_position_ids,
            batch_data_indexes = self.batch_data_indexes,
        )

        data['nested_attention_masks'] = self.nested_attention_masks
        data['sample_lens_und'] = self.sample_lens_und
        data['split_lens_gen'] = self.split_lens_gen

        if hasattr(self, 'padded_images'):
            data['padded_images'] = self.padded_images
            data['patchified_vae_latent_shapes'] = self.patchified_vae_latent_shapes
            data['packed_vae_token_indexes'] = self.packed_vae_token_indexes

        if hasattr(self, 'packed_vit_tokens'):
            data['packed_vit_tokens'] = self.packed_vit_tokens
            data['packed_vit_token_indexes'] = self.packed_vit_token_indexes
            data['vit_image_grid_thws'] = self.vit_image_grid_thws
            data['packed_text_indexes_und'] = self.packed_text_indexes_und
            data['packed_vit_token_indexes_und'] = self.packed_vit_token_indexes_und

        if hasattr(self, 'packed_timesteps'):
            data['packed_timesteps'] = self.packed_timesteps

        if hasattr(self, 'packed_label_ids'):
            data['packed_label_ids'] = self.packed_label_ids
            data['und_ce_loss_indexes'] = self.und_ce_loss_indexes
            data['ce_loss_weights'] = self.ce_loss_weights

        return data


def collate_wrapper():
    def collate_fn(batch):
        return SimpleCustomBatch(batch)
    return collate_fn
