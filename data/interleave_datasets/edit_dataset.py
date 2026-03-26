# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 Zeyu Wang & Zilong Chen.
# SPDX-License-Identifier: Apache-2.0

import io
import pyarrow.parquet as pq
import json
import os
import copy
import pickle

from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image, ImageFile, PngImagePlugin, ImageDraw
import matplotlib.pyplot as plt

from data.data_utils import pil_img2rgb
from data.distributed_iterable_dataset import DistributedIterableDataset
from data.interleave_datasets.interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from data.parquet_utils import get_parquet_data_paths, init_arrow_pf_fs
from data.t2i_dataset import T2IIterableDataset
from data.transforms import (
    crop, 
    decolorization, 
    downscale, 
    inpainting,
    motion_blur_opencv, 
    shuffle_patch,
)

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


SYSTEM_PROMPT = """You should first think about the planning process in the mind and then generate the image.
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here"""

CAPTION_BEGIN_TEMP = [
    "I can generate an image here to visually enhance this paragraph like the following one: ",
    "Let me create a supporting image here to illustrate this concept clearly: ",
    "I can visualize this section with a quick image like the following: ",
    "Here's an image that might help clarify the idea just mentioned: ",
    "A picture here would perfectly demonstrate the point I'm making: ",
    "Allow me to add an illustrative image that complements the explanation: ",
    "I'll generate a visual component to make this notion more tangible: ",
    "Let me insert an image here to further highlight the essence of this paragraph: ",
    "I believe a simple diagram will help anchor this part of the discussion: ",
    "An image could enhance the impact of these words, so here it is: ",
    "Let's visualize this idea with an image to deepen understanding: ",
    "I'll craft a quick visual aid to reinforce the point just made: ",
    "Now, I'll provide an illustrative image to accompany these remarks: ",
    "Placing an image here could offer a clearer perspective on the topic: ",
    "I'll produce a relevant graphic to reinforce the notion mentioned above: ",
    "Let me include an informative picture that aligns with this discussion: ",
    "I think an image at this juncture would be particularly illuminating: ",
    "A quick visual might capture the core idea far better, so let's add it: ",
    "Let me showcase an image here to tie all these ideas together: ",
    "A short illustration here can bring out the key point more vividly: ",
    "Let me provide a concise image to spotlight the main takeaway here: ",
]


class DeAugmentationIteratbleDataset(T2IIterableDataset, InterleavedBaseIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        """
        super().__init__(
            dataset_name, transform, tokenizer, data_dir_list, num_used_data, 
            local_rank=local_rank, world_size=world_size, num_workers=num_workers, data_status=data_status,
        )
        self.editings = [
            'decolorization', 'downscale', 'motion_blur', 'zoom_in', 
            'out_painting', 'in_painting', 'shuffle_patch',
        ]

    def decolorization(self, image):
        image = decolorization(image)
        prompt = random.choice([
            "Colorization: ",
            "给这张图片上色：",
            "将这张图变成彩色图。",
            "Colorize this picture.",
            "Turn this picture into a colorful style",
        ])
        return image, prompt

    def downscale(self, image):
        image = downscale(image, random.random() * 0.25 + 0.25)
        prompt = random.choice([
            "放大这张图。",
            "将这幅图片放大。",
            "Upscale this image. ",
            "Upscale: ",
            "Upsample this image. ",
        ])
        return image, prompt

    def motion_blur(self, image):
        image = motion_blur_opencv(
            image, kernel_size=random.randint(40, 60), angle=random.randint(0, 360)
        )
        prompt = random.choice([
            "去掉图像中的运动模糊。",
            "Deblur this image. ",
            "去模糊：",
            "Deblur: ",
        ])
        return image, prompt

    def zoom_in(self, image):
        w, h = image.size
        new_w = int(w * (random.random() * 0.25 + 0.25))
        new_h = int(h * (random.random() * 0.25 + 0.25))
        image, bbox = crop(image, (new_h, new_w))
        points = []
        for x, y in bbox:
            x = max(0, min(x, w))
            y = max(0, min(y, h))
            points.append((x, y))
        normalized_points = ', '.join(
            [f"{x / w:.3f}, {y / h:.3f}" for x, y in points]
        )
        bbox_prompt = '<bbox> [' + normalized_points + '] </bbox>'
        prompt = random.choice([
            f"将图中区域{bbox_prompt}放大。",
            f"放大区域{bbox_prompt}。",
            f"Upscale the region {bbox_prompt}. ",
            f"Upsample region {bbox_prompt} in the image. ",
            f"Zoom-in to the region {bbox_prompt}.",
        ])
        return image, prompt

    def out_painting(self, image):
        w, h = image.size
        new_w = int(w * (random.random() * 0.25 + 0.25))
        new_h = int(h * (random.random() * 0.25 + 0.25))
        image, bbox = crop(image, (new_h, new_w))
        points = []
        for x, y in bbox:
            x = max(0, min(x, w))
            y = max(0, min(y, h))
            points.append((x, y))
        normalized_points = ', '.join(
            [f"{x / w:.3f}, {y / h:.3f}" for x, y in points]
        )
        bbox_prompt = '<bbox> [' + normalized_points + '] </bbox>'

        prompt = random.choice([
            f"以此图像作为区域{bbox_prompt}内的内容，填补区域外的内容。",
            f"填补区域{bbox_prompt}外的内容。",
            f"Take this image as the content within the area {bbox_prompt} and expand and fill the scene outside the area. ",
            f"expand and fill the scene outside the area {bbox_prompt}. ",
            f"Outpaint {bbox_prompt}: ",
        ])
        return image, prompt

    def in_painting(self, image):
        image = inpainting(
            image, 
            num_splits=(random.randint(3, 5), random.randint(3, 5)),
            blank_ratio=random.random() * 0.3 + 0.6,
        )
        prompt = random.choice([
            "Inpainting: ",
            "图像补全：",
            "还原这幅图像。",
        ])
        return image, prompt

    def shuffle_patch(self, image):
        image = shuffle_patch(
            image, num_splits=(random.randint(3, 5), random.randint(3, 5))
        )

        prompt = random.choice([
            "还原这幅图像。",
            "拼回原图：",
            "Rearrange this image: ",
            "Restore this picture to its original composition. ",
            "Rearrange the image patches to restore the original image. ",
        ])
        return image, prompt

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        while True:
            for parquet_file_path in data_paths_per_worker:
                fs = init_arrow_hdfs_fs()
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))

                    for row_group_id in row_group_ids:
                        df = fr.read_row_group(row_group_id).to_pandas()

                        for row_idx, row in df.iterrows():
                            try:
                                # image_byte = row['image']
                                if isinstance(row['image'], dict):
                                    image_byte = row['image']['bytes']
                                else:
                                    image_byte = row['image'][0]
                                src_img = Image.open(io.BytesIO(image_byte))
                                tgt_img = copy.deepcopy(src_img)

                                edit_name = random.choice(self.editings)
                                edit_ops = getattr(self, edit_name)
                                if edit_name != "zoom_in":
                                    src_img, prompt = edit_ops(src_img)
                                else:
                                    tgt_img, prompt = edit_ops(tgt_img)

                                src_img = pil_img2rgb(src_img)
                                tgt_img = pil_img2rgb(tgt_img)
                            except Exception as e:
                                print(f'Error: {e} in {edit_name}, row#{row_idx} rg#{row_group_id}, {parquet_file_path}')
                                continue

                            sample = self._init_data()
                            self._add_image(sample, src_img, need_loss=False, need_vae=True, need_vit=False)
                            self._add_text(sample, prompt, need_loss=False)
                            self._add_image(sample, tgt_img, need_loss=True, need_vae=False, need_vit=False)

                            yield sample

            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class OmniGenIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    def parse_row(self, row):
        image_num = len(row["image"])
        if image_num > 4:
            return {}
        
        data = self._init_data()
        for idx in range(image_num):
            if idx != image_num - 1:
                data = self._add_image(
                    data, pil_img2rgb(Image.open(io.BytesIO(row["image"][idx]))),
                    need_loss=False, need_vae=True, need_vit=True
                )
            else:
                data = self._add_text(data, row["caption"], need_loss=False)
                data = self._add_image(
                    data, pil_img2rgb(Image.open(io.BytesIO(row["image"][idx]))),
                    need_loss=True, need_vae=False, need_vit=False
                )

        yield data


class OmniEditAugIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    def parse_row(self, row):
        prompt_list = list(row['edited_prompt_list']) + list(row['prompt_rewrite'])
        edit_prompt = random.choice(prompt_list)

        input_img = pil_img2rgb(Image.open(io.BytesIO(row["src_img"])))
        output_img = pil_img2rgb(Image.open(io.BytesIO(row["edited_img"])))

        data = self._init_data()
        data = self._add_image(data, input_img, 
            need_loss=False, need_vae=True, need_vit=True)
        data = self._add_text(data, edit_prompt, need_loss=False)
        data = self._add_image(data, output_img, 
            need_loss=True, need_vae=False, need_vit=False)

        yield data


class UnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    def parse_row(self, row):
        image_num = len(row["image_list"])
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = random.choice(range(image_num - 1))
        max_end = min(start_idx + 3, image_num)
        end_idx = random.choice(range(start_idx + 1, max_end))

        data = self._init_data()
        data = self._add_image(
            data, 
            pil_img2rgb(Image.open(io.BytesIO(row["image_list"][start_idx]))),
            need_loss=False, 
            need_vae=True, 
            need_vit=True,
        )

        if end_idx - start_idx > 1 and random.random() < 0.5: # concat multiple insturction
            if end_idx == image_num - 1:
                end_idx -= 1

            instruction = ""
            for idx in range(start_idx + 1, end_idx + 1):
                instruction += random.choice(row["instruction_list"][idx-1]) + ". "
            data = self._add_text(data, instruction.rstrip(), need_loss=False)
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][end_idx]))),
                need_loss=True, 
                need_vae=False, 
                need_vit=False,
            )
        else:
            for idx in range(start_idx + 1, end_idx + 1):
                instruction = random.choice(row["instruction_list"][idx-1])
                data = self._add_text(data, instruction, need_loss=False)
                if idx != end_idx:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True, 
                        need_vae=True, 
                        need_vit=True,
                    )
                else:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True, 
                        need_vae=False, 
                        need_vit=False,
                    )
        yield data


class UniworldEditWithRewriteIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    def __init__(self, *args, rewrite_prob=0.0, rewrite_prompt_base_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewrite_prompt_base_dir = rewrite_prompt_base_dir
        self.rewrite_prob = rewrite_prob

    def parse_row(self, row, rewrite_prompt):
        data = self._init_data()
        conversations = json.loads(row["conversations"])
        input_images_bytes = row["input_images"]
        output_image_byte = row["output_image"]
        num_input_images = len(input_images_bytes)

        human_conversation = conversations[0]
        gpt_conversation = conversations[1]

        source_image_count = human_conversation["value"].count("<image>\n")
        edit_prompt = human_conversation["value"].replace("<image>\n", "")
        source_image_count = source_image_count + edit_prompt.count("\n<image>")
        edit_prompt = edit_prompt.replace("\n<image>", "")

        for idx in range(num_input_images):
            curr_input_image = pil_img2rgb(Image.open(io.BytesIO(input_images_bytes[idx]["bytes"])))
            data = self._add_image(data, curr_input_image, need_loss=False, need_vae=True, need_vit=True)

        if rewrite_prompt is not None and random.random() < self.rewrite_prob:
            edit_prompt = rewrite_prompt["prompt"]
        data = self._add_text(data, edit_prompt.strip(), need_loss=False)

        output_image = pil_img2rgb(Image.open(io.BytesIO(output_image_byte["bytes"])))
        data = self._add_image(data, output_image, need_loss=True, need_vae=False, need_vit=False,)

        return data

    def __iter__(self):
        file_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            global_row_group_start_id = self.data_status[worker_id][0]
            row_start_id = self.data_status[worker_id][1] + 1
        else:
            global_row_group_start_id = 0
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at global_rg#{global_row_group_start_id}, row#{row_start_id}"
        )

        while True:
            file_paths_per_worker_ = file_paths_per_worker[global_row_group_start_id:]
            for global_row_group_idx, (parquet_file_path, row_group_id) in enumerate(
                file_paths_per_worker_, start=global_row_group_start_id
            ):
                fs = init_arrow_pf_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    try:
                        fr = pq.ParquetFile(f)
                        row_group_lengths = [fr.metadata.row_group(i).num_rows for i in range(fr.num_row_groups)]
                        prefix_sum = sum(row_group_lengths[:row_group_id]) + row_start_id
                        df = fr.read_row_group(row_group_id).to_pandas()
                        df = df.iloc[row_start_id:]
                    except Exception as e:
                        print(f'Error {e} in rg#{row_group_id}, {parquet_file_path}')
                        continue
                    parquet_id = Path(parquet_file_path).stem
                    etype_name = Path(parquet_file_path).parent.stem
                    rewrite_prompt_file = os.path.join(self.rewrite_prompt_base_dir, etype_name, f"{parquet_id}_rewrite.json")
                    if not os.path.exists(rewrite_prompt_file):
                        rewrite_prompts = None
                    else:
                        with open(rewrite_prompt_file) as ff:
                            rewrite_prompts = json.load(ff)

                    for row_idx, row in df.iterrows():
                        global_idx = idx + prefix_sum
                        try:
                            if rewrite_prompts is not None:
                                rewrite_prompt = rewrite_prompts[str(global_idx)]
                            else:
                                rewrite_prompt = None
                                print("[Failed!!!] fail to load rewrite prompt for", parquet_file_path)
                            data = self.parse_row(row, rewrite_prompt)
                            if len(data) == 0:
                                continue
                            data['data_indexes'] = {
                                "data_indexes": [global_row_group_idx, row_idx],
                                "worker_id": worker_id,
                                "dataset_name": self.dataset_name,
                            }
                        except Exception as e:
                            print(f'Error {e} in rg#{row_group_id}, {parquet_file_path}')
                            continue
                        yield data

                    row_start_id = 0
            global_row_group_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class DeAugmentationFluxIteratbleDataset(DeAugmentationIteratbleDataset):
    """Flux variant of DeAugmentation that handles different image byte formats."""
    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        while True:
            for parquet_file_path in data_paths_per_worker:
                fs = init_arrow_pf_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))

                    for row_group_id in row_group_ids:
                        df = fr.read_row_group(row_group_id).to_pandas()

                        for row_idx, row in df.iterrows():
                            try:
                                if isinstance(row['image'], dict):
                                    image_byte = row['image']['bytes']
                                else:
                                    image_byte = row['image'][0]
                                src_img = Image.open(io.BytesIO(image_byte))
                                tgt_img = copy.deepcopy(src_img)

                                edit_name = random.choice(self.editings)
                                edit_ops = getattr(self, edit_name)
                                if edit_name != "zoom_in":
                                    src_img, prompt = edit_ops(src_img)
                                else:
                                    tgt_img, prompt = edit_ops(tgt_img)

                                src_img = pil_img2rgb(src_img)
                                tgt_img = pil_img2rgb(tgt_img)
                            except Exception as e:
                                print(f'Error: {e} in {edit_name}, row#{row_idx} rg#{row_group_id}, {parquet_file_path}')
                                continue

                            sample = self._init_data()
                            self._add_image(sample, src_img, need_loss=False, need_vae=True, need_vit=False)
                            self._add_text(sample, prompt, need_loss=False)
                            self._add_image(sample, tgt_img, need_loss=True, need_vae=False, need_vit=False)

                            yield sample

            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class WebOmniV1IterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    """
    WebOmni V1 dataset for Think-Gen training.
    Supports interleaved web content with thinking/planning capabilities.
    Parent class for ThinkGenV1IterableDataset and ThinkGenV1SingleQIterableDataset.
    """
    def __init__(
        self, *args, cap_loss_ratio=0.1, add_system_prompt=False, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cap_loss_ratio = cap_loss_ratio
        self.add_system_prompt = add_system_prompt

    def get_trunc_idx(self, flags, start_img_idx, end_img_idx):
        image_indices = [i for i, flag in enumerate(flags) if flag == 'image']
        if start_img_idx < 0 or end_img_idx >= len(image_indices) or start_img_idx > end_img_idx:
            raise ValueError("Image index out of range")
        if start_img_idx == 0:
            start_index = 0
        else:
            start_index = image_indices[start_img_idx - 1] + 1
        end_index = image_indices[end_img_idx]
        return start_index, end_index

    def parse_row(self, row):
        interleaved = row['interleaved']
        captions = json.loads(row['extra_info[qwen2_5vl_caption]'])
        flags = row['flag']
        images = row['image']
        image_num = len(images)
        if image_num < 2:
            return {}

        used_image_num = random.randint(2, min(image_num, 4))
        start_img_idx = random.choice(range(0, image_num - used_image_num + 1))
        end_img_idx = start_img_idx + used_image_num - 1

        start_idx, end_idx = self.get_trunc_idx(flags, start_img_idx, end_img_idx)
        select_images = images[start_img_idx:end_img_idx+1]
        select_captions = captions[start_img_idx:end_img_idx+1]

        select_interleaved = interleaved[start_idx:end_idx+1]
        select_flags = flags[start_idx: end_idx+1]

        data = self._init_data()
        if self.add_system_prompt:
            data = self._add_text(data, SYSTEM_PROMPT, need_loss=False, enable_cfg=False)
        img_idx = 0
        for idx, (text, flag) in enumerate(zip(select_interleaved, select_flags)):
            if flag == 'text':
                if idx != len(flags) - 1:
                    data = self._add_text(data, text, need_loss=False)
            elif flag == 'image':
                if idx == 0:
                    data = self._add_image(
                        data, pil_img2rgb(Image.open(io.BytesIO(select_images[img_idx]))),
                        need_loss=False, need_vae=True, need_vit=True,
                    )
                else:
                    if flags[idx - 1] == "image":
                        caption_begin = ""
                    else:
                        caption_begin = random.choice(CAPTION_BEGIN_TEMP)

                    data = self._add_text(
                        data, f"<think>\n{caption_begin}{select_captions[img_idx]}\n</think>",
                        need_loss=random.random() < self.cap_loss_ratio,
                    )
                    data = self._add_image(
                        data,
                        pil_img2rgb(Image.open(io.BytesIO(select_images[img_idx]))),
                        need_loss=True,
                        need_vae=idx < len(select_flags) - 1,
                        need_vit=idx < len(select_flags) - 1,
                    )
                img_idx += 1
        return data


class InstructionWebIterableDataset(WebOmniV1IterableDataset):
    """Dataset for instruction-based ThinkGen (instruct_processed_thinkgen type)."""
    def exponential_decay_sample(self, all_questions, decay_rate=0.75):
        if not all_questions:
            return None
        n = len(all_questions)
        if n == 1:
            return all_questions[0]
        weights = [decay_rate ** i for i in range(n)]
        selected_idx = random.choices(range(n), weights=weights)[0]
        return all_questions[selected_idx]

    def parse_row(self, row):
        question_text = ''
        if 'qas' in row.keys():
            original_question = json.loads(row['qas'])['q']
            if 'rewrite_qas' in row.keys():
                rewrite_questions = [rewrite[0] if isinstance(rewrite, np.ndarray) else rewrite
                                    for rewrite in row['rewrite_qas'] if len(rewrite) > 0]
                all_questions = rewrite_questions + [original_question]
                question_text = self.exponential_decay_sample(all_questions)
            else:
                question_text = original_question

        if 'qa_image' in row.keys():
            images = [pil_img2rgb(Image.open(io.BytesIO(byte))) for byte in row['qa_image']]
            question_image = images[0]
            answer_image = images[1]
        elif 'images' in row.keys():
            images = [pil_img2rgb(Image.open(io.BytesIO(byte))) for byte in row['images']]
            question_image = images[0]
            answer_image = images[1]
        elif 'image' in row.keys():
            question_image = None
            answer_image = pil_img2rgb(Image.open(io.BytesIO(row['image'])))

        data = self._init_data()
        if question_image is not None:
            data = self._add_image(data, question_image,
                need_loss=False, need_vae=True, need_vit=True, enable_cfg=False)
        data = self._add_text(data, question_text, need_loss=False, enable_cfg=False)
        data = self._add_image(data, answer_image,
            need_loss=True, need_vae=False, need_vit=False)
        return data

# TODO: update think data
class ThinkGenV1IterableDataset(WebOmniV1IterableDataset):
    def exponential_decay_sample(self,all_questions, decay_rate=0.75):
        if not all_questions:
            return None

        n = len(all_questions)
        if n == 1:
            return all_questions[0]

        weights = []
        for i in range(n):
            weight = decay_rate ** i
            weights.append(weight)

        selected_idx = random.choices(range(n), weights=weights)[0]

        return all_questions[selected_idx]
  
    def parse_row(self, row):
        web_flag = False 
        question_text = ''
        think = ''
        if 'qas' in row.keys():
            original_question = json.loads(row['qas'])['q']
            if 'rewrite_qas' in row.keys():
                rewrite_questions = [rewrite[0] if isinstance(rewrite, np.ndarray) else rewrite
                                    for rewrite in row['rewrite_qas'] if len(rewrite) > 0]
                all_questions = rewrite_questions + [original_question]
                question_text = self.exponential_decay_sample(all_questions)
            else:
                question_text = original_question      
        if 'think' in row:
            think = row['think']
        elif 'think_v2' in row:
            think = row['think_v2']

        if 'qa_image' in row.keys():
            images = [pil_img2rgb(Image.open(io.BytesIO(byte))) for byte in row['qa_image']]
            question_image = images[0]
            answer_image = images[1]
        elif 'images' in row.keys():
            images = [pil_img2rgb(Image.open(io.BytesIO(byte))) for byte in row['images']]
            question_image = images[0]
            answer_image = images[1]
        elif 'image' in row.keys():
            question_image = None
            answer_image = pil_img2rgb(Image.open(io.BytesIO(row['image'])))

        data = self._init_data()

        if question_image is not None:
            data = self._add_image(data, question_image, 
                need_loss=False, need_vae=True, need_vit=True, enable_cfg=False)
        
        if len(think) > 0:
            if self.add_system_prompt:
                data = self._add_text(data, SYSTEM_PROMPT, need_loss=False, enable_cfg=False)

        data = self._add_text(data, question_text, need_loss=False, enable_cfg=False)
        if len(think) > 0:
            data = self._add_text(data, think, need_loss=random.random() < self.cap_loss_ratio, enable_cfg=False)
        else:
            print("No thinking")
            data = self._add_text(data, " ", need_loss=random.random() < self.cap_loss_ratio, enable_cfg=False)
        data = self._add_image(data, answer_image, 
            need_loss=True, need_vae=False, need_vit=False)

        return data

class ThinkGenV1SingleQIterableDataset(WebOmniV1IterableDataset):
    def exponential_decay_sample(self,all_questions, decay_rate=0.75):
        if not all_questions:
            return None

        n = len(all_questions)
        if n == 1:
            return all_questions[0]

        weights = []
        for i in range(n):
            weight = decay_rate ** i
            weights.append(weight)

        selected_idx = random.choices(range(n), weights=weights)[0]

        return all_questions[selected_idx]
  
    def parse_row(self, row):
        web_flag = False 
        question_text = ''
        think = ''
        if 'qas' in row.keys():
            original_question = json.loads(row['qas'])['q']
            question_text = original_question      
        if 'think' in row:
            think = row['think']
        elif 'think_v2' in row:
            think = row['think_v2']

        if 'qa_image' in row.keys():
            images = [pil_img2rgb(Image.open(io.BytesIO(byte))) for byte in row['qa_image']]
            question_image = images[0]
            answer_image = images[1]
        elif 'images' in row.keys():
            images = [pil_img2rgb(Image.open(io.BytesIO(byte))) for byte in row['images']]
            question_image = images[0]
            answer_image = images[1]
        elif 'image' in row.keys():
            question_image = None
            answer_image = pil_img2rgb(Image.open(io.BytesIO(row['image'])))

        data = self._init_data()

        if question_image is not None:
            data = self._add_image(data, question_image, 
                need_loss=False, need_vae=True, need_vit=True, enable_cfg=False)
        
        if len(think) > 0:
            if self.add_system_prompt:
                data = self._add_text(data, SYSTEM_PROMPT, need_loss=False, enable_cfg=False)

        data = self._add_text(data, question_text, need_loss=False, enable_cfg=False)
        if len(think) > 0:
            data = self._add_text(data, think, need_loss=random.random() < self.cap_loss_ratio, enable_cfg=False)
        else:
            print("No thinking")
            data = self._add_text(data, " ", need_loss=random.random() < self.cap_loss_ratio, enable_cfg=False)
        data = self._add_image(data, answer_image, 
            need_loss=True, need_vae=False, need_vit=False)

        return data
