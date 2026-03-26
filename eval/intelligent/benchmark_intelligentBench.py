import os
import io
from io import BytesIO
import json
import base64
import argparse
import ast
from collections import defaultdict
import random
from glob import glob
import re
from multiprocessing import Pool
import logging
import threading

from tqdm import tqdm
import numpy as np
import pandas as pd
# import pyarrow.hdfs as hdfs  # hdfs module removed in newer pyarrow versions
import pyarrow.parquet as pq
from PIL import Image
import decord
import openai
from decord import VideoReader
from tenacity import (
    retry,
    stop_after_attempt,
    stop_after_delay,
    wait_random_exponential,
    retry_if_not_exception_type,
    retry_if_exception_type,
    RetryError
)

import time

import pickle
from multiprocessing import Pool, Manager


# Set your OpenAI-compatible API base URL
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

evaluation_prompt = """
You are given a question, the corresponding question image, a human answered image, and the model-generated (AS) answer image. Your task is to evaluate whether the AS answers the question based on the following criteria:

Must Exact Fulfillment of Request: The answer image must fulfi2ll the request made in the question. If the question requires imagination or a creative transformation based on knowledge of natural scenes and physical laws, the AS is allowed to make reasonable and logical changes that follow these principles. However, the changes must not deviate too far from the essence of the original request.

Must Satisfy Completeness: Every element requested in the question must be reasonably present and completed in the answer image. Missing elements should be noted, but some degree of creative interpretation is acceptable as long as the request is overall fulfilled.

Must No Visual Errors: The answer image must not contain major visual errors such as proportion issues, blurriness, or logical inconsistencies. Minor imperfections that do not affect the overall quality or coherence are acceptable, but significant visual errors should be avoided.

Can Allow Creative Changes Based on Knowledge: If the question requires imaginative thinking or knowledge of natural scenes and physical laws, minor changes or additions that help fulfill the request are allowed. These changes should align with the natural world, physical principles, or the context of the question. However, large or inconsistent changes that break the scene's logic or introduce factual inaccuracies are not acceptable.

The human answered image is just an example answer for your reference to understand how to answer this question. The AS does not need to be the same as the human answered image.
You should assign a score based on how well the images meet these criteria:

0: The AS can't be used for answering this question based on previous criteria. Compared with the AS, the human answered image is significantly better.
1: The AS can answer the question, but is worse than the human answered image in terms of quality.
2: The AS can answer the question with similar or better quality than the human answered image.

{
  "score": int, 
  "reason_of_score": "Detailed explanation of the reasoning for the score."
}

Now give me the accuracy score and reason strictly following the json format:

"""





ATTEMPT_NUM = 8
DELAY_TIME = 60 * 5
TIMEOUT_SECONDS = 600  # 10分钟超时
MAX_KEY_RETRIES = 3    # 每个key最多重试次数

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def concatenate_images_horizontally(images):
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_image = Image.new("RGB", (total_width, max_height))
    # 将图像逐一粘贴到新图像中
    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return new_image


def pil2base64(pil_img):
    buf = io.BytesIO()
    # Save the image as a PNG to the buffer
    pil_img.save(buf, format='jpeg')
    # Retrieve the byte data
    image_bytes = buf.getvalue()
    # Encode as base64
    image_base64 = base64.b64encode(image_bytes)
    # Convert bytes to string
    image_base64_str = image_base64.decode('utf-8')
    return image_base64_str


def message_creator(pil_imgs, prompt, sys_prompt="You are a helpful assistant.", detail='low'):
    system_msg = {"role": "system", "content": sys_prompt}
    user_msg = user_message_creator(pil_imgs, prompt, detail)
    return [system_msg, user_msg]


def user_message_creator(pil_imgs, prompt, detail='low'):
    if isinstance(pil_imgs, Image.Image):
        pil_imgs = [pil_imgs]
    
    user_content = []
    if prompt:
        user_content.append({"type": "text", "text": prompt})
    user_msg = {"role": "user", "content": user_content}
    for pil_img in pil_imgs:
        base64_img = pil2base64(pil_img)
        new_msg = {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}", 
                "detail": detail}
        }
        user_content.append(new_msg)
    return user_msg


def pack_data(parsed_item):
    interleaved = []
    flags = []
    #for qa,img in zip(parsed_item['caption'],parsed_item['images']):
    print(parsed_item['caption'])
    
    interleaved.append(parsed_item['caption']['q'])
    interleaved.append(parsed_item['images'][0])
    interleaved.append(parsed_item['caption']['a'])
    interleaved.append(parsed_item['images'][1])
    
    flags.append(["image"])
    flags.append(["text"])
    flags.append(["image"])
    flags.append(["text"])
    
    return interleaved, flags






def user_message_creator_interleaved(parsed_items,detail='low'):

    user_content = []

    #[Image.open(io.BytesIO(img)).convert("RGB") for img in parsed_items['images']]

    #print('len("parsed_items")',len(parsed_items))
    for data_type,item in parsed_items:

        #print("data_type",data_type)
        #return
        if data_type in ["image"]:
            img = Image.open(io.BytesIO(item)).convert("RGB")
            base64_img = pil2base64(img)
            new_msg = {
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}", 
                    "detail": detail}
            }
            user_content.append(new_msg)
            #user_content.append({"type": "text", "text": "nihao"})
        #elif data_type==["text"]:
        else:
            user_content.append({"type": "text", "text": item})
        #else:
        #    return []

    user_msg = {"role": "user", "content": user_content}

    return user_msg


def request(client, messages, temperature=1.0, n=1, worker_id=None, api_key_suffix=None):
    @retry(
        retry=retry_if_not_exception_type(openai.BadRequestError) | retry_if_exception_type((
            openai.APITimeoutError, 
            openai.RateLimitError, 
            openai.APIConnectionError,
            openai.InternalServerError,
            ConnectionError,
            TimeoutError,
            Exception
        )),
        wait=wait_random_exponential(min=2, max=20),
        stop=(stop_after_attempt(ATTEMPT_NUM) | stop_after_delay(DELAY_TIME))
    )
    def completion_with_backoff(**kwargs):
        try:
            logger.info(f"Worker {worker_id} (Key: {api_key_suffix}) - Making API request")
            result = client.chat.completions.create(**kwargs)
            logger.info(f"Worker {worker_id} (Key: {api_key_suffix}) - API request successful")
            return result
        except Exception as e:
            logger.warning(f"Worker {worker_id} (Key: {api_key_suffix}) - API request failed: {type(e).__name__}: {str(e)}")
            raise

    try:
        completion = completion_with_backoff(
            model=client.model,
            messages=messages,
            max_tokens=500,
            temperature=temperature,
            n=n,
        )
        responses = [choice.message.content for choice in completion.choices]
        return responses, completion
    except Exception as e:
        logger.error(f"Worker {worker_id} (Key: {api_key_suffix}) - Final API request failure after all retries: {type(e).__name__}: {str(e)}")
        raise


def sample_mp4_frames(mp4_p, n_frames=None, fps=None):
    if isinstance(mp4_p, str):
        vr = decord.VideoReader(mp4_p)
    elif isinstance(mp4_p, decord.video_reader.VideoReader):
        vr = mp4_p
    video_fps = vr.get_avg_fps()  # 获取视频的帧率
    
    if n_frames is not None:
        frame_indices = np.linspace(0, len(vr)-1, n_frames, dtype=int).tolist()
    else:
        frame_indices = [int(i) for i in np.arange(0, len(vr)-1, video_fps/fps)]
        
        
    frames = vr.get_batch(frame_indices).asnumpy()  # 转换为 numpy 数组
    frames = [Image.fromarray(frame) for frame in frames]
    return frames





        
        
        
def decode_video_byte(video_bytes):
    video_stream = BytesIO(video_bytes)
    vr = decord.VideoReader(video_stream)

    return vr


def recursive_json_loads(data):
    if isinstance(data, str):
        try:
            # 尝试解析字符串
            parsed_data = json.loads(data)
            # 递归解析可能的嵌套结构
            return recursive_json_loads(parsed_data)
        except json.JSONDecodeError:
            # 无法解析时返回原始字符串
            return data
    elif isinstance(data, dict):
        # 对字典里的值递归解析
        return {k: recursive_json_loads(v) for k, v in data.items()}
    elif isinstance(data, list):
        # 对列表中的每个元素递归解析
        return [recursive_json_loads(item) for item in data]
    else:
        # 对于其他数据类型，直接返回
        return data

def parse_data(data):

    if 'caption' not in data or 'images' not in data:
        print("??")
        return []

    if len(data['caption'])<1 or len(data['images']) <2:
        #print("len(data['images'])",len(data['images']))
        #print("len(data['caption'])",len(data['caption']))
        #print("??dsad")
        return []
    parsed_items =[]
    if 'model_out_image' not in data:
        return []
    #print("data.keys()",data.keys())
    parsed_items.append(("text"," question is: "))
    parsed_items.append(("text",data['caption']['q']))
    parsed_items.append(("text"," Question image"))
    parsed_items.append(("image",data['images'][0]))
    parsed_items.append(("text"," GT answer image"))
    parsed_items.append(("image", data['images'][1]))
    parsed_items.append(("text","model-generated (AS) answer image"))
    parsed_items.append(("image",data['model_out_image']))
    #parsed_items.append(("text",data['caption']['a']))
    
    #print("data['number']",data['number'])
    #for i, img in enumerate(data['images']):
    #print("parsed_items",len(parsed_items))
    #return
    return parsed_items


def clean_json_str(json_str):
    # 移除markdown代码块标记
    json_str = re.sub(r'```json\s*', '', json_str)
    json_str = re.sub(r'\s*```', '', json_str)
    # 清理可能的前后空白
    json_str = json_str.strip()
    return json_str

def eval_image_agent(parsed_items, client, worker_id=None, api_key_suffix=None):
    system_msg = {"role": "system", "content": evaluation_prompt}
    user_msg = user_message_creator_interleaved(parsed_items, detail='low')
    messages = [system_msg, user_msg]
    
    try:
        responses, _ = request(client, messages, worker_id=worker_id, api_key_suffix=api_key_suffix)
        json_str = responses[0]
        
        # 清理JSON字符串
        cleaned_json_str = clean_json_str(json_str)
        
        try:
            response_dict = json.loads(cleaned_json_str)
            score = response_dict.get('score')
            reason = response_dict.get('reason_of_score')
            
            return {
                'score': score,
                'reason': reason
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"Worker {worker_id} (Key: {api_key_suffix}) - JSON 解析错误: {e}")
            logger.warning(f"Worker {worker_id} (Key: {api_key_suffix}) - 原始json_str: {json_str}")
            logger.warning(f"Worker {worker_id} (Key: {api_key_suffix}) - 清理后json_str: {cleaned_json_str}")
            
            # 如果 JSON 解析失败，使用正则表达式作为后备方案
            score_pattern = r'"score":\s*(\d+)'
            reason_pattern = r'"reason_of_score":\s*"([^"]*)"'
            
            score_match = re.search(score_pattern, cleaned_json_str)
            reason_match = re.search(reason_pattern, cleaned_json_str)
            
            score = int(score_match.group(1)) if score_match else None
            reason = reason_match.group(1) if reason_match else None
            
            return {
                'score': score,
                'reason': reason
            }
    except Exception as e:
        logger.error(f"Worker {worker_id} (Key: {api_key_suffix}) - eval_image_agent failed: {type(e).__name__}: {str(e)}")
        raise
    # first_res = result.copy()
    # #print("first result is ",result)
    # system_msg = {"role": "system", "content": Second_filter_prompt}
    # # Convert list of tuples to dictionary
    # dict_data = dict(result)
    # #user_content.append({"type": "text", "text": item})

    # # Convert dictionary to JSON string
    # json_str = json.dumps(dict_data)

    # user_msg["content"].append({"type": "text", "text": json_str})
    # messages = [system_msg, user_msg]
    # responses, _ = request(client, messages)
    
    # json_str = responses[0]

    # pattern = r'"([^"]+)":\s*(true|false|"[^"]*"|\d+)'


    # matches = re.findall(pattern, json_str)

    # result = []
    # # 输出结果
    # if matches:
    #     print("提取的键值对:")
    #     for key, value in matches:
    #         print(f"second {key}: {value}")
    # else:
    #     print("未找到匹配的键值对。")
    # result = matches


    # print("second result is ",result)
    
    # return first_res+result

def create_gpt4o_client(api_key):
    """创建GPT-4o客户端"""
    client = openai.AzureOpenAI(
        azure_endpoint=BASE_URL,
        timeout=TIMEOUT_SECONDS,
        api_version="2024-03-01-preview",
        api_key=api_key
    )
    client.model = "gpt-4o-2024-11-20"
    return client

def eval_flow_with_key_retry(parsed_items, api_keys, worker_id):
    """使用多个API key进行重试的评估流程"""
    last_exception = None
    
    for key_idx, api_key in enumerate(api_keys):
        api_key_suffix = api_key[-10:] if len(api_key) > 10 else api_key
        logger.info(f"Worker {worker_id} - Trying key {key_idx + 1}/{len(api_keys)} (suffix: {api_key_suffix})")
        
        try:
            gpt4o_client = create_gpt4o_client(api_key)
            
            for attempt in range(MAX_KEY_RETRIES):
                try:
                    logger.info(f"Worker {worker_id} (Key: {api_key_suffix}) - Attempt {attempt + 1}/{MAX_KEY_RETRIES}")
                    score = eval_image_agent(parsed_items, gpt4o_client, worker_id=worker_id, api_key_suffix=api_key_suffix)
                    
                    if score and score.get('score') is not None:
                        logger.info(f"Worker {worker_id} (Key: {api_key_suffix}) - Success with score: {score.get('score')}")
                        return score
                    else:
                        logger.warning(f"Worker {worker_id} (Key: {api_key_suffix}) - Got invalid score, retrying...")
                        
                except Exception as e:
                    logger.warning(f"Worker {worker_id} (Key: {api_key_suffix}) - Attempt {attempt + 1} failed: {type(e).__name__}: {str(e)}")
                    last_exception = e
                    if attempt < MAX_KEY_RETRIES - 1:
                        time.sleep(2 ** attempt)  # 指数退避
                        
        except Exception as e:
            logger.error(f"Worker {worker_id} (Key: {api_key_suffix}) - Failed to create client or all attempts failed: {type(e).__name__}: {str(e)}")
            last_exception = e
            continue
    
    # 所有key都尝试失败
    logger.error(f"Worker {worker_id} - All API keys exhausted, final failure")
    if last_exception:
        raise last_exception
    else:
        raise Exception("All API keys failed without specific exception")

def eval_flow(parsed_items, gpt4o_client, worker_id=None):
    """保持向后兼容的eval_flow函数"""
    try:
        api_key_suffix = gpt4o_client.api_key[-10:] if len(gpt4o_client.api_key) > 10 else gpt4o_client.api_key
        score = eval_image_agent(parsed_items, gpt4o_client, worker_id=worker_id, api_key_suffix=api_key_suffix)
        return score
    except Exception as e:
        logger.error(f"Worker {worker_id} - eval_flow failed: {type(e).__name__}: {str(e)}")
        raise

import ctypes

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Input parquet file path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of workers, defaults to number of API keys')
    parser.add_argument('--tag_id', type=str, required=True, help='Tag ID for output file naming')
    parser.add_argument('--hdfs', action='store_true', help='Whether to use HDFS')
    return parser.parse_args()

def get_parquet_reader(file_path, is_hdfs=False):
    """获取parquet文件读取器"""
    if is_hdfs:
        raise NotImplementedError("HDFS support removed due to pyarrow.hdfs deprecation")
    else:
        return pq.ParquetFile(file_path)

def get_row_group_boundaries(parquet_file):
    """获取每个行组的起始和结束行号"""
    boundaries = []
    current_row = 0
    for i in range(parquet_file.num_row_groups):
        num_rows = parquet_file.metadata.row_group(i).num_rows
        boundaries.append((current_row, current_row + num_rows))
        current_row += num_rows
    return boundaries

def get_row_groups_for_range(boundaries, start_row, end_row):
    """确定给定行范围需要的行组"""
    needed_groups = []
    row_ranges = []
    total_rows_covered = 0
    
    for group_idx, (group_start, group_end) in enumerate(boundaries):
        if group_start < end_row and group_end > start_row:
            # 计算这个行组中需要处理的实际行范围
            range_start = max(start_row, group_start) - group_start
            range_end = min(end_row, group_end) - group_start
            rows_in_group = range_end - range_start
            
            needed_groups.append((group_idx, range_start, range_end))
            row_ranges.append(rows_in_group)
            total_rows_covered += rows_in_group
    
    expected_rows = end_row - start_row
    if total_rows_covered != expected_rows:
        print(f"Warning: Row coverage mismatch. Expected {expected_rows} rows, got {total_rows_covered} rows.")
        print(f"Range {start_row}-{end_row}, Groups: {needed_groups}")
    
    return needed_groups, sum(row_ranges)

def process_chunk(args):
    chunk_info, api_key, worker_id = args
    start_row, end_row, input_path, is_hdfs = chunk_info
    
    # 初始化
    total_score = 0
    processed_rows = 0
    all_data = []
    
    # 获取parquet读取器
    if is_hdfs:
        raise NotImplementedError("HDFS support removed due to pyarrow.hdfs deprecation")
    else:
        parquet_file = pq.ParquetFile(input_path)
    
    # 获取行组边界
    boundaries = get_row_group_boundaries(parquet_file)
    needed_groups, total_rows_to_process = get_row_groups_for_range(boundaries, start_row, end_row)
    
    # 为每个worker准备多个API key
    all_api_keys = [
        '***REMOVED_API_KEY***',
        '***REMOVED_API_KEY***',
        '***REMOVED_API_KEY***',
        '***REMOVED_API_KEY***',
        '***REMOVED_API_KEY***',
        #'***REMOVED_API_KEY***',
        '***REMOVED_API_KEY***',
        '***REMOVED_API_KEY***',
        '***REMOVED_API_KEY***',
        #'***REMOVED_API_KEY***'
    ]
    
    # 为这个worker分配keys（主key + 备用keys）
    worker_keys = [api_key]  # 主key
    # 添加其他可用keys作为备用
    for backup_key in all_api_keys:
        if backup_key != api_key and backup_key not in worker_keys:
            worker_keys.append(backup_key)
    
    # 进度条
    pbar = tqdm(total=total_rows_to_process,
                desc=f'Worker {worker_id} ({start_row}-{end_row})',
                position=worker_id,
                unit='rows')
    
    current_processed = 0
    
    # 处理每个需要的行组
    for group_idx, range_start, range_end in needed_groups:
        try:
            # 读取行组数据
            table = parquet_file.read_row_group(group_idx)
            df_chunk = table.to_pandas()
            
            # 只处理需要的行范围
            df_chunk = df_chunk.iloc[range_start:range_end]
            
            for _, row in df_chunk.iterrows():
                try:
                    parsed_items = parse_data(row)
                    
                    if len(parsed_items) < 1:  # 无效数据
                        current_processed += 1
                        pbar.update(1)
                        continue
                    
                    # 使用带有key重试的评估流程
                    results = eval_flow_with_key_retry(parsed_items, worker_keys, worker_id)
                    
                    if len(results) < 1:
                        current_processed += 1
                        pbar.update(1)
                        continue
                    
                    # 计算得分    
                    score = int(results['score'])
                    total_score += score
                    processed_rows += 1
                    
                    to_save = row.copy()
                    to_save['answer_image_score'] = results
                    all_data.append(to_save)
                    
                    # 更新进度
                    current_processed += 1
                    pbar.update(1)
                    
                    # 显示当前完成百分比
                    percentage = (current_processed / total_rows_to_process) * 100
                    pbar.set_description(f'Worker {worker_id} ({percentage:.1f}%)')
                    
                except Exception as e:
                    print(f"Error processing row in group {group_idx}: {str(e)}")
                    current_processed += 1
                    pbar.update(1)
                    continue
                
        except Exception as e:
            print(f"Error processing row group {group_idx}: {str(e)}")
            # 更新这个行组中所有行的进度
            rows_in_group = range_end - range_start
            current_processed += rows_in_group
            pbar.update(rows_in_group)
            continue
    
    pbar.close()
    
    return {
        'worker_id': worker_id,
        'processed_count': len(all_data),
        'total_score': total_score,
        'processed_rows': processed_rows,
        'data': all_data
    }



def main():
    args = get_args()
    
    # API keys列表
    api_keys = [
        '***REMOVED_API_KEY***',
        '***REMOVED_API_KEY***',
        '***REMOVED_API_KEY***',
        # '***REMOVED_API_KEY***',
        # '***REMOVED_API_KEY***',
        # '***REMOVED_API_KEY***',
        # '***REMOVED_API_KEY***',
        # '***REMOVED_API_KEY***',
        # '***REMOVED_API_KEY***',
        # '***REMOVED_API_KEY***'
    ]
    
    # 设置工作进程数
    num_workers = args.num_workers if args.num_workers else len(api_keys)
    
    # 获取parquet文件总行数
    if args.hdfs:
        hdfs_client = hdfs.connect()
        parquet_file = pq.ParquetFile(hdfs_client.open(args.input_path))
    else:
        parquet_file = pq.ParquetFile(args.input_path)
    
    total_rows = parquet_file.metadata.num_rows
    print(f"Total rows in file: {total_rows}")
    
    # 计算每个worker处理的行数，确保覆盖所有行
    base_chunk_size = total_rows // num_workers
    remainder = total_rows % num_workers
    
    # 准备任务列表
    tasks = []
    current_start = 0
    for i in range(num_workers):
        # 如果有余数，前remainder个worker多处理一行
        current_chunk_size = base_chunk_size + (1 if i < remainder else 0)
        start_idx = current_start
        end_idx = start_idx + current_chunk_size
        current_start = end_idx
        
        print(f"Worker {i} range: {start_idx} - {end_idx} ({current_chunk_size} rows)")
        
        chunk_info = (start_idx, end_idx, args.input_path, args.hdfs)
        api_key = api_keys[i % len(api_keys)]
        
        tasks.append((chunk_info, api_key, i))
    
    # 使用进程池进行并行处理
    with Pool(num_workers) as pool:
        results = pool.map(process_chunk, tasks)
    
    # 合并所有结果
    all_data = []
    total_processed = 0
    total_score = 0
    total_processed_rows = 0
    
    for result in results:
        all_data.extend(result['data'])
        total_processed += result['processed_count']
        total_score += result['total_score']
        total_processed_rows += result['processed_rows']
    
    # 计算最终得分
    final_score = total_score / (total_processed_rows * 2) if total_processed_rows > 0 else 0
    
    # 保存结果
    if all_data:
        final_df = pd.DataFrame(all_data)
        final_output = os.path.join(args.output_dir, f'final_output_{args.tag_id}.parquet')
        final_df.to_parquet(final_output, index=False)
    
    # 计算每个class的分数
    class_scores = {}
    class_counts = {}
    class_total_scores = {}
    
    if all_data:
        for row in all_data:
            if 'class' in row and 'answer_image_score' in row:
                class_name = row['class']
                score_data = row['answer_image_score']
                
                if isinstance(score_data, dict) and 'score' in score_data:
                    score = score_data['score']
                    if isinstance(score, (int, float)) and 0 <= score <= 2:
                        if class_name not in class_total_scores:
                            class_total_scores[class_name] = 0
                            class_counts[class_name] = 0
                        
                        class_total_scores[class_name] += score
                        class_counts[class_name] += 1
        
        # 计算每个class的最终分数
        for class_name in class_total_scores:
            if class_counts[class_name] > 0:
                class_scores[class_name] = class_total_scores[class_name] / (class_counts[class_name] * 2)
    
    # 保存得分信息
    score_output = os.path.join(args.output_dir, f'final_score_{args.tag_id}.txt')
    with open(score_output, 'w') as f:
        f.write(f"Total processed items: {total_processed}\n")
        f.write(f"Total valid rows processed: {total_processed_rows}\n")
        f.write(f"Total score: {total_score}\n")
        f.write(f"Final score (out of 1): {final_score:.4f}\n")
        f.write(f"Final score (percentage): {final_score * 100:.2f}%\n")
        
        # 添加每个class的分数信息
        f.write(f"\n{'='*50}\n")
        f.write(f"Class-wise Score Analysis:\n")
        f.write(f"{'='*50}\n")
        
        if class_scores:
            for class_name in sorted(class_scores.keys()):
                f.write(f"Class '{class_name}':\n")
                f.write(f"  Sample count: {class_counts[class_name]}\n")
                f.write(f"  Total score: {class_total_scores[class_name]}\n")
                f.write(f"  Final score (out of 1): {class_scores[class_name]:.4f}\n")
                f.write(f"  Final score (percentage): {class_scores[class_name] * 100:.2f}%\n")
                f.write(f"\n")
        else:
            f.write("No class information found in the data.\n")
    
    # 打印结果
    print(f"Total processed items: {total_processed}")
    print(f"Total valid rows processed: {total_processed_rows}")
    print(f"Total score: {total_score}")
    print(f"Final score (out of 1): {final_score:.4f}")
    print(f"Final score (percentage): {final_score * 100:.2f}%")
    
    # 打印每个class的分数
    print(f"\n{'='*50}")
    print(f"Class-wise Score Analysis:")
    print(f"{'='*50}")
    
    if class_scores:
        for class_name in sorted(class_scores.keys()):
            print(f"Class '{class_name}':")
            print(f"  Sample count: {class_counts[class_name]}")
            print(f"  Total score: {class_total_scores[class_name]}")
            print(f"  Final score (out of 1): {class_scores[class_name]:.4f}")
            print(f"  Final score (percentage): {class_scores[class_name] * 100:.2f}%")
            print()
    else:
        print("No class information found in the data.")

if __name__ == '__main__':
    start_time = time.time()
    main()
    execution_time = time.time() - start_time
    print(f"Total execution time: {execution_time:.4f} seconds")
