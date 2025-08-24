# -*- coding: utf-8 -*-
"""
OCRFlux Pipeline 核心模块

这是OCRFlux项目的核心处理管道，负责将PDF和图像文件转换为Markdown格式的文本。
该模块实现了OCRFlux的三阶段处理流程：

1. Stage 1: Page to Markdown (页面转Markdown)
   - 将PDF/图像的每一页转换为Markdown格式
   - 处理自然文本、图像和HTML表格
   - 支持图像旋转和尺寸调整

2. Stage 2: Element Merge Detect (元素合并检测)
   - 检测跨页面的元素（如文本段落、表格）
   - 识别需要合并的相邻页面元素
   - 为跨页表格合并做准备

3. Stage 3: HTML Table Merge (HTML表格合并)
   - 合并跨页面的HTML表格
   - 保持表格结构的完整性
   - 生成最终的文档文本

模块特性：
- 异步并发处理，支持多worker并行
- 内置重试机制和错误处理
- 实时指标监控和工作状态跟踪
- 支持本地模型和HuggingFace模型
- 集成vLLM服务器管理

启动流程：
1. 解析命令行参数
2. 检查系统依赖（poppler、vLLM、torch GPU）
3. 下载/加载模型
4. 初始化工作队列
5. 启动vLLM服务器
6. 创建worker任务并发处理
7. 监控处理进度和指标
"""

import argparse
import asyncio
import atexit
import base64
import json
import logging
import shutil
import os
import copy
import random
import re
import sys
import time
from concurrent.futures.process import BrokenProcessPool
from io import BytesIO
from urllib.parse import urlparse

import httpx
from huggingface_hub import snapshot_download
from PIL import Image
from pypdf import PdfReader
from tqdm import tqdm

# 导入OCRFlux内部模块
from ocrflux.check import (
    check_poppler_version,    # 检查poppler版本（PDF处理依赖）
    # check_vllm_version,       # 检查vLLM版本（推理服务依赖）
    # check_torch_gpu_available, # 检查GPU可用性
)
from ocrflux.image_utils import get_page_image, is_image  # 图像处理工具
from ocrflux.table_format import table_matrix2html       # 表格格式转换
from ocrflux.metrics import MetricsKeeper, WorkerTracker # 指标监控和工作状态跟踪
from ocrflux.prompts import PageResponse, build_page_to_markdown_prompt, build_element_merge_detect_prompt, build_html_table_merge_prompt  # 提示词构建
from ocrflux.work_queue import LocalWorkQueue, WorkQueue # 工作队列管理

# ================================
# 日志配置
# ================================
# 初始化主日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG
logger.propagate = False        # 防止日志向上级传播

# 初始化vLLM专用日志记录器
vllm_logger = logging.getLogger("vllm")
vllm_logger.propagate = False

# 配置文件日志处理器 - 记录所有DEBUG级别的日志到文件
file_handler = logging.FileHandler("OCRFlux-debug.log", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# 配置控制台日志处理器 - 只显示INFO级别及以上的日志
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# 将处理器添加到日志记录器
logger.addHandler(file_handler)
logger.addHandler(console_handler)
vllm_logger.addHandler(file_handler)

# 抑制pypdf的冗余日志输出
logging.getLogger("pypdf").setLevel(logging.ERROR)

# ================================
# 全局变量 - 性能监控
# ================================
# 指标收集器：监控token处理速度等性能指标，窗口期为5分钟
metrics = MetricsKeeper(window=60 * 5)
# 工作状态跟踪器：监控各个worker的工作状态和进度
tracker = WorkerTracker()

# ================================
# Stage 1: Page to Markdown 查询构建
# ================================
def build_page_to_markdown_query(args, pdf_path: str, page_number: int, target_longest_image_dim: int, image_rotation: int = 0) -> dict:
    """
    构建页面转Markdown的查询请求
    
    这是OCRFlux三阶段处理流程的第一阶段，负责将PDF/图像页面转换为Markdown格式。
    
    Args:
        args: 命令行参数对象，包含模型配置
        pdf_path: PDF文件路径或图像文件路径
        page_number: 页面编号（从1开始）
        target_longest_image_dim: 图像最长边的目标尺寸
        image_rotation: 图像旋转角度（0, 90, 180, 270度）
    
    Returns:
        dict: 包含模型、消息和参数的查询字典，用于发送给vLLM服务器
    
    处理流程：
    1. 验证图像旋转角度的有效性
    2. 获取指定页面的图像（支持PDF页面提取和图像文件）
    3. 将图像转换为base64编码
    4. 构建包含提示词和图像的多模态查询
    """
    assert image_rotation in [0, 90, 180, 270], "Invalid image rotation provided in build_page_query"

    # 获取页面图像，支持PDF页面提取和直接图像文件
    image = get_page_image(pdf_path, page_number, target_longest_image_dim=target_longest_image_dim, image_rotation=image_rotation)
    
    # 将图像转换为base64编码，用于多模态模型输入
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # 构建多模态查询，包含文本提示词和图像
    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_page_to_markdown_prompt()},  # 页面转Markdown的提示词
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},  # base64编码的图像
                ],
            }
        ],
        "temperature": 0.0,  # 使用确定性输出
    }

# ================================
# Stage 2: Element Merge Detect 查询构建
# ================================
def build_element_merge_detect_query(args,text_list_1,text_list_2) -> dict:
    """
    构建元素合并检测的查询请求
    
    这是OCRFlux三阶段处理流程的第二阶段，负责识别跨页面的元素（如段落、表格）
    并确定哪些元素需要合并。
    
    Args:
        args: 命令行参数对象，包含模型配置
        text_list_1: 第一个页面的文本元素列表
        text_list_2: 第二个页面的文本元素列表
    
    Returns:
        dict: 包含模型、消息和参数的查询字典，用于发送给vLLM服务器
    
    功能说明：
    - 分析相邻两个页面的内容
    - 识别跨页面的文本段落、表格等元素
    - 返回需要合并的元素信息
    """
    image = Image.new('RGB', (28, 28), color='black')

    buffered = BytesIO()
    image.save(buffered, format="PNG")

    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_element_merge_detect_prompt(text_list_1,text_list_2)},  # 元素合并检测提示词
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        "temperature": 0.0,  # 使用确定性输出
    }

# ================================
# Stage 3: HTML Table Merge 查询构建
# ================================
def build_html_table_merge_query(args,text_1,text_2) -> dict:
    """
    构建HTML表格合并的查询请求
    
    这是OCRFlux三阶段处理流程的第三阶段，负责合并跨页面的HTML表格。
    
    Args:
        args: 命令行参数对象，包含模型配置
        text_1: 第一个HTML表格的字符串
        text_2: 第二个HTML表格的字符串
    
    Returns:
        dict: 包含模型、消息和参数的查询字典，用于发送给vLLM服务器
    
    功能说明：
    - 分析两个HTML表格的结构和内容
    - 智能合并表格行和列
    - 保持表格格式的一致性
    """
    image = Image.new('RGB', (28, 28), color='black')

    buffered = BytesIO()
    image.save(buffered, format="PNG")

    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_html_table_merge_prompt(text_1,text_2)},  # HTML表格合并提示词
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        "temperature": 0.0,  # 使用确定性输出
    }

# ================================
# HTTP通信工具函数
# ================================
async def apost(url, json_data):
    """
    异步HTTP POST请求工具函数
    
    用于与vLLM服务器进行通信，发送查询请求并获取响应。
    采用手动实现的HTTP客户端，避免复杂HTTP库在大规模请求时的死锁问题。
    
    Args:
        url: 目标URL地址（通常是vLLM服务器的API端点）
        json_data: 要发送的JSON数据（包含模型查询信息）
    
    Returns:
        tuple: (status_code, response_body) HTTP状态码和响应体
    
    异常处理：
    - ConnectionError: 服务器连接失败
    - ValueError: HTTP响应格式错误
    - 自动关闭网络连接资源
    
    注意：
    在100M+请求规模下，httpx和aiohttp等复杂HTTP库容易出现死锁，
    因此采用简单的手动实现来确保稳定性。
    """
    parsed_url = urlparse(url)
    host = parsed_url.hostname
    port = parsed_url.port or 80
    path = parsed_url.path or "/"

    writer = None
    try:
        reader, writer = await asyncio.open_connection(host, port)

        json_payload = json.dumps(json_data)
        request = (
            f"POST {path} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(json_payload)}\r\n"
            f"Connection: close\r\n\r\n"
            f"{json_payload}"
        )
        writer.write(request.encode())
        await writer.drain()

        # Read status line
        status_line = await reader.readline()
        if not status_line:
            raise ConnectionError("No response from server")
        status_parts = status_line.decode().strip().split(" ", 2)
        if len(status_parts) < 2:
            raise ValueError(f"Malformed status line: {status_line.decode().strip()}")
        status_code = int(status_parts[1])

        # Read headers
        headers = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            key, _, value = line.decode().partition(":")
            headers[key.strip().lower()] = value.strip()

        # Read response body
        if "content-length" in headers:
            body_length = int(headers["content-length"])
            response_body = await reader.readexactly(body_length)
        else:
            raise ConnectionError("Anything other than fixed content length responses are not implemented yet")

        return status_code, response_body
    except Exception as e:
        # Pass through errors
        raise e
    finally:
        # But just make sure to close the socket on your way out
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass

# ================================
# 核心任务处理函数
# ================================
async def process_task(args, worker_id, task_name, task_args):
    """
    处理单个任务的核心函数，支持重试机制和指数退避
    
    这是OCRFlux的核心处理函数，负责协调三个阶段的处理流程：
    1. page_to_markdown: 页面转Markdown（Stage 1）
    2. element_merge_detect: 元素合并检测（Stage 2）
    3. html_table_merge: HTML表格合并（Stage 3）
    
    Args:
        args: 命令行参数对象，包含模型和服务器配置
        worker_id: 工作进程ID，用于日志跟踪
        task_name: 任务类型名称
        task_args: 任务参数元组
    
    Returns:
        处理结果数据，失败时返回None
    
    重试机制：
    - 最大重试次数：由args.max_page_retries控制
    - 温度递增：每次重试增加temperature以克服重复问题
    - 指数退避：网络错误时使用指数退避策略
    
    性能监控：
    - 记录token使用量
    - 跟踪工作状态
    - 异常处理和日志记录
    """
    # COMPLETION_URL = f"http://localhost:{args.port}/v1/chat/completions"
    COMPLETION_URL = f"http://192.168.8.36:8004/v1/chat/completions"
    MAX_RETRIES = args.max_page_retries
    TEMPERATURE_BY_ATTEMPT = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    exponential_backoffs = 0
    local_image_rotation = 0
    attempt = 0
    await tracker.track_work(worker_id, f"{worker_id}", "started")
    
    while attempt < MAX_RETRIES:
        # 根据任务类型构建相应的查询
        if task_name == 'page_to_markdown':
            # Stage 1: 页面转Markdown
            pdf_path,page_number = task_args
            query = build_page_to_markdown_query(args, pdf_path, page_number, args.target_longest_image_dim, image_rotation=local_image_rotation)
        elif task_name == 'element_merge_detect':
            # Stage 2: 元素合并检测
            text_list_1,text_list_2 = task_args
            query = build_element_merge_detect_query(args, text_list_1, text_list_2)
        elif task_name == 'html_table_merge':
            # Stage 3: HTML表格合并
            table_1,table_2 = task_args
            query = build_html_table_merge_query(args, table_1, table_2)
        
        # 随着重试次数增加temperature，以克服重复问题（以质量为代价）
        query["temperature"] = TEMPERATURE_BY_ATTEMPT[
            min(attempt, len(TEMPERATURE_BY_ATTEMPT) - 1)
        ]

        try:
            # 发送请求到vLLM服务器
            status_code, response_body = await apost(COMPLETION_URL, json_data=query)

            # 检查HTTP状态码
            if status_code == 400:
                raise ValueError(f"Got BadRequestError from server: {response_body}, skipping this response")
            elif status_code == 500:
                raise ValueError(f"Got InternalServerError from server: {response_body}, skipping this response")
            elif status_code != 200:
                raise ValueError(f"Error http status {status_code}")

            # 解析响应数据
            base_response_data = json.loads(response_body)

            # 更新性能指标
            metrics.add_metrics(
                vllm_input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
                vllm_output_tokens=base_response_data["usage"].get("completion_tokens", 0),
            )

            # 提取模型响应内容
            response_content = base_response_data["choices"][0]["message"]["content"]
            
            # 根据任务类型处理响应内容
            if task_name == 'page_to_markdown':
                # 处理页面转Markdown的响应
                model_response_json = json.loads(response_content)
                page_response = PageResponse(**model_response_json)
                natural_text = page_response.natural_text
                markdown_element_list = []
                for text in natural_text.split('\n\n'):
                    if text.startswith("<Image>") and text.endswith("</Image>"):
                        # 跳过图像标记
                        pass
                    elif text.startswith("<table>") and text.endswith("</table>"):
                        # 处理表格格式转换
                        try:
                            new_text = table_matrix2html(text)
                        except:
                            new_text = text.replace("<t>","").replace("<l>","").replace("<lt>","")
                        markdown_element_list.append(new_text)
                    else:
                        markdown_element_list.append(text)
                return_data = "\n\n".join(markdown_element_list)
                    
            elif task_name == 'element_merge_detect':
                # 处理元素合并检测的响应
                return_data = eval(response_content)
            elif task_name == 'html_table_merge':
                # 处理HTML表格合并的响应
                if not (response_content.startswith("<table>") and response_content.endswith("</table>")):
                    raise ValueError("Response is not a table")
                return_data = response_content
            else:
                raise ValueError(f"Unknown task_name {task_name}")
            
            # 标记任务完成
            await tracker.track_work(worker_id, f"{worker_id}", "finished")
            return return_data
        
        except (ConnectionError, OSError, asyncio.TimeoutError) as e:
            # 网络连接错误，使用指数退避而不计入页面重试次数
            logger.warning(f"Client error on attempt {attempt} for {worker_id}: {type(e)} {e}")

            # 指数退避策略：页面重试用于修复模型的错误结果，但vLLM请求应该正常工作
            # 如果出现连接错误，可能是服务器正在重启
            sleep_delay = 10 * (2**exponential_backoffs)
            exponential_backoffs += 1
            logger.info(f"Sleeping for {sleep_delay} seconds on {worker_id} to allow server restart")
            await asyncio.sleep(sleep_delay)
        except asyncio.CancelledError:
            # 任务被取消
            logger.info(f"Process {worker_id} cancelled")
            await tracker.track_work(worker_id, f"{worker_id}", "cancelled")
            raise
        except json.JSONDecodeError as e:
            # JSON解析错误，计入重试次数
            logger.warning(f"JSON decode error on attempt {attempt} for {worker_id}: {e}")
            attempt += 1
        except ValueError as e:
            # 值错误，计入重试次数
            logger.warning(f"ValueError on attempt {attempt} for {worker_id}: {type(e)} - {e}")
            attempt += 1
        except Exception as e:
            # 其他未预期的错误，计入重试次数
            logger.exception(f"Unexpected error on attempt {attempt} for {worker_id}: {type(e)} - {e}")
            attempt += 1

    # 所有重试都失败
    logger.error(f"Failed to process {worker_id} after {MAX_RETRIES} attempts.")
    await tracker.track_work(worker_id, f"{worker_id}", "errored")

    return None

# ================================
# 文本后处理函数
# ================================
def postprocess_markdown_text(args, response_text, pdf_path, page_number):
    """
    对页面转Markdown的响应文本进行后处理
    
    主要功能：
    1. 分割文本为段落列表
    2. 过滤掉图像标记（<Image>...</Image>）
    3. 保留其他所有文本内容
    
    Args:
        args: 命令行参数对象
        response_text: 模型返回的原始文本
        pdf_path: PDF文件路径
        page_number: 页面编号
    
    Returns:
        str: 清理后的Markdown文本
    
    处理逻辑：
    - 按双换行符分割文本为段落
    - 跳过图像标记段落
    - 重新组合剩余段落
    """
    # 打印阶段一的输出
    print(f"\n=== 阶段一输出 - 页面 {page_number} ===")
    print(f"文件路径: {pdf_path}")
    print(f"原始响应文本:")
    print(response_text)
    print("=" * 50)
    
    text_list = response_text.split("\n\n")
    new_text_list = []
    for text in text_list:
        if text.startswith("<Image>") and text.endswith("</Image>"):
            # 跳过图像标记，这些在最终文档中不需要
            pass
        else:
            new_text_list.append(text)
    return "\n\n".join(new_text_list)

# ================================
# 文档构建函数
# ================================
def build_document_text(page_to_markdown_result, element_merge_detect_result, html_table_merge_result):
    """
    从三个阶段的处理结果构建最终文档文本
    
    这是OCRFlux处理流程的最后一步，整合三个阶段的结果：
    1. Stage 1: page_to_markdown_result - 页面转Markdown的结果
    2. Stage 2: element_merge_detect_result - 元素合并检测的结果
    3. Stage 3: html_table_merge_result - HTML表格合并的结果
    
    Args:
        page_to_markdown_result: 字典，键为页面编号，值为文本元素列表
        element_merge_detect_result: 字典，键为页面对，值为需要合并的元素索引对列表
        html_table_merge_result: 字典，键为表格合并标识，值为合并后的HTML表格
    
    Returns:
        str: 最终的完整文档Markdown文本
    
    处理逻辑：
    1. 首先处理HTML表格合并（Stage 3结果）
    2. 然后处理跨页面元素合并（Stage 2结果）
    3. 最后组装所有页面的文本内容
    
    合并策略：
    - 表格合并：将合并后的表格放在第一个位置，清空第二个位置
    - 文本合并：根据文本结尾特征决定是否添加空格连接
    - 中文文本、连字符结尾文本：直接连接
    - 其他文本：用空格连接
    """
    page_to_markdown_keys = list(page_to_markdown_result.keys())
    element_merge_detect_keys = list(element_merge_detect_result.keys())
    html_table_merge_keys = list(html_table_merge_result.keys())

    # Stage 3: 处理HTML表格合并
    # 按页面编号倒序处理，避免索引变化影响后续操作
    for page_1,page_2,elem_idx_1,elem_idx_2 in sorted(html_table_merge_keys,key=lambda x: -x[0]):
        page_to_markdown_result[page_1][elem_idx_1] = html_table_merge_result[(page_1,page_2,elem_idx_1,elem_idx_2)]
        page_to_markdown_result[page_2][elem_idx_2] = ''

    # Stage 2: 处理跨页面元素合并
    # 按页面编号倒序处理，避免索引变化影响后续操作
    for page_1,page_2 in sorted(element_merge_detect_keys,key=lambda x: -x[0]):
        for elem_idx_1,elem_idx_2 in element_merge_detect_result[(page_1,page_2)]:
            # 智能文本连接：根据第一个文本的结尾特征决定连接方式
            if len(page_to_markdown_result[page_1][elem_idx_1]) == 0 or page_to_markdown_result[page_1][elem_idx_1][-1] == '-' or ('\u4e00' <= page_to_markdown_result[page_1][elem_idx_1][-1] <= '\u9fff'):
                # 空文本、连字符结尾、中文字符结尾：直接连接
                page_to_markdown_result[page_1][elem_idx_1] = page_to_markdown_result[page_1][elem_idx_1] + '' + page_to_markdown_result[page_2][elem_idx_2]
            else:
                # 其他情况：用空格连接
                page_to_markdown_result[page_1][elem_idx_1] = page_to_markdown_result[page_1][elem_idx_1] + ' ' + page_to_markdown_result[page_2][elem_idx_2]
            # 清空第二个位置的文本，避免重复
            page_to_markdown_result[page_2][elem_idx_2] = ''
    
    # 组装最终文档文本
    document_text_list = []
    for page in page_to_markdown_keys:
        # 过滤掉空字符串，只保留有内容的文本段落
        page_text_list = [s for s in page_to_markdown_result[page] if s]
        document_text_list += page_text_list
    return "\n\n".join(document_text_list)

async def process_pdf(args, worker_id: int, pdf_path: str):
    """
    PDF处理的主协调函数 - OCRFlux三阶段处理的核心入口
    
    功能说明：
    - 协调执行OCRFlux的完整三阶段处理流程
    - Stage 1: Page to Markdown - 将每页转换为Markdown格式
    - Stage 2: Element Merge Detect - 检测跨页元素合并需求
    - Stage 3: HTML Table Merge - 合并跨页HTML表格
    - 处理错误页面和回退机制
    - 构建最终文档文本
    
    参数：
        args: 命令行参数对象，包含处理配置
        worker_id: 工作进程ID，用于日志标识
        pdf_path: PDF文件路径或图像文件路径
    
    返回值：
        dict: 包含处理结果的字典
            - orig_path: 原始文件路径
            - num_pages: 页面总数
            - document_text: 最终文档文本
            - page_texts: 各页面文本字典
            - fallback_pages: 处理失败的页面列表
        None: 处理失败时返回
    
    处理流程：
    1. 获取文档页数（PDF文件或单页图像）
    2. Stage 1: 并行处理所有页面转Markdown
    3. 检查错误率，超过阈值则丢弃文档
    4. Stage 2: 检测相邻页面间的元素合并需求
    5. Stage 3: 批量处理HTML表格合并
    6. 构建最终文档文本并返回结果
    """
    logger.info(f"Start process_pdf for {pdf_path}")
    
    # 第一步：确定文档页数
    # 如果是PDF文件，使用pypdf读取页数；如果是图像文件，默认为1页
    if pdf_path.lower().endswith(".pdf"):
        try:
            reader = PdfReader(pdf_path)
            num_pages = reader.get_num_pages()
        except:
            logger.exception(f"Could not count number of pages for {pdf_path}, aborting document")
            return None
    else:
        # 图像文件（如PNG、JPG等）默认为单页处理
        num_pages = 1
    
    logger.info(f"Got {num_pages} pages to do for {pdf_path} in worker {worker_id}")

    try:
        # ========== Stage 1: Page to Markdown 处理阶段 ==========
        # 并行处理所有页面，将每页转换为Markdown格式
        tasks = []
        results = []
        async with asyncio.TaskGroup() as tg:
            # 为每个页面创建异步任务
            for page_num in range(1, num_pages + 1):
                task = tg.create_task(process_task(args, worker_id, task_name='page_to_markdown', task_args=(pdf_path,page_num)))
                tasks.append(task)
        
        # 收集所有页面的处理结果
        results = [task.result() for task in tasks]

        # 处理Stage 1的结果，准备Stage 2的输入数据
        fallback_pages = []  # 处理失败的页面列表
        page_to_markdown_result = {}  # 页面号 -> Markdown文本段落列表的映射
        page_pairs = []  # 相邻页面对列表，用于Stage 2处理
        
        for i,result in enumerate(results):
            if result != None:
                page_number = i+1
                # 后处理Markdown文本并按段落分割
                page_to_markdown_result[i+1] = postprocess_markdown_text(args,result,pdf_path,page_number).split("\n\n")
                # 如果前一页也处理成功，则添加到相邻页面对列表中
                if page_number-1 in page_to_markdown_result.keys():
                    page_pairs.append((page_number-1,page_number))
            else:
                # 记录处理失败的页面
                fallback_pages.append(i)
        
        # 错误率检查：如果失败页面过多，则丢弃整个文档
        num_fallback_pages = len(fallback_pages)

        if num_fallback_pages / num_pages > args.max_page_error_rate:
            logger.error(
                f"Document {pdf_path} has {num_fallback_pages} fallback pages out of {num_pages} exceeding max_page_error_rate of {args.max_page_error_rate}, discarding document."
            )
            return None
        elif num_fallback_pages > 0:
            logger.warning(
                f"Document {pdf_path} processed with {num_fallback_pages} fallback pages out of {num_pages}."
            )

        # 如果配置为跳过跨页合并，则直接返回Stage 1的结果
        if args.skip_cross_page_merge:
            page_texts = {}
            document_text_list = []
            sorted_page_keys = sorted(list(page_to_markdown_result.keys()))
            for page_number in sorted_page_keys:
                # 将每页的段落重新组合为完整文本
                page_texts[str(page_number-1)] = "\n\n".join(page_to_markdown_result[page_number])
                document_text_list.append(page_texts[str(page_number-1)])
            document_text = "\n\n".join(document_text_list)
            return {
                "orig_path": pdf_path,
                "num_pages": num_pages,
                "document_text": document_text,
                "page_texts": page_texts,
                "fallback_pages": fallback_pages,
            }

        # ========== Stage 2: Element Merge Detect 处理阶段 ==========
        # 检测相邻页面间需要合并的元素（如跨页表格、列表等）
        tasks = []
        results = []
        async with asyncio.TaskGroup() as tg:
            # 为每对相邻页面创建元素合并检测任务
            for page_1,page_2 in page_pairs:
                # 打印阶段二的输入参数
                print(f"\n=== 阶段二输入 - 页面对 ({page_1}, {page_2}) ===")
                print(f"页面 {page_1} 的 Markdown 结果:")
                print(page_to_markdown_result[page_1])
                print(f"\n页面 {page_2} 的 Markdown 结果:")
                print(page_to_markdown_result[page_2])
                print("=" * 50)
                
                task = tg.create_task(process_task(args, worker_id, task_name='element_merge_detect', task_args=(page_to_markdown_result[page_1], page_to_markdown_result[page_2])))
                tasks.append(task)
        results = [task.result() for task in tasks]
        
        # 打印阶段二返回的结果
        print("\n=== 阶段二返回结果 ===")
        for i, (page_pair, result) in enumerate(zip(page_pairs, results)):
            page_1, page_2 = page_pair
            print(f"页面对 ({page_1}, {page_2}) 的检测结果:")
            if result is not None:
                print(f"  检测到 {len(result)} 个需要合并的元素对:")
                for j, (elem_idx_1, elem_idx_2) in enumerate(result):
                    print(f"    元素对 {j+1}: 页面{page_1}的元素{elem_idx_1} <-> 页面{page_2}的元素{elem_idx_2}")
            else:
                print("  未检测到需要合并的元素")
            print()
        print("=" * 50)
        
        # 处理Stage 2的结果，识别需要合并的元素对
        element_merge_detect_result = {}  # 存储元素合并检测结果
        table_pairs = []  # 存储需要进行HTML表格合并的页面对
        
        for page_pair,result in zip(page_pairs,results):
            if result != None:
                page_1,page_2 = page_pair
                element_merge_detect_result[(page_1,page_2)] = result
                
                # 检查合并的元素对中是否包含HTML表格
                for elem_idx_1,elem_idx_2 in result:
                    text_1 = page_to_markdown_result[page_1][elem_idx_1]
                    text_2 = page_to_markdown_result[page_2][elem_idx_2]
                    # 如果两个元素都是完整的HTML表格，则添加到表格合并列表
                    if text_1.startswith("<table>") and text_1.endswith("</table>") and text_2.startswith("<table>") and text_2.endswith("</table>"):
                        table_pairs.append((page_1,page_2,elem_idx_1,elem_idx_2))

        # ========== Stage 3: HTML Table Merge 处理阶段 ==========
        # 批量处理HTML表格合并，避免冲突和重复处理
        tmp_page_to_markdown_result = copy.deepcopy(page_to_markdown_result)
        table_pairs = sorted(table_pairs,key=lambda x: -x[0])  # 按页面号倒序排列，避免索引冲突
        html_table_merge_result = {}  # 存储HTML表格合并结果
        
        # 批处理逻辑：将不冲突的表格合并任务分组并行处理
        i = 0
        while i < len(table_pairs):
            async with asyncio.TaskGroup() as tg:
                tasks = []
                ids_1 = []  # 记录第一个表格的位置，避免重复使用
                ids_2 = []  # 记录第二个表格的位置，避免重复使用
                
                # 处理第一个表格对
                page_1,page_2,elem_idx_1,elem_idx_2 = table_pairs[i]
                task = tg.create_task(process_task(args, worker_id, task_name='html_table_merge', task_args=(tmp_page_to_markdown_result[page_1][elem_idx_1], tmp_page_to_markdown_result[page_2][elem_idx_2])))
                tasks.append(task)
                ids_1.append((page_1,elem_idx_1))
                ids_2.append((page_2,elem_idx_2))
                
                # 寻找可以并行处理的其他表格对（不与当前批次冲突）
                j = i + 1
                while j < len(table_pairs):
                    page_1,page_2,elem_idx_1,elem_idx_2 = table_pairs[j]
                    # 检查是否与已选择的表格位置冲突
                    if (page_2, elem_idx_2) not in ids_1:
                        task = tg.create_task(process_task(args, worker_id, task_name='html_table_merge', task_args=(tmp_page_to_markdown_result[page_1][elem_idx_1], tmp_page_to_markdown_result[page_2][elem_idx_2])))
                        tasks.append(task)
                        ids_1.append((page_1,elem_idx_1))
                        ids_2.append((page_2,elem_idx_2))
                        j = j + 1
                    else:
                        # 发现冲突，结束当前批次
                        break
                    
            # 收集当前批次的表格合并结果
            results = [task.result() for task in tasks]

            # 处理合并结果，更新临时结果集
            for k,result in enumerate(results):
                page_1,elem_idx_1 = ids_1[k]
                page_2,elem_idx_2 = ids_2[k]
                if result != None:
                    # 保存合并结果
                    html_table_merge_result[(page_1,page_2,elem_idx_1,elem_idx_2)] = result
                    # 用合并后的表格替换第一个表格的内容
                    tmp_page_to_markdown_result[page_1][elem_idx_1] = html_table_merge_result[(page_1,page_2,elem_idx_1,elem_idx_2)]
            # 移动到下一个批次
            i = j

        # ========== 构建最终结果 ==========
        # 准备各页面的原始文本（用于调试和分析）
        page_texts = {}
        for page_number in page_to_markdown_result.keys():
            page_texts[str(page_number-1)] = "\n\n".join(page_to_markdown_result[page_number])
        
        # 构建整合了三个阶段处理结果的最终文档文本
        document_text = build_document_text(page_to_markdown_result, element_merge_detect_result, html_table_merge_result)

        # 返回完整的处理结果
        return {
            "orig_path": pdf_path,
            "num_pages": num_pages,
            "document_text": document_text,
            "page_texts": page_texts,
            "fallback_pages": fallback_pages,
        }
    except Exception as e:
        # Check for ExceptionGroup with BrokenProcessPool
        if isinstance(e, ExceptionGroup):
            broken_pool, other = e.split(BrokenProcessPool)
            if broken_pool is not None:  # Found at least one BrokenProcessPool
                logger.critical("Encountered BrokenProcessPool, exiting process.")
                sys.exit(1)

        logger.exception(f"Exception in process_pdf for {pdf_path}: {e}")
        return None

async def process_json(args, worker_id: int, json_path: str):
    """
    JSON数据处理函数 - 处理单独的合并任务
    
    功能说明：
    - 处理预先准备的JSON格式的合并任务
    - 支持两种任务类型：
      1. merge_pages: 页面元素合并检测（Stage 2单独执行）
      2. merge_tables: HTML表格合并（Stage 3单独执行）
    - 用于测试、调试或批量处理特定的合并场景
    
    参数：
        args: 命令行参数对象，包含任务类型等配置
        worker_id: 工作进程ID，用于日志标识
        json_path: JSON文件路径，包含待处理的数据
    
    返回值：
        dict: 包含处理结果的字典
            - orig_path: 原始JSON文件路径
            - merge_pairs: 元素合并对（merge_pages任务）
            - merged_tables: 合并后的表格（merge_tables任务）
        None: 处理失败时返回
    
    JSON文件格式：
    - merge_pages任务: {"page_1": "页面1文本", "page_2": "页面2文本"}
    - merge_tables任务: {"table_1": "表格1HTML", "table_2": "表格2HTML"}
    """
    try:
        # 加载JSON数据
        json_data = json.load(open(json_path,'r'))
    except:
        logger.exception(f"Could not load {json_path}, aborting document")
        
    try:
        # 根据任务类型执行相应的处理逻辑
        if args.task == 'merge_pages':
            # 页面元素合并检测任务（Stage 2）
            page_1 = json_data['page_1'].split("\n\n")  # 将页面文本按段落分割
            page_2 = json_data['page_2'].split("\n\n")
            async with asyncio.TaskGroup() as tg:
                task = tg.create_task(process_task(args, worker_id, task_name='element_merge_detect', task_args=(page_1, page_2)))
            result = task.result()
            return {
                "orig_path": json_path,
                "merge_pairs": result  # 返回需要合并的元素对索引
            }
        elif args.task == 'merge_tables':
            # HTML表格合并任务（Stage 3）
            table_1 = json_data['table_1']  # 第一个表格的HTML内容
            table_2 = json_data['table_2']  # 第二个表格的HTML内容
            async with asyncio.TaskGroup() as tg:
                task = tg.create_task(process_task(args, worker_id, task_name='html_table_merge', task_args=(table_1, table_2)))
            result = task.result()
            return {
                "orig_path": json_path,
                "merged_tables": result  # 返回合并后的表格HTML
            }
        else:
            raise ValueError(f"Unknown task {args.task}")
    
    except Exception as e:
        # Check for ExceptionGroup with BrokenProcessPool
        if isinstance(e, ExceptionGroup):
            broken_pool, other = e.split(BrokenProcessPool)
            if broken_pool is not None:  # Found at least one BrokenProcessPool
                logger.critical("Encountered BrokenProcessPool, exiting process.")
                sys.exit(1)

        logger.exception(f"Exception in process_json for {json_path}: {e}")
        return None

async def worker(args, work_queue: WorkQueue, semaphore, worker_id):
    """
    工作进程函数 - OCRFlux的核心工作单元
    
    功能说明：
    - 从工作队列中获取任务并处理
    - 支持并发控制（通过信号量限制同时运行的任务数）
    - 处理三种任务类型：pdf2markdown、merge_pages、merge_tables
    - 将处理结果写入JSONL文件
    - 具备异常处理和资源清理机制
    
    参数：
        args: 命令行参数对象，包含任务配置
        work_queue: 工作队列对象，提供任务分发
        semaphore: 信号量，控制并发数量
        worker_id: 工作进程ID，用于日志标识和追踪
    
    工作流程：
    1. 等待信号量许可（控制并发）
    2. 从队列获取工作项
    3. 根据任务类型调用相应处理函数
    4. 收集处理结果并写入输出文件
    5. 标记任务完成并释放信号量
    
    输出格式：
    - 结果文件：workspace/results/output_{work_hash}.jsonl
    - 每行一个JSON对象，包含完整的处理结果
    """
    while True:
        # 第一步：等待信号量许可，控制并发数量
        await semaphore.acquire()

        # 第二步：从工作队列获取任务
        work_item = await work_queue.get_work()

        # 检查是否还有任务需要处理
        if work_item is None:
            logger.info(f"Worker {worker_id} exiting due to empty queue")
            semaphore.release()
            break

        logger.info(f"Worker {worker_id} processing work item {work_item.hash}")
        # 清理工作进程的状态追踪
        await tracker.clear_work(worker_id)

        try:
            # 第三步：根据任务类型创建并执行异步任务组
            async with asyncio.TaskGroup() as tg:
                if args.task == 'pdf2markdown':
                    # PDF转Markdown任务：处理PDF文件，执行完整的三阶段流程
                    tasks = [tg.create_task(process_pdf(args, worker_id, pdf_path)) for pdf_path in work_item.work_paths]
                elif args.task == 'merge_pages' or args.task == 'merge_tables':
                    # 独立合并任务：处理JSON文件，执行特定阶段的合并操作
                    tasks = [tg.create_task(process_json(args, worker_id, json_path)) for json_path in work_item.work_paths]
                else:
                    raise ValueError(f"Unknown task {args.task}")

                logger.info(f"Created all tasks for {work_item.hash}")

            logger.info(f"Finished TaskGroup for worker on {work_item.hash}")

            # 第四步：收集任务执行结果
            results = []
            for task in tasks:
                try:
                    result = task.result()
                except:
                    # 忽略失败的任务，继续处理其他结果
                    pass

                if result is not None:
                    results.append(result)

            logger.info(f"Got {len(results)} docs for {work_item.hash}")

            # 第五步：将结果写入JSONL文件
            # 输出格式：每行一个JSON对象，便于后续处理和分析
            output_final_path = os.path.join(args.workspace, "results", f"output_{work_item.hash}.jsonl")
            with open(output_final_path, "w") as f:
                for result in results:
                    f.write(json.dumps(result))
                    f.write("\n")

            # 第六步：标记工作项完成
            await work_queue.mark_done(work_item)
        except Exception as e:
            # 异常处理：记录错误信息，但不中断整个工作流程
            logger.exception(f"Exception occurred while processing work_hash {work_item.hash}: {e}")
        finally:
            # 第七步：无论成功或失败，都要释放信号量
            # 确保其他等待的工作进程能够继续执行
            semaphore.release()

async def vllm_server_task(args, semaphore):
    """
    vLLM服务器任务管理函数 - 负责启动和监控vLLM推理服务器
    
    功能说明：
    - 启动vLLM服务器子进程，配置模型和运行参数
    - 实时监控服务器输出，解析运行状态和队列信息
    - 智能管理信号量释放，优化并发控制
    - 处理服务器错误和异常情况
    - 支持服务器就绪检测和自动重启机制
    
    参数：
        args: 命令行参数对象，包含模型配置和服务器设置
        semaphore: 信号量对象，用于控制工作进程的并发数量
    
    服务器配置：
    - 模型路径：args.model
    - 端口：args.port
    - 最大上下文长度：args.model_max_context
    - GPU内存利用率：args.gpu_memory_utilization
    - 张量并行大小：args.tensor_parallel_size
    - 数据类型：args.dtype
    
    监控机制：
    - 解析服务器日志，提取运行中和排队中的请求数量
    - 当服务器空闲时自动释放信号量，允许新的工作进程启动
    - 检测采样错误和索引错误，自动重启服务器
    """
    # 获取模型路径
    model_name_or_path = args.model

    # 构建vLLM服务器启动命令
    # 配置模型服务的各项参数，包括端口、上下文长度、GPU设置等
    cmd = [
        "vllm",
        "serve",
         model_name_or_path,
        "--port",
        str(args.port),
        "--max-model-len",
        str(args.model_max_context),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--tensor_parallel_size",
        str(args.tensor_parallel_size),
        "--dtype",
        str(args.dtype)
    ]

    # 创建vLLM服务器子进程
    # 捕获标准输出和错误输出，用于监控服务器状态
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # 注册进程清理函数，确保程序退出时终止子进程
    def _kill_proc():
        proc.terminate()

    atexit.register(_kill_proc)

    # 服务器状态监控的共享变量
    last_running_req, last_queue_req = 0, 0  # 运行中和排队中的请求数量
    server_printed_ready_message = False      # 服务器是否已就绪
    last_semaphore_release = time.time()      # 上次释放信号量的时间

    async def process_line(line):
        """
        处理vLLM服务器输出的单行日志
        
        功能：
        - 解析服务器状态信息（运行中/排队中的请求数量）
        - 检测服务器就绪状态
        - 识别和处理错误情况
        - 更新监控变量
        """
        nonlocal last_running_req, last_queue_req, last_semaphore_release, server_printed_ready_message
        
        # 记录vLLM服务器的原始日志
        vllm_logger.info(line)

        # 服务器初始化期间，将所有日志同时输出到主日志器
        # 便于用户查看启动过程中的警告和错误信息
        if not server_printed_ready_message:
            logger.info(line)

        # 检测采样错误：模型可能损坏，需要退出
        if "Detected errors during sampling" in line:
            logger.error("Cannot continue, sampling errors detected, model is probably corrupt")
            sys.exit(1)

        # 检测索引错误：vLLM的已知问题，会导致服务器锁死，需要重启
        # TODO: 需要在vLLM本身中追踪这个问题
        if "IndexError: list index out of range" in line:
            logger.error("IndexError in model, restarting server")
            proc.terminate()

        # 检测服务器就绪消息
        if not server_printed_ready_message and "The server is fired up and ready to roll!" in line:
            server_printed_ready_message = True
            last_semaphore_release = time.time()

        # 解析运行中的请求数量
        match = re.search(r"Running: (\d+)", line)
        if match:
            last_running_req = int(match.group(1))

        # 解析排队中的请求数量
        match = re.search(r"(?:Waiting|Pending):\s*(\d+)", line)
        if match:
            last_queue_req = int(match.group(1))
            logger.info(f"vllm running req: {last_running_req} queue req: {last_queue_req}")
            
    async def read_stream(stream):
        """
        异步读取服务器输出流（stdout或stderr）
        
        功能：
        - 持续读取服务器的输出流
        - 解码并处理每一行日志
        - 处理读取过程中的异常
        """
        while True:
            line = await stream.readline()
            if not line:
                break
            try:
                # 解码UTF-8并去除行尾空白字符
                line = line.decode("utf-8").rstrip()
                await process_line(line)
            except Exception as ex:
                # 忽略日志读取异常，避免影响主流程
                logger.warning(f"Got {ex} when reading log line from inference server, skipping")

    async def timeout_task():
        """
        超时任务：智能管理信号量释放
        
        功能：
        - 监控服务器空闲状态
        - 当服务器空闲超过30秒时自动释放信号量
        - 允许等待中的工作进程继续执行
        
        释放条件：
        1. 服务器已就绪
        2. 队列中没有等待的请求
        3. 距离上次释放超过30秒
        4. 信号量当前被锁定
        """
        nonlocal last_running_req, last_queue_req, last_semaphore_release
        try:
            while True:
                await asyncio.sleep(1)
                # 检查是否满足信号量释放条件
                if server_printed_ready_message and last_queue_req == 0 and time.time() - last_semaphore_release > 30 and semaphore.locked():
                    semaphore.release()
                    last_semaphore_release = time.time()
                    logger.info("Semaphore released, allowing a worker to proceed.")
        except asyncio.CancelledError:
            # 任务被取消时的清理处理
            pass

    # 启动并发任务：监控服务器输出和超时管理
    stdout_task = asyncio.create_task(read_stream(proc.stdout))  # 监控标准输出
    stderr_task = asyncio.create_task(read_stream(proc.stderr))  # 监控错误输出
    timeout_task = asyncio.create_task(timeout_task())           # 超时和信号量管理

    try:
        # 等待vLLM服务器进程结束
        await proc.wait()
    except asyncio.CancelledError:
        # 处理取消请求：优雅地终止服务器
        logger.info("Got cancellation request for VLLM server")
        proc.terminate()
        raise

    # 清理并发任务
    timeout_task.cancel()
    # 等待所有监控任务完成，忽略异常
    await asyncio.gather(stdout_task, stderr_task, timeout_task, return_exceptions=True)

async def vllm_server_host(args, semaphore):
    """
    vLLM服务器宿主管理函数 - 负责服务器的重启和故障恢复
    
    功能说明：
    - 管理vLLM服务器的生命周期
    - 实现自动重启机制，处理服务器异常退出
    - 限制重启次数，避免无限循环
    - 提供安装指导信息
    
    参数：
        args: 命令行参数对象
        semaphore: 信号量对象
    
    重启策略：
    - 最大重试次数：5次
    - 每次服务器异常退出后自动重启
    - 超过最大重试次数后退出程序
    """
    MAX_RETRIES = 5
    retry = 0

    while retry < MAX_RETRIES:
        # 启动vLLM服务器任务
        await vllm_server_task(args, semaphore)
        logger.warning("VLLM server task ended")
        retry += 1

    # 超过最大重试次数，退出程序
    if retry >= MAX_RETRIES:
        logger.error(f"Ended up starting the vllm server more than {retry} times, cancelling pipeline")
        logger.error("")
        logger.error("Please make sure vllm is installed according to the latest instructions here: https://docs.vllm.ai/start/install.html")
        sys.exit(1)

async def vllm_server_ready(args):
    """
    vLLM服务器就绪检测函数 - 等待服务器完全启动
    
    功能说明：
    - 通过HTTP请求检测服务器是否就绪
    - 实现轮询机制，持续检测直到服务器响应
    - 提供详细的状态反馈
    
    参数：
        args: 命令行参数对象，包含端口配置
    
    检测策略：
    - 最大尝试次数：300次
    - 检测间隔：1秒
    - 检测端点：/v1/models
    - 成功条件：HTTP状态码200
    """
    max_attempts = 300
    delay_sec = 1
    url = f"http://192.168.8.36:8004/v1/models"

    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient() as session:
                response = await session.get(url)

                if response.status_code == 200:
                    logger.info("vllm server is ready.")
                    return
                else:
                    logger.info(f"Attempt {attempt}: Unexpected status code {response.status_code}")
        except Exception:
            logger.warning(f"Attempt {attempt}: Please wait for vllm server to become ready...")

        await asyncio.sleep(delay_sec)

    raise Exception("vllm server did not become ready after waiting.")

async def download_model(model_name_or_path: str):
    """
    模型下载函数 - 处理本地模型和远程模型的加载
    
    功能说明：
    - 检测模型路径类型（本地路径或HuggingFace模型ID）
    - 对于本地路径：直接使用，无需下载
    - 对于远程模型：从HuggingFace下载到本地缓存
    
    参数：
        model_name_or_path: 模型路径或HuggingFace模型ID
    
    支持格式：
    - 本地绝对路径：如 /path/to/local/model
    - HuggingFace模型ID：如 ChatDOC/OCRFlux-3B
    """
    if os.path.isabs(model_name_or_path) and os.path.isdir(model_name_or_path):
        # 使用本地模型路径
        logger.info(f"Using local model path at '{model_name_or_path}'")
    else:
        # 从HuggingFace下载模型
        logger.info(f"Downloading model with hugging face '{model_name_or_path}'")
        snapshot_download(repo_id=model_name_or_path)

async def metrics_reporter(work_queue):
    """
    指标报告函数 - 定期输出系统运行状态
    
    功能说明：
    - 每10秒输出一次系统状态信息
    - 显示工作队列剩余任务数量
    - 显示性能指标统计
    - 显示工作进程状态表
    
    参数：
        work_queue: 工作队列对象
    
    输出内容：
    - 队列剩余任务数
    - MetricsKeeper的性能统计
    - WorkerTracker的进程状态表
    """
    while True:
        # 输出队列状态和性能指标
        # 前导换行符保持表格格式在日志中的可读性
        logger.info(f"Queue remaining: {work_queue.size}")
        logger.info("\n" + str(metrics))
        logger.info("\n" + str(await tracker.get_status_table()))
        await asyncio.sleep(10)

async def main():
    """
    OCRFlux主函数 - 系统启动入口和流程控制
    
    功能说明：
    - 解析命令行参数和配置
    - 初始化工作队列和任务分组
    - 启动vLLM推理服务器
    - 创建并管理工作进程
    - 协调整个OCR处理流程
    
    支持的任务类型：
    1. pdf2markdown: 完整的PDF到Markdown转换（三阶段流程）
    2. merge_pages: 仅执行Stage 2元素合并检测
    3. merge_tables: 仅执行Stage 3 HTML表格合并
    
    启动流程：
    1. 参数解析和验证
    2. 工作空间初始化
    3. 任务队列构建
    4. 模型下载和服务器启动
    5. 工作进程创建和任务执行
    6. 监控和清理
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Manager for running millions of PDFs through a batch inference pipeline")
    
    # 必需参数：工作空间路径
    parser.add_argument(
        "workspace",
        help="The filesystem path where work will be stored, can be a local folder",
    )

    # 任务类型选择
    parser.add_argument("--task", type=str, choices=['pdf2markdown','merge_pages','merge_tables'], default='pdf2markdown', help="task names, could be 'pdf2markdown', 'merge_pages' or 'merge_tables'")

    # 输入数据路径
    parser.add_argument(
        "--data",
        nargs="*",
        help="List of paths to files to process",
        default=None,
    )

    # 处理参数配置
    parser.add_argument("--pages_per_group", type=int, default=500, help="Aiming for this many pdf pages per work item group")
    parser.add_argument("--max_page_retries", type=int, default=8, help="Max number of times we will retry rendering a page")
    parser.add_argument("--max_page_error_rate", type=float, default=0.004, help="Rate of allowable failed pages in a document, 1/250 by default")
    
    # GPU和并发配置
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="Fraction of GPU memory to use, default is 0.8")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of tensor parallel replicas")
    parser.add_argument("--dtype", type=str, choices=['auto','half','float16', 'float', 'bfloat16', 'float32'], default="auto", help="Data type for model weights and activations.")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers to run at a time")

    # 模型相关参数
    parser.add_argument(
        "--model",
        help="The path to the model",
        default="ChatDOC/OCRFlux-3B",
    )
    parser.add_argument("--model_max_context", type=int, default=16384, help="Maximum context length that the model was fine tuned under")
    parser.add_argument("--model_chat_template", type=str, default="qwen2-vl", help="Chat template to pass to vllm server")
    parser.add_argument("--target_longest_image_dim", type=int, help="Dimension on longest side to use for rendering the pdf pages", default=1024)

    # 处理流程控制
    parser.add_argument("--skip_cross_page_merge", action="store_true", help="Whether to skip cross-page merging")

    # 服务器配置
    parser.add_argument("--port", type=int, default=40078, help="Port to use for the VLLM server")
    args = parser.parse_args()

    # 工作空间初始化：清理旧数据，确保干净的开始
    if os.path.exists(args.workspace):
        shutil.rmtree(args.workspace)

    # 依赖检查：验证Poppler库版本（PDF处理必需）
    check_poppler_version()

    # 创建本地工作队列管理器
    work_queue = LocalWorkQueue(args.workspace)

    # 任务类型1：PDF到Markdown的完整转换流程
    if args.task == 'pdf2markdown':
        pdf_work_paths = set()

        # 遍历输入数据，验证文件格式和存在性
        for pdf_path in args.data:
            if os.path.exists(pdf_path):
                # 检查PDF文件：验证文件头部魔数
                if pdf_path.lower().endswith(".pdf") and open(pdf_path, "rb").read(4) == b"%PDF":
                    logger.info(f"Loading file at {pdf_path} as PDF document")
                    pdf_work_paths.add(pdf_path)
                # 检查图像文件：支持常见图像格式
                elif is_image(pdf_path):
                    logger.info(f"Loading file at {pdf_path} as image document")
                    pdf_work_paths.add(pdf_path)
                else:
                    raise ValueError(f"Unsupported file extension for {pdf_path}")
            else:
                raise ValueError(f"{pdf_path} does not exist")

        logger.info(f"Found {len(pdf_work_paths):,} total pdf paths to add")

        # 智能分组策略：估算平均页数以优化任务分组
        sample_size = min(100, len(pdf_work_paths))
        sampled_pdfs = random.sample(list(pdf_work_paths), sample_size)
        page_counts = []

        # 采样PDF文件，统计页数分布
        for pdf_path in tqdm(sampled_pdfs, desc="Sampling PDFs to calculate optimal length"):
            try:
                if pdf_path.lower().endswith(".pdf"):
                    reader = PdfReader(pdf_path)
                    page_counts.append(len(reader.pages))
                else:
                    # 图像文件视为单页
                    page_counts.append(1)
            except Exception as e:
                logger.warning(f"Failed to read {pdf_path}: {e}")

        # 计算平均页数，用于优化任务分组
        if page_counts:
            avg_pages_per_pdf = sum(page_counts) / len(page_counts)
        else:
            logger.warning("Could not read any PDFs to estimate average page count.")
            avg_pages_per_pdf = 10  # 采样失败时的默认值

        # 根据目标页数和平均页数计算每组文件数
        items_per_group = max(1, int(args.pages_per_group / avg_pages_per_pdf))
        logger.info(f"Calculated items_per_group: {items_per_group} based on average pages per PDF: {avg_pages_per_pdf:.2f}")

        # 填充工作队列
        await work_queue.populate_queue(pdf_work_paths, items_per_group)
    # 任务类型2&3：独立的合并任务（Stage 2或Stage 3）
    elif args.task == 'merge_pages' or args.task == 'merge_tables':
        json_work_paths = set()
        
        # 处理输入数据：支持JSON文件和路径列表文件
        for json_path in args.data:
            if os.path.exists(json_path):
                # 直接的JSON文件
                if json_path.lower().endswith(".json"):
                    json_work_paths.add(json_path)
                # 包含路径列表的文本文件
                elif json_path.lower().endswith(".txt"):
                    logger.info(f"Loading file at {json_path} as list of paths")
                    with open(json_path, "r") as f:
                        # 读取每行路径，过滤空行
                        json_work_paths |= set(filter(None, (line.strip() for line in f)))
                else:
                    raise ValueError(f"Unsupported file extension for {json_path}")
            else:
                raise ValueError(f"{json_path} does not exist")

        # 填充工作队列（合并任务使用固定分组大小）
        await work_queue.populate_queue(json_work_paths, args.pages_per_group)


    # 推理环境检查：确保GPU和vLLM环境可用
    # check_vllm_version()  # 用户指出不需要检查
    # check_torch_gpu_available()  # 函数未定义，暂时注释

    logger.info(f"Starting pipeline with PID {os.getpid()}")

    # 模型准备：下载或验证模型可用性
    # await download_model(args.model)  # 使用外部API服务，跳过模型下载

    # 工作队列初始化：获取待处理任务数量
    qsize = await work_queue.initialize_queue()

    if qsize == 0:
        logger.info("No work to do, exiting")
        return
        
    # 并发控制：创建信号量管理GPU资源访问
    # 只允许一个工作进程向服务器发送请求，直到服务器队列为空
    # 这样既能充分利用GPU，又能尽快输出处理结果
    # 当一个工作进程不再饱和GPU时，下一个进程可以开始发送请求
    semaphore = asyncio.Semaphore(1)

    # 启动vLLM推理服务器（异步任务）
    vllm_server = asyncio.create_task(vllm_server_host(args, semaphore))

    # 等待服务器就绪
    await vllm_server_ready(args)

    # 启动指标监控任务
    metrics_task = asyncio.create_task(metrics_reporter(work_queue))

    # 创建工作进程池：并发处理队列中的任务
    worker_tasks = []
    for i in range(args.workers):
        task = asyncio.create_task(worker(args, work_queue, semaphore, worker_id=i))
        worker_tasks.append(task)

    # 等待所有工作进程完成
    await asyncio.gather(*worker_tasks)

    # 清理：取消服务器和监控任务
    vllm_server.cancel()
    metrics_task.cancel()
    logger.info("Work done")


# 程序入口：启动异步主函数
if __name__ == "__main__":
    asyncio.run(main())
