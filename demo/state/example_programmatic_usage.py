#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编程方式调用 OCRFlux 的示例脚本

这个脚本展示了如何通过配置字典的方式来调用 OCRFlux。
适用于需要在其他 Python 程序中集成 OCRFlux 功能的场景。
"""

import asyncio
import os
from control_page_proccess import run_with_config

async def example_pdf_to_markdown():
    """
    示例1：完整的 PDF 到 Markdown 转换流程
    """
    config = {
        # 'workspace': './example_workspace1',
        'task': 'pdf2markdown',
        'data': ['../file/1223309368.pdf'],
        'workers': 4,
        'model': 'OCRFlux-3B',
        'port': 8004,
        'server_host': '192.168.8.36',  # 服务器IP地址
        'server_port': 8004,             # 服务器端口
        'gpu_memory_utilization': 0.8,
        'tensor_parallel_size': 1,
        'dtype': 'auto',
        'pages_per_group': 500,
        'max_page_retries': 8,
        'max_page_error_rate': 0.004,
        'model_max_context': 16384,
        'model_chat_template': 'qwen2-vl',
        'target_longest_image_dim': 1024,
        'skip_cross_page_merge': False
    }
    
    print("开始执行 PDF 到 Markdown 转换...")
    await run_with_config(config)
    print("转换完成！")

async def example_merge_pages_only():
    """
    示例2：仅执行页面元素合并检测（Stage 2）
    注意：merge_pages任务需要JSON格式的输入文件，而不是PDF文件
    """
    config = {
        # 'workspace': './example_workspace_stage2',
        'task': 'merge_pages',
        'data': ['./test_workspace/results/output_dc16c76ec18eb151fbc180d2da3e37d0f9497b40.json'],  # 使用已处理的JSON文件
        'workers': 2,
        'model': 'OCRFlux-3B',
        'port': 40079,
        'server_host': '192.168.8.36',  # 服务器IP地址
        'server_port': 8004,             # 服务器端口
        'gpu_memory_utilization': 0.6,
        'tensor_parallel_size': 1,
        'dtype': 'auto'
    }
    
    print("开始执行页面元素合并检测...")
    await run_with_config(config)
    print("合并检测完成！")

async def example_merge_tables_only():
    """
    示例3：仅执行 HTML 表格合并（Stage 3）
    适用于已有页面处理结果，只需要合并表格的场景
    注意：merge_tables任务需要JSON格式的输入文件，而不是PDF文件
    """
    config = {
        # 'workspace': './example_workspace_stage3',
        'task': 'merge_tables',
        'data': ['./test_workspace/results/output_dc16c76ec18eb151fbc180d2da3e37d0f9497b40.json'],  # 使用已处理的JSON文件
        'workers': 2,
        'model': 'OCRFlux-3B',
        'port': 40080,  # 使用不同端口避免冲突
        'server_host': '192.168.8.36',  # 服务器IP地址
        'server_port': 8004,             # 服务器端口
        'gpu_memory_utilization': 0.6
    }
    
    print("开始执行 HTML 表格合并...")
    await run_with_config(config)
    print("表格合并完成！")

async def example_custom_config():
    """
    示例4：自定义配置的高级用法
    """
    # 可以根据运行环境动态调整配置
    import os
    
    # 根据环境变量或默认值调整 GPU 内存利用率
    gpu_util = float(os.environ.get('GPU_UTIL', '0.8'))
    
    # 根据 CPU 核心数动态调整工作进程数
    cpu_count = os.cpu_count() or 4
    workers = min(cpu_count, 8)
    
    config = {
        # 'workspace': './dynamic_workspace',
        'task': 'pdf2markdown',
        'data': ['../file/1223309368.pdf'],
        'workers': workers,
        'model': 'OCRFlux-3B',
        'port': 40081,
        'server_host': '192.168.8.36',  # 服务器IP地址
        'server_port': 8004,             # 服务器端口
        'gpu_memory_utilization': gpu_util,
        'tensor_parallel_size': 1,
        'dtype': 'bfloat16',  # 使用更高效的数据类型
        'pages_per_group': 1000,  # 增大批次大小提高效率
        'max_page_retries': 5,
        'max_page_error_rate': 0.01,  # 允许更高的错误率
        'target_longest_image_dim': 1536,  # 更高的图像分辨率
        'skip_cross_page_merge': True  # 跳过跨页合并以提高速度
    }
    
    print(f"使用动态配置：{workers} 个工作进程，GPU 利用率 {gpu_util}")
    await run_with_config(config)
    print("动态配置处理完成！")

def main_sync():
    """
    同步入口函数，方便从其他脚本调用
    """
    print("OCRFlux 编程方式调用示例")
    print("=" * 50)
    
    # 选择要运行的示例
    examples = {
        '1': ('完整 PDF 到 Markdown 转换', example_pdf_to_markdown),
        '2': ('仅页面元素合并检测', example_merge_pages_only),
        '3': ('仅 HTML 表格合并', example_merge_tables_only),
        '4': ('动态配置示例', example_custom_config)
    }
    
    print("可用示例：")
    for key, (desc, _) in examples.items():
        print(f"  {key}. {desc}")
    
    choice = input("\n请选择要运行的示例 (1-4): ").strip()
    
    if choice in examples:
        desc, func = examples[choice]
        print(f"\n运行示例: {desc}")
        asyncio.run(func())
    else:
        print("无效选择，运行默认示例...")
        asyncio.run(example_pdf_to_markdown())

if __name__ == "__main__":
    main_sync()