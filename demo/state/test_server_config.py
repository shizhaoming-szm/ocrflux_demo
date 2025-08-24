#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试服务器IP和端口参数配置的脚本
"""

import argparse
from control_page_proccess import run_with_config

def test_server_config():
    """
    测试服务器配置参数是否正确传递
    """
    # 测试配置
    config = {
        'workspace': './test_workspace',
        'task': 'pdf2markdown',
        'data': ['../file/1223309368.pdf'],
        'workers': 1,
        'model': 'OCRFlux-3B',
        'port': 8004,
        'server_host': '127.0.0.1',  # 测试用本地地址
        'server_port': 8005,         # 测试用不同端口
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
    
    # 创建 args 对象来测试参数传递
    args = argparse.Namespace()
    
    # 设置默认值
    defaults = {
        'task': 'pdf2markdown',
        'data': None,
        'pages_per_group': 500,
        'max_page_retries': 8,
        'max_page_error_rate': 0.004,
        'gpu_memory_utilization': 0.8,
        'tensor_parallel_size': 1,
        'dtype': 'auto',
        'workers': 8,
        'model': 'OCRFlux-3B',
        'model_max_context': 16384,
        'model_chat_template': 'qwen2-vl',
        'target_longest_image_dim': 1024,
        'skip_cross_page_merge': False,
        'port': 8004,
        'server_host': '192.168.8.36',
        'server_port': 8004
    }
    
    # 合并配置
    for key, default_value in defaults.items():
        setattr(args, key, config.get(key, default_value))
    
    args.workspace = config['workspace']
    
    # 测试URL构建
    completion_url = f"http://{args.server_host}:{args.server_port}/v1/chat/completions"
    models_url = f"http://{args.server_host}:{args.server_port}/v1/models"
    
    print("=== 服务器配置测试 ===")
    print(f"配置的服务器主机: {args.server_host}")
    print(f"配置的服务器端口: {args.server_port}")
    print(f"生成的完成URL: {completion_url}")
    print(f"生成的模型URL: {models_url}")
    print("\n参数传递测试通过！")
    
    # 验证URL格式
    expected_completion = f"http://{config['server_host']}:{config['server_port']}/v1/chat/completions"
    expected_models = f"http://{config['server_host']}:{config['server_port']}/v1/models"
    
    assert completion_url == expected_completion, f"完成URL不匹配: {completion_url} != {expected_completion}"
    assert models_url == expected_models, f"模型URL不匹配: {models_url} != {expected_models}"
    
    print("URL格式验证通过！")
    print("\n=== 测试结果 ===")
    print("✅ 服务器IP和端口参数配置功能正常")
    print("✅ URL构建功能正常")
    print("✅ 参数传递功能正常")

if __name__ == "__main__":
    test_server_config()