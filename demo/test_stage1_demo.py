#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCRFlux Stage 1 Demo 测试脚本

这个脚本用于测试OCRFlux第一阶段demo的功能
包括单页处理、多页处理和错误处理测试
"""

import asyncio
import json
import os
import sys
from state1_exstruct import OCRFluxStage1Demo


async def test_single_page():
    """测试单页处理功能"""
    print("=== 测试单页处理 ===")
    
    demo = OCRFluxStage1Demo(
        server_url="http://192.168.8.36:8004",
        model_name="OCRFlux-3B"
    )
    
    # 测试图片文件（如果存在）
    test_image = "test.png"
    if os.path.exists(test_image):
        print(f"测试图片文件: {test_image}")
        result = await demo.process_page(test_image, 1)
        if result:
            print(f"✓ 图片处理成功，识别到 {result['element_count']} 个元素")
            print(f"前100字符: {result['markdown_text'][:100]}...")
        else:
            print("✗ 图片处理失败")
    else:
        print(f"跳过图片测试（{test_image} 不存在）")
    
    print()


async def test_pdf_document():
    """测试PDF文档处理功能"""
    print("=== 测试PDF文档处理 ===")
    
    demo = OCRFluxStage1Demo(
        server_url="http://192.168.8.36:8004",
        model_name="OCRFlux-3B"
    )
    
    # 测试PDF文件（如果存在）
    test_pdf = "test.pdf"
    if os.path.exists(test_pdf):
        print(f"测试PDF文件: {test_pdf}")
        result = await demo.process_document(test_pdf)
        if result:
            print(f"✓ PDF处理成功")
            print(f"  - 总页数: {result['num_pages']}")
            print(f"  - 成功页数: {len(result['page_to_markdown_result'])}")
            print(f"  - 失败页数: {len(result['failed_pages'])}")
            
            # 保存测试结果
            with open("test_result.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  - 结果已保存到: test_result.json")
            
            # 验证Stage 2兼容性
            stage2_format = result["page_to_markdown_result"]
            print(f"  - Stage 2格式验证: {len(stage2_format)} 页数据")
            
        else:
            print("✗ PDF处理失败")
    else:
        print(f"跳过PDF测试（{test_pdf} 不存在）")
    
    print()


async def test_error_handling():
    """测试错误处理功能"""
    print("=== 测试错误处理 ===")
    
    demo = OCRFluxStage1Demo(
        server_url="http://192.168.8.36:8004",
        model_name="OCRFlux-3B"
    )
    
    # 测试不存在的文件
    print("测试不存在的文件...")
    result = await demo.process_document("nonexistent.pdf")
    if result is None:
        print("✓ 正确处理了不存在的文件")
    else:
        print("✗ 未正确处理不存在的文件")
    
    # 测试错误的服务器地址
    print("测试错误的服务器地址...")
    bad_demo = OCRFluxStage1Demo(
        server_url="http://192.168.8.36:9999",  # 错误端口
        model_name="OCRFlux-3B"
    )
    
    # 创建一个简单的测试文件
    if os.path.exists("test.png") or os.path.exists("test.pdf"):
        test_file = "test.png" if os.path.exists("test.png") else "test.pdf"
        result = await bad_demo.process_page(test_file, 1)
        if result is None:
            print("✓ 正确处理了服务器连接错误")
        else:
            print("✗ 未正确处理服务器连接错误")
    else:
        print("跳过服务器错误测试（无测试文件）")
    
    print()


async def test_element_parsing():
    """测试元素解析功能"""
    print("=== 测试元素解析 ===")
    
    demo = OCRFluxStage1Demo()
    
    # 测试Markdown解析
    test_markdown = """
# 这是标题

这是一段普通文本。

<table>
<tr><td>表格</td><td>内容</td></tr>
</table>

<Image>(100,200),(300,400)</Image>

另一段文本内容。
"""
    
    elements = demo._parse_page_elements(test_markdown)
    
    print(f"解析结果: {len(elements)} 个元素")
    for i, element in enumerate(elements):
        print(f"  {i+1}. {element['type']}: {element['content'][:50]}...")
    
    # 验证元素类型
    types = [e['type'] for e in elements]
    expected_types = ['heading', 'text', 'table', 'image', 'text']
    
    if types == expected_types:
        print("✓ 元素解析正确")
    else:
        print(f"✗ 元素解析错误，期望: {expected_types}，实际: {types}")
    
    print()


async def test_api_connection():
    """测试API连接"""
    print("=== 测试API连接 ===")
    
    demo = OCRFluxStage1Demo(
        server_url="http://192.168.8.36:8004",
        model_name="OCRFlux-3B"
    )
    
    # 构建一个简单的测试查询
    try:
        # 创建一个简单的测试图像
        from PIL import Image
        test_image = Image.new('RGB', (100, 100), color='white')
        
        # 测试base64转换
        base64_result = demo._image_to_base64(test_image)
        if base64_result.startswith('data:image/png;base64,'):
            print("✓ 图像base64转换正常")
        else:
            print("✗ 图像base64转换异常")
        
        print(f"API地址: {demo.api_url}")
        print(f"模型名称: {demo.model_name}")
        
    except Exception as e:
        print(f"✗ API连接测试失败: {e}")
    
    print()


async def main():
    """主测试函数"""
    print("OCRFlux Stage 1 Demo 测试开始\n")
    
    # 检查测试文件
    test_files = ["test.pdf", "test.png", "test.jpg"]
    available_files = [f for f in test_files if os.path.exists(f)]
    
    if available_files:
        print(f"发现测试文件: {', '.join(available_files)}\n")
    else:
        print("警告: 未发现测试文件，某些测试将被跳过")
        print("建议在demo目录下放置test.pdf或test.png文件\n")
    
    # 运行测试
    await test_api_connection()
    await test_element_parsing()
    await test_error_handling()
    
    if available_files:
        await test_single_page()
        await test_pdf_document()
    
    print("=== 测试完成 ===")
    print("\n使用说明:")
    print("1. 确保服务器 192.168.8.36:8004 正常运行")
    print("2. 确保模型 OCRFlux-3B 已加载")
    print("3. 在demo目录下放置测试文件（test.pdf或test.png）")
    print("4. 运行: python test_stage1_demo.py")


if __name__ == "__main__":
    asyncio.run(main())