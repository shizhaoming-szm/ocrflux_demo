#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片处理工具模块
提供图片分辨率缩小和base64编码功能
"""

import os
import base64
import tempfile
from PIL import Image
from io import BytesIO
from typing import Union, Optional


def compress_image_to_base64(
    image_input: Union[str, Image.Image], 
    max_size_mb: float = 1.0,
    max_width: int = 1920,
    max_height: int = 1920,
    quality: int = 85
) -> str:
    """
    将图片压缩并转换为base64编码
    
    Args:
        image_input: 图片文件路径或PIL Image对象
        max_size_mb: 最大文件大小（MB），默认1MB
        max_width: 最大宽度，默认1920像素
        max_height: 最大高度，默认1920像素
        quality: JPEG质量（1-100），默认85
        
    Returns:
        str: base64编码的图片字符串
        
    Raises:
        Exception: 当图片处理失败时抛出异常
    """
    try:
        # 加载图片
        if isinstance(image_input, str):
            # 从文件路径加载
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"图片文件不存在: {image_input}")
            img = Image.open(image_input)
            original_format = img.format or 'JPEG'
        elif isinstance(image_input, Image.Image):
            # 直接使用PIL Image对象
            img = image_input.copy()
            original_format = 'JPEG'
        else:
            raise ValueError("image_input必须是文件路径字符串或PIL Image对象")
        
        # 转换为RGB模式（如果是RGBA等其他模式）
        if img.mode in ('RGBA', 'LA', 'P'):
            # 创建白色背景
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 获取原始尺寸
        original_width, original_height = img.size
        print(f"原始尺寸: {original_width}x{original_height}")
        
        # 计算新尺寸（保持宽高比）
        new_width, new_height = _calculate_new_size(
            original_width, original_height, max_width, max_height
        )
        
        # 如果需要缩放，则调整图片大小
        if new_width != original_width or new_height != original_height:
            print(f"缩放尺寸: {new_width}x{new_height}")
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            print("无需缩放")
        
        # 转换为base64
        base64_str = _image_to_base64_with_size_limit(img, max_size_mb, quality)
        
        return base64_str
        
    except Exception as e:
        raise Exception(f"图片处理失败: {str(e)}")


def _calculate_new_size(width: int, height: int, max_width: int, max_height: int) -> tuple:
    """
    计算保持宽高比的新尺寸
    
    Args:
        width: 原始宽度
        height: 原始高度
        max_width: 最大宽度
        max_height: 最大高度
        
    Returns:
        tuple: (新宽度, 新高度)
    """
    # 如果原始尺寸已经在限制范围内，不需要缩放
    if width <= max_width and height <= max_height:
        return width, height
    
    # 计算缩放比例
    width_ratio = max_width / width
    height_ratio = max_height / height
    
    # 选择较小的比例以确保两个维度都在限制内
    scale_ratio = min(width_ratio, height_ratio)
    
    # 计算新尺寸
    new_width = int(width * scale_ratio)
    new_height = int(height * scale_ratio)
    
    # 确保尺寸不会太小
    min_size = 100
    if new_width < min_size or new_height < min_size:
        aspect_ratio = width / height
        if aspect_ratio > 1:
            new_width = min_size
            new_height = int(min_size / aspect_ratio)
        else:
            new_height = min_size
            new_width = int(min_size * aspect_ratio)
    
    return new_width, new_height


def _image_to_base64_with_size_limit(img: Image.Image, max_size_mb: float, initial_quality: int) -> str:
    """
    将PIL Image转换为base64，并控制文件大小
    
    Args:
        img: PIL Image对象
        max_size_mb: 最大文件大小（MB）
        initial_quality: 初始JPEG质量
        
    Returns:
        str: base64编码的图片字符串
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    
    # 首先尝试PNG格式
    buffered = BytesIO()
    img.save(buffered, format="PNG", optimize=True)
    
    if buffered.tell() <= max_size_bytes:
        print(f"PNG格式，大小: {buffered.tell() / (1024*1024):.2f}MB")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # PNG太大，尝试JPEG格式
    print(f"PNG格式过大({buffered.tell() / (1024*1024):.2f}MB)，尝试JPEG格式")
    
    # 尝试不同的JPEG质量
    for quality in range(initial_quality, 10, -5):
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=quality, optimize=True)
        
        current_size_mb = buffered.tell() / (1024 * 1024)
        print(f"JPEG质量 {quality}，大小: {current_size_mb:.2f}MB")
        
        if buffered.tell() <= max_size_bytes:
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # 如果还是太大，使用最低质量
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=10, optimize=True)
    final_size_mb = buffered.tell() / (1024 * 1024)
    print(f"使用最低质量(10)，最终大小: {final_size_mb:.2f}MB")
    
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def check_image_needs_compression(
    image_path: str, 
    max_size_mb: float = 1.0,
    max_width: int = 1920,
    max_height: int = 1920
) -> dict:
    """
    检查图片是否需要压缩
    
    Args:
        image_path: 图片文件路径
        max_size_mb: 最大文件大小（MB）
        max_width: 最大宽度
        max_height: 最大高度
        
    Returns:
        dict: 包含检查结果的字典
            {
                'needs_compression': bool,  # 是否需要压缩
                'file_size_mb': float,      # 文件大小（MB）
                'width': int,               # 图片宽度
                'height': int,              # 图片高度
                'reasons': list             # 需要压缩的原因列表
            }
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 检查文件大小
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        
        # 检查图片尺寸
        with Image.open(image_path) as img:
            width, height = img.size
        
        # 判断是否需要压缩
        needs_compression = False
        reasons = []
        
        if file_size_mb > max_size_mb:
            needs_compression = True
            reasons.append(f"文件大小({file_size_mb:.2f}MB)超过限制({max_size_mb}MB)")
        
        if width > max_width:
            needs_compression = True
            reasons.append(f"宽度({width}px)超过限制({max_width}px)")
        
        if height > max_height:
            needs_compression = True
            reasons.append(f"高度({height}px)超过限制({max_height}px)")
        
        return {
            'needs_compression': needs_compression,
            'file_size_mb': file_size_mb,
            'width': width,
            'height': height,
            'reasons': reasons
        }
        
    except Exception as e:
        raise Exception(f"检查图片失败: {str(e)}")


# 示例用法
if __name__ == "__main__":
    # 测试函数
    test_image_path = "test.jpg"  # 替换为实际的图片路径
    
    try:
        # 检查是否需要压缩
        check_result = check_image_needs_compression(test_image_path)
        print("检查结果:", check_result)
        
        # 压缩并转换为base64
        if check_result['needs_compression']:
            print("\n开始压缩图片...")
            base64_str = compress_image_to_base64(test_image_path)
            print(f"\n压缩完成，base64长度: {len(base64_str)} 字符")
            print(f"base64前100字符: {base64_str[:100]}...")
        else:
            print("\n图片无需压缩，直接转换为base64")
            base64_str = compress_image_to_base64(test_image_path)
            print(f"\n转换完成，base64长度: {len(base64_str)} 字符")
            
    except Exception as e:
        print(f"处理失败: {e}")