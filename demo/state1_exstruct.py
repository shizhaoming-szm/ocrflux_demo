#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCRFlux Stage 1 Demo - Page to Markdown

这个demo演示了OCRFlux第一阶段的功能：将PDF页面或图片转换为Markdown格式
模型配置：
- 服务器地址: 192.168.8.36:8004
- 模型名称: OCRFlux-3B

输出格式：为第二阶段(Element Merge Detect)准备的输入格式
"""

import asyncio
import aiohttp
import base64
import json
import os
import sys
from io import BytesIO
from PIL import Image
from pypdf import PdfReader
from typing import Dict, List, Optional, Union

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocrflux.image_utils import get_page_image
from ocrflux.prompts import build_page_to_markdown_prompt


class OCRFluxStage1Demo:
    """OCRFlux第一阶段演示类"""
    
    def __init__(self, server_url: str = "http://192.168.8.36:8004", model_name: str = "OCRFlux-3B"):
        """
        初始化OCRFlux Stage 1 Demo
        
        Args:
            server_url: vLLM服务器地址
            model_name: 模型名称
        """
        self.server_url = server_url
        self.model_name = model_name
        self.api_url = f"{server_url}/v1/chat/completions"
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """将PIL Image转换为base64编码"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def _build_query(self, file_path: str, page_number: int, target_longest_image_dim: int = 1024) -> dict:
        """
        构建Stage 1查询请求
        
        Args:
            file_path: 文件路径（PDF或图片）
            page_number: 页面编号（从1开始）
            target_longest_image_dim: 图像最长边目标尺寸
            
        Returns:
            dict: API请求数据
        """
        # 获取页面图像
        image = get_page_image(file_path, page_number, target_longest_image_dim=target_longest_image_dim)
        
        # 转换为base64
        image_base64 = self._image_to_base64(image)
        
        # 构建提示词
        prompt = build_page_to_markdown_prompt()
        
        # 构建API请求
        query = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_base64
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.0
        }
        
        return query
    
    async def _call_api(self, query: dict) -> Optional[str]:
        """调用vLLM API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=query,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=300)  # 5分钟超时
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        print(f"API调用失败，状态码: {response.status}")
                        error_text = await response.text()
                        print(f"错误信息: {error_text}")
                        return None
        except Exception as e:
            print(f"API调用异常: {e}")
            return None
    
    def _parse_page_elements(self, markdown_text: str) -> List[Dict]:
        """
        解析页面Markdown文本为元素列表（为Stage 2准备）
        
        Args:
            markdown_text: Stage 1输出的Markdown文本
            
        Returns:
            List[Dict]: 元素列表，每个元素包含type、index、content等字段
        """
        elements = []
        lines = markdown_text.strip().split('\n')
        
        current_element = ""
        element_type = "text"
        index = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_element:
                    elements.append({
                        "type": element_type,
                        "index": index,
                        "content": current_element.strip()
                    })
                    current_element = ""
                    index += 1
                continue
            
            # 检测表格
            if line.startswith("<table") or "<table>" in line:
                if current_element:
                    elements.append({
                        "type": element_type,
                        "index": index,
                        "content": current_element.strip()
                    })
                    index += 1
                current_element = line
                element_type = "table"
            elif line.startswith("</table>") or "</table>" in line:
                current_element += "\n" + line
                elements.append({
                    "type": element_type,
                    "index": index,
                    "content": current_element.strip()
                })
                current_element = ""
                element_type = "text"
                index += 1
            # 检测图像
            elif line.startswith("<Image>") and line.endswith("</Image>"):
                if current_element:
                    elements.append({
                        "type": element_type,
                        "index": index,
                        "content": current_element.strip()
                    })
                    index += 1
                elements.append({
                    "type": "image",
                    "index": index,
                    "content": line
                })
                current_element = ""
                element_type = "text"
                index += 1
            # 检测标题
            elif line.startswith("#"):
                if current_element:
                    elements.append({
                        "type": element_type,
                        "index": index,
                        "content": current_element.strip()
                    })
                    index += 1
                elements.append({
                    "type": "heading",
                    "index": index,
                    "content": line
                })
                current_element = ""
                element_type = "text"
                index += 1
            else:
                if element_type == "table":
                    current_element += "\n" + line
                else:
                    if current_element:
                        current_element += "\n" + line
                    else:
                        current_element = line
        
        # 处理最后一个元素
        if current_element:
            elements.append({
                "type": element_type,
                "index": index,
                "content": current_element.strip()
            })
        
        return elements
    
    async def process_page(self, file_path: str, page_number: int) -> Optional[Dict]:
        """
        处理单个页面
        
        Args:
            file_path: 文件路径
            page_number: 页面编号
            
        Returns:
            Dict: 包含页面处理结果的字典
        """
        print(f"正在处理页面 {page_number}...")
        
        # 构建查询
        query = self._build_query(file_path, page_number)
        
        # 调用API
        markdown_result = await self._call_api(query)
        
        if markdown_result is None:
            print(f"页面 {page_number} 处理失败")
            return None
        
        # 解析元素（为Stage 2准备）
        elements = self._parse_page_elements(markdown_result)
        
        result = {
            "page_number": page_number,
            "markdown_text": markdown_result,
            "elements": elements,
            "element_count": len(elements)
        }
        
        print(f"页面 {page_number} 处理完成，识别到 {len(elements)} 个元素")
        return result
    
    async def process_document(self, file_path: str) -> Optional[Dict]:
        """
        处理整个文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict: 文档处理结果，格式适合Stage 2输入
        """
        print(f"开始处理文档: {file_path}")
        
        # 确定页面数量
        if file_path.lower().endswith(".pdf"):
            try:
                reader = PdfReader(file_path)
                num_pages = len(reader.pages)
            except Exception as e:
                print(f"读取PDF失败: {e}")
                return None
        else:
            # 图像文件
            num_pages = 1
        
        print(f"文档共 {num_pages} 页")
        
        # 处理所有页面
        page_results = {}
        failed_pages = []
        
        for page_num in range(1, num_pages + 1):
            result = await self.process_page(file_path, page_num)
            if result:
                page_results[page_num] = result
            else:
                failed_pages.append(page_num)
        
        # 构建Stage 2输入格式
        stage2_input = {
            "orig_path": file_path,
            "num_pages": num_pages,
            "page_to_markdown_result": {},  # Stage 2需要的格式
            "page_results": page_results,  # 详细结果
            "failed_pages": failed_pages
        }
        
        # 转换为Stage 2需要的格式
        for page_num, result in page_results.items():
            stage2_input["page_to_markdown_result"][page_num] = result["elements"]
        
        print(f"文档处理完成，成功处理 {len(page_results)} 页，失败 {len(failed_pages)} 页")
        return stage2_input


async def main():
    """主函数演示"""
    # 创建demo实例
    demo = OCRFluxStage1Demo(
        server_url="http://192.168.8.36:8004",
        model_name="OCRFlux-3B"
    )
    
    # 测试文件路径（请根据实际情况修改）
    test_file = "test.png"  # 或 "test.pdf"
    
    if not os.path.exists(test_file):
        print(f"测试文件 {test_file} 不存在，请提供有效的PDF或图片文件")
        return
    
    # 处理文档
    result = await demo.process_document(test_file)
    
    if result:
        print("\n=== 处理结果 ===")
        print(f"文件: {result['orig_path']}")
        print(f"页面数: {result['num_pages']}")
        print(f"成功页面: {len(result['page_to_markdown_result'])}")
        print(f"失败页面: {result['failed_pages']}")
        
        # 保存结果
        output_file = "stage1_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")
        
        # 显示第一页的处理结果示例
        if result['page_results']:
            first_page = list(result['page_results'].values())[0]
            print(f"\n=== 第一页示例 ===")
            print(f"元素数量: {first_page['element_count']}")
            print("Markdown内容:")
            print(first_page['markdown_text'][:500] + "..." if len(first_page['markdown_text']) > 500 else first_page['markdown_text'])
    else:
        print("文档处理失败")


if __name__ == "__main__":
    asyncio.run(main())