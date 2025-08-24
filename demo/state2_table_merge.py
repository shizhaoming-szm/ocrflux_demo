#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCRFlux Stage 2 Demo - Element Merge Detect

这个demo演示了OCRFlux第二阶段的功能：元素合并检测
功能：
- 检测跨页面的文档元素（表格、段落等）
- 识别需要合并的元素边界
- 为第三阶段的表格合并提供指导

模型配置：
- 服务器地址: 192.168.8.36:8004
- 模型名称: OCRFlux-3B

输入：第一阶段的输出结果
输出：为第三阶段(HTML Table Merge)准备的输入格式
"""

import asyncio
import aiohttp
import base64
import json
import os
import sys
from io import BytesIO
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocrflux.prompts import build_element_merge_detect_prompt, build_html_table_merge_prompt


class OCRFluxStage2Demo:
    """OCRFlux第二阶段演示类 - 元素合并检测"""
    
    def __init__(self, server_url: str = "http://192.168.8.36:8004", model_name: str = "OCRFlux-3B"):
        """
        初始化OCRFlux Stage 2 Demo
        
        Args:
            server_url: vLLM服务器地址
            model_name: 模型名称
        """
        self.server_url = server_url
        self.model_name = model_name
        self.api_url = f"{server_url}/v1/chat/completions"
    
    def _create_placeholder_image(self) -> str:
        """创建占位符图像的base64编码（Stage 2主要处理文本）"""
        image = Image.new('RGB', (28, 28), color='black')
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def _build_element_merge_query(self, text_list_1: List[str], text_list_2: List[str]) -> dict:
        """
        构建元素合并检测查询请求
        
        Args:
            text_list_1: 前一页的文本元素列表
            text_list_2: 后一页的文本元素列表
            
        Returns:
            dict: API请求数据
        """
        # 构建提示词
        prompt = build_element_merge_detect_prompt(text_list_1, text_list_2)
        
        # 创建占位符图像
        image_base64 = self._create_placeholder_image()
        
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
    
    def _build_html_table_merge_query(self, table1: str, table2: str) -> dict:
        """
        构建HTML表格合并查询请求（Stage 3功能）
        
        Args:
            table1: 第一个HTML表格
            table2: 第二个HTML表格
            
        Returns:
            dict: API请求数据
        """
        # 构建提示词
        prompt = build_html_table_merge_prompt(table1, table2)
        
        # 创建占位符图像
        image_base64 = self._create_placeholder_image()
        
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
    
    def _parse_merge_pairs(self, response_text: str) -> List[Tuple[int, int]]:
        """
        解析模型返回的合并对
        
        Args:
            response_text: 模型返回的文本，格式如 "[(0, 1), (2, 3)]" 或 "[]"
            
        Returns:
            List[Tuple[int, int]]: 需要合并的元素索引对列表
        """
        try:
            # 尝试直接解析为Python列表
            import ast
            pairs = ast.literal_eval(response_text.strip())
            if isinstance(pairs, list):
                return [(int(p[0]), int(p[1])) for p in pairs if len(p) == 2]
            else:
                return []
        except Exception as e:
            print(f"解析合并对失败: {e}")
            print(f"原始响应: {response_text}")
            return []
    
    def _extract_text_from_elements(self, elements: List[Dict]) -> List[str]:
        """
        从元素列表中提取文本内容
        
        Args:
            elements: 元素列表，每个元素包含type、index、content字段
            
        Returns:
            List[str]: 文本内容列表
        """
        return [elem["content"] for elem in elements]
    
    def _identify_table_pairs(self, merge_pairs: List[Tuple[int, int]], 
                             elements_1: List[Dict], elements_2: List[Dict]) -> List[Tuple[int, int, int, int]]:
        """
        识别需要进行表格合并的元素对
        
        Args:
            merge_pairs: 需要合并的元素索引对
            elements_1: 第一页的元素列表
            elements_2: 第二页的元素列表
            
        Returns:
            List[Tuple[int, int, int, int]]: 表格合并信息 (page1, page2, elem_idx1, elem_idx2)
        """
        table_pairs = []
        
        for idx1, idx2 in merge_pairs:
            # 检查索引是否有效
            if idx1 < len(elements_1) and idx2 < len(elements_2):
                elem1 = elements_1[idx1]
                elem2 = elements_2[idx2]
                
                # 检查是否都是表格元素
                if elem1.get("type") == "table" and elem2.get("type") == "table":
                    table_pairs.append((1, 2, idx1, idx2))  # 假设处理页面1和页面2
        
        return table_pairs
    
    async def detect_element_merges(self, page_to_markdown_result: Dict[str, List[Dict]]) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """
        检测跨页面的元素合并需求
        
        Args:
            page_to_markdown_result: Stage 1的输出结果，格式为 {page_num: [elements]}
            
        Returns:
            Dict[Tuple[int, int], List[Tuple[int, int]]]: 元素合并检测结果
            键为页面对 (page1, page2)，值为需要合并的元素索引对列表
        """
        print("开始元素合并检测...")
        
        element_merge_result = {}
        page_numbers = sorted([int(k) for k in page_to_markdown_result.keys()])
        
        # 检测相邻页面之间的元素合并需求
        for i in range(len(page_numbers) - 1):
            page1 = page_numbers[i]
            page2 = page_numbers[i + 1]
            
            print(f"检测页面 {page1} 和 {page2} 之间的元素合并...")
            
            # 获取两页的元素
            elements_1 = page_to_markdown_result[str(page1)]
            elements_2 = page_to_markdown_result[str(page2)]
            
            # 提取文本内容
            text_list_1 = self._extract_text_from_elements(elements_1)
            text_list_2 = self._extract_text_from_elements(elements_2)
            
            # 构建查询并调用API
            query = self._build_element_merge_query(text_list_1, text_list_2)
            response = await self._call_api(query)
            
            if response:
                # 解析合并对
                merge_pairs = self._parse_merge_pairs(response)
                element_merge_result[(page1, page2)] = merge_pairs
                print(f"页面 {page1}-{page2} 检测到 {len(merge_pairs)} 个合并对: {merge_pairs}")
            else:
                element_merge_result[(page1, page2)] = []
                print(f"页面 {page1}-{page2} 检测失败")
        
        return element_merge_result
    
    async def merge_html_tables(self, page_to_markdown_result: Dict[str, List[Dict]], 
                               element_merge_result: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> Dict[Tuple[int, int, int, int], str]:
        """
        合并跨页面的HTML表格（Stage 3功能）
        
        Args:
            page_to_markdown_result: Stage 1的输出结果
            element_merge_result: Stage 2的输出结果
            
        Returns:
            Dict[Tuple[int, int, int, int], str]: HTML表格合并结果
            键为 (page1, page2, elem_idx1, elem_idx2)，值为合并后的HTML表格
        """
        print("开始HTML表格合并...")
        
        html_table_merge_result = {}
        
        for (page1, page2), merge_pairs in element_merge_result.items():
            elements_1 = page_to_markdown_result[str(page1)]
            elements_2 = page_to_markdown_result[str(page2)]
            
            # 识别表格合并对
            table_pairs = self._identify_table_pairs(merge_pairs, elements_1, elements_2)
            
            for page1, page2, elem_idx1, elem_idx2 in table_pairs:
                print(f"合并表格: 页面{page1}元素{elem_idx1} + 页面{page2}元素{elem_idx2}")
                
                table1 = elements_1[elem_idx1]["content"]
                table2 = elements_2[elem_idx2]["content"]
                
                # 构建查询并调用API
                query = self._build_html_table_merge_query(table1, table2)
                response = await self._call_api(query)
                
                if response:
                    html_table_merge_result[(page1, page2, elem_idx1, elem_idx2)] = response.strip()
                    print(f"表格合并成功")
                else:
                    print(f"表格合并失败")
        
        return html_table_merge_result
    
    def build_document_text(self, page_to_markdown_result: Dict[str, List[Dict]], 
                           element_merge_result: Dict[Tuple[int, int], List[Tuple[int, int]]], 
                           html_table_merge_result: Dict[Tuple[int, int, int, int], str]) -> str:
        """
        从三个阶段的处理结果构建最终文档文本
        
        Args:
            page_to_markdown_result: Stage 1结果
            element_merge_result: Stage 2结果
            html_table_merge_result: Stage 3结果
            
        Returns:
            str: 最终的完整文档Markdown文本
        """
        print("构建最终文档文本...")
        
        # 创建副本以避免修改原始数据
        import copy
        result_copy = copy.deepcopy(page_to_markdown_result)
        
        # 将字符串键转换为整数键以便排序
        page_result = {}
        for k, v in result_copy.items():
            page_result[int(k)] = [elem["content"] for elem in v]
        
        page_keys = list(page_result.keys())
        element_merge_keys = list(element_merge_result.keys())
        html_table_merge_keys = list(html_table_merge_result.keys())
        
        # Stage 3: 处理HTML表格合并
        for page_1, page_2, elem_idx_1, elem_idx_2 in sorted(html_table_merge_keys, key=lambda x: -x[0]):
            if page_1 in page_result and page_2 in page_result:
                if elem_idx_1 < len(page_result[page_1]) and elem_idx_2 < len(page_result[page_2]):
                    page_result[page_1][elem_idx_1] = html_table_merge_result[(page_1, page_2, elem_idx_1, elem_idx_2)]
                    page_result[page_2][elem_idx_2] = ''
        
        # Stage 2: 处理跨页面元素合并
        for page_1, page_2 in sorted(element_merge_keys, key=lambda x: -x[0]):
            if (page_1, page_2) in element_merge_result:
                for elem_idx_1, elem_idx_2 in element_merge_result[(page_1, page_2)]:
                    if (page_1 in page_result and page_2 in page_result and 
                        elem_idx_1 < len(page_result[page_1]) and elem_idx_2 < len(page_result[page_2])):
                        
                        text1 = page_result[page_1][elem_idx_1]
                        text2 = page_result[page_2][elem_idx_2]
                        
                        # 智能文本连接
                        if len(text1) == 0 or text1[-1] == '-' or ('\u4e00' <= text1[-1] <= '\u9fff'):
                            # 空文本、连字符结尾、中文字符结尾：直接连接
                            page_result[page_1][elem_idx_1] = text1 + text2
                        else:
                            # 其他情况：用空格连接
                            page_result[page_1][elem_idx_1] = text1 + ' ' + text2
                        
                        # 清空第二个位置的文本
                        page_result[page_2][elem_idx_2] = ''
        
        # 组装最终文档文本
        document_text_list = []
        for page in sorted(page_keys):
            # 过滤掉空字符串
            page_text_list = [s for s in page_result[page] if s.strip()]
            document_text_list.extend(page_text_list)
        
        return "\n\n".join(document_text_list)
    
    async def process_stage1_result(self, stage1_result: Dict) -> Dict:
        """
        处理Stage 1的结果，执行Stage 2和Stage 3
        
        Args:
            stage1_result: Stage 1的输出结果
            
        Returns:
            Dict: 包含所有三个阶段结果的完整输出
        """
        print("开始处理Stage 1结果...")
        
        # 提取Stage 1的结果
        page_to_markdown_result = stage1_result.get("page_to_markdown_result", {})
        
        if not page_to_markdown_result:
            print("Stage 1结果为空，无法继续处理")
            return stage1_result
        
        # Stage 2: 元素合并检测
        element_merge_result = await self.detect_element_merges(page_to_markdown_result)
        
        # Stage 3: HTML表格合并
        html_table_merge_result = await self.merge_html_tables(page_to_markdown_result, element_merge_result)
        
        # 构建最终文档文本
        document_text = self.build_document_text(page_to_markdown_result, element_merge_result, html_table_merge_result)
        
        # 构建完整结果
        final_result = {
            **stage1_result,  # 保留Stage 1的所有结果
            "element_merge_detect_result": element_merge_result,
            "html_table_merge_result": html_table_merge_result,
            "document_text": document_text,
            "page_texts": []
        }
        
        # 构建每页的文本
        page_numbers = sorted([int(k) for k in page_to_markdown_result.keys()])
        for page_num in page_numbers:
            elements = page_to_markdown_result[str(page_num)]
            page_text = "\n\n".join([elem["content"] for elem in elements if elem["content"].strip()])
            final_result["page_texts"].append(page_text)
        
        print(f"处理完成，最终文档长度: {len(document_text)} 字符")
        return final_result


async def main():
    """主函数演示"""
    # 创建demo实例
    demo = OCRFluxStage2Demo(
        server_url="http://192.168.8.36:8004",
        model_name="OCRFlux-3B"
    )
    
    # 读取Stage 1的结果
    stage1_file = "stage1_result.json"
    
    if not os.path.exists(stage1_file):
        print(f"Stage 1结果文件 {stage1_file} 不存在")
        print("请先运行 state1_exstruct.py 生成Stage 1结果")
        return
    
    try:
        with open(stage1_file, 'r', encoding='utf-8') as f:
            stage1_result = json.load(f)
        
        print(f"读取Stage 1结果: {stage1_file}")
        print(f"文档页数: {stage1_result.get('num_pages', 0)}")
        print(f"成功页面: {len(stage1_result.get('page_to_markdown_result', {}))}")
        
        # 处理Stage 1结果，执行Stage 2和Stage 3
        final_result = await demo.process_stage1_result(stage1_result)
        
        # 保存最终结果
        output_file = "stage2_3_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== 处理结果 ===")
        print(f"原始文件: {final_result['orig_path']}")
        print(f"总页数: {final_result['num_pages']}")
        print(f"元素合并检测结果: {len(final_result['element_merge_detect_result'])} 个页面对")
        print(f"HTML表格合并结果: {len(final_result['html_table_merge_result'])} 个表格")
        print(f"最终文档长度: {len(final_result['document_text'])} 字符")
        print(f"\n结果已保存到: {output_file}")
        
        # 保存最终Markdown文档
        markdown_file = "final_document.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(final_result['document_text'])
        print(f"最终Markdown文档已保存到: {markdown_file}")
        
        # 显示文档预览
        print(f"\n=== 文档预览 ===\n")
        preview_text = final_result['document_text'][:1000]
        print(preview_text + ("..." if len(final_result['document_text']) > 1000 else ""))
        
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())