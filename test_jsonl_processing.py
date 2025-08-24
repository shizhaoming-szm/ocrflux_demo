#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCRFlux逐页JSONL处理功能测试脚本

该脚本用于测试新实现的逐页处理功能，包括：
1. 逐页PDF处理并生成JSONL格式数据
2. 相邻页面表格和段落的智能合并
3. 最终Markdown文档的生成

使用方法：
    python test_jsonl_processing.py --pdf_path ./test.pdf --model_path Qwen/Qwen2.5-VL-7B-Instruct

作者: OCRFlux团队
日期: 2024
"""

import argparse
import json
import os
import time
from pathlib import Path

try:
    from vllm import LLM
    from ocrflux.inference import parse_page_by_page_jsonl, parse
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装vllm和ocrflux依赖")
    exit(1)


def test_jsonl_processing(pdf_path, model_path, max_retries=2, output_dir="./test_output"):
    """
    测试逐页JSONL处理功能
    
    Args:
        pdf_path (str): PDF文件路径
        model_path (str): 模型路径
        max_retries (int): 最大重试次数
        output_dir (str): 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("OCRFlux逐页JSONL处理功能测试")
    print("=" * 60)
    
    # 检查PDF文件是否存在
    if not os.path.exists(pdf_path):
        print(f"错误: PDF文件不存在: {pdf_path}")
        return False
    
    print(f"PDF文件: {pdf_path}")
    print(f"模型路径: {model_path}")
    print(f"输出目录: {output_dir}")
    print(f"最大重试次数: {max_retries}")
    print()
    
    try:
        # 初始化vLLM推理引擎
        print("正在初始化vLLM推理引擎...")
        start_time = time.time()
        llm = LLM(model=model_path, max_model_len=8192)
        init_time = time.time() - start_time
        print(f"模型初始化完成，耗时: {init_time:.2f}秒")
        print()
        
        # 测试原始批量处理方式
        print("1. 测试原始批量处理方式")
        print("-" * 40)
        start_time = time.time()
        
        result_batch = parse(llm, pdf_path, max_page_retries=max_retries)
        batch_time = time.time() - start_time
        
        if result_batch:
            batch_output_path = os.path.join(output_dir, "result_batch.md")
            with open(batch_output_path, "w", encoding="utf-8") as f:
                f.write(result_batch["document_text"])
            
            print(f"✓ 批量处理成功")
            print(f"  - 总页数: {result_batch['num_pages']}")
            print(f"  - 失败页面: {result_batch['fallback_pages']}")
            print(f"  - 处理时间: {batch_time:.2f}秒")
            print(f"  - 输出文件: {batch_output_path}")
        else:
            print("✗ 批量处理失败")
            return False
        
        print()
        
        # 测试新的逐页JSONL处理方式
        print("2. 测试逐页JSONL处理方式")
        print("-" * 40)
        start_time = time.time()
        
        result_jsonl = parse_page_by_page_jsonl(llm, pdf_path, max_page_retries=max_retries)
        jsonl_time = time.time() - start_time
        
        if result_jsonl:
            # 保存最终Markdown
            jsonl_md_path = os.path.join(output_dir, "result_jsonl.md")
            with open(jsonl_md_path, "w", encoding="utf-8") as f:
                f.write(result_jsonl["final_markdown"])
            
            # 保存详细的JSONL数据
            jsonl_data_path = os.path.join(output_dir, "result_jsonl_data.json")
            with open(jsonl_data_path, "w", encoding="utf-8") as f:
                json.dump(result_jsonl, f, ensure_ascii=False, indent=2)
            
            # 保存每页的JSONL数据
            pages_dir = os.path.join(output_dir, "pages_jsonl")
            os.makedirs(pages_dir, exist_ok=True)
            for page_num, page_data in result_jsonl["page_jsonl_results"].items():
                page_file = os.path.join(pages_dir, f"page_{page_num}.json")
                with open(page_file, "w", encoding="utf-8") as f:
                    json.dump(page_data, f, ensure_ascii=False, indent=2)
            
            print(f"✓ 逐页JSONL处理成功")
            print(f"  - 总页数: {result_jsonl['num_pages']}")
            print(f"  - 失败页面: {result_jsonl['fallback_pages']}")
            print(f"  - 合并页面对: {len(result_jsonl['merged_elements'])}")
            print(f"  - 处理时间: {jsonl_time:.2f}秒")
            print(f"  - Markdown输出: {jsonl_md_path}")
            print(f"  - 详细数据: {jsonl_data_path}")
            print(f"  - 分页数据: {pages_dir}/")
            
            # 显示合并信息
            if result_jsonl['merged_elements']:
                print("  - 合并详情:")
                for (p1, p2), merge_info in result_jsonl['merged_elements'].items():
                    merge_pairs = merge_info.get('merge_pairs', [])
                    print(f"    页面 {p1}-{p2}: {len(merge_pairs)} 个合并项")
                    for pair in merge_pairs:
                        print(f"      - {pair['type']} (置信度: {pair.get('confidence', 'N/A')})")
            
        else:
            print("✗ 逐页JSONL处理失败")
            return False
        
        print()
        
        # 性能对比
        print("3. 性能对比")
        print("-" * 40)
        print(f"批量处理时间: {batch_time:.2f}秒")
        print(f"逐页处理时间: {jsonl_time:.2f}秒")
        print(f"时间差异: {jsonl_time - batch_time:+.2f}秒")
        
        if batch_time > 0:
            speed_ratio = jsonl_time / batch_time
            print(f"速度比率: {speed_ratio:.2f}x (>1表示逐页处理更慢)")
        
        print()
        
        # 结果对比
        print("4. 结果对比")
        print("-" * 40)
        batch_length = len(result_batch["document_text"])
        jsonl_length = len(result_jsonl["final_markdown"])
        
        print(f"批量处理输出长度: {batch_length} 字符")
        print(f"逐页处理输出长度: {jsonl_length} 字符")
        print(f"长度差异: {jsonl_length - batch_length:+d} 字符")
        
        # 保存对比报告
        report_path = os.path.join(output_dir, "comparison_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("OCRFlux处理方式对比报告\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"PDF文件: {pdf_path}\n")
            f.write(f"模型: {model_path}\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("批量处理结果:\n")
            f.write(f"  - 总页数: {result_batch['num_pages']}\n")
            f.write(f"  - 失败页面: {result_batch['fallback_pages']}\n")
            f.write(f"  - 处理时间: {batch_time:.2f}秒\n")
            f.write(f"  - 输出长度: {batch_length} 字符\n\n")
            
            f.write("逐页JSONL处理结果:\n")
            f.write(f"  - 总页数: {result_jsonl['num_pages']}\n")
            f.write(f"  - 失败页面: {result_jsonl['fallback_pages']}\n")
            f.write(f"  - 合并页面对: {len(result_jsonl['merged_elements'])}\n")
            f.write(f"  - 处理时间: {jsonl_time:.2f}秒\n")
            f.write(f"  - 输出长度: {jsonl_length} 字符\n\n")
            
            if result_jsonl['merged_elements']:
                f.write("合并详情:\n")
                for (p1, p2), merge_info in result_jsonl['merged_elements'].items():
                    merge_pairs = merge_info.get('merge_pairs', [])
                    f.write(f"  页面 {p1}-{p2}: {len(merge_pairs)} 个合并项\n")
                    for pair in merge_pairs:
                        f.write(f"    - {pair['type']} (置信度: {pair.get('confidence', 'N/A')})\n")
        
        print(f"对比报告已保存: {report_path}")
        print()
        print("✓ 测试完成！")
        return True
        
    except Exception as e:
        print(f"✗ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主函数：解析命令行参数并执行测试
    """
    parser = argparse.ArgumentParser(
        description="OCRFlux逐页JSONL处理功能测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python test_jsonl_processing.py --pdf_path ./test.pdf
  python test_jsonl_processing.py --pdf_path ./document.pdf --model_path Qwen/Qwen2.5-VL-7B-Instruct --max_retries 3
  python test_jsonl_processing.py --pdf_path ./sample.pdf --output_dir ./my_output
        """
    )
    
    parser.add_argument(
        "--pdf_path",
        type=str,
        default="./test.pdf",
        help="PDF文件路径 (默认: ./test.pdf)"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="模型路径 (默认: Qwen/Qwen2.5-VL-7B-Instruct)"
    )
    
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="最大重试次数 (默认: 2)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_output",
        help="输出目录 (默认: ./test_output)"
    )
    
    args = parser.parse_args()
    
    # 执行测试
    success = test_jsonl_processing(
        pdf_path=args.pdf_path,
        model_path=args.model_path,
        max_retries=args.max_retries,
        output_dir=args.output_dir
    )
    
    if success:
        print("\n🎉 所有测试通过！")
        exit(0)
    else:
        print("\n❌ 测试失败！")
        exit(1)


if __name__ == "__main__":
    main()