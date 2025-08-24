#!/bin/bash

# OCRFlux评估脚本
# 该脚本用于评估OCRFlux在四个核心任务上的性能表现
# 每个任务包含两个步骤：1) 运行推理生成结果 2) 与标准答案对比评估

# ============================================================================
# 任务1: page_to_markdown (页面转Markdown)
# 功能：将PDF页面转换为结构化的Markdown格式
# 数据集：OCRFlux-bench-single (单页PDF文档)
# ============================================================================

# 步骤1：运行OCRFlux pipeline进行页面到Markdown的转换
# --task pdf2markdown: 指定任务类型为PDF转Markdown
# --data: 输入PDF文件路径（支持通配符）
# --model: 使用的视觉语言模型路径
python -m ocrflux.pipeline ./eval_page_to_markdown_result --task pdf2markdown --data /data/OCRFlux-bench-single/pdfs/*.pdf --model /data/OCRFlux-7B

# 步骤2：评估生成的Markdown与标准答案的相似度
# 使用BLEU、ROUGE等指标评估文本质量
python -m eval.eval_page_to_markdown ./eval_page_to_markdown_result --gt_file /data/OCRFlux-bench-single/data.jsonl

# ============================================================================
# 任务2: element_merge_detect (元素合并检测)
# 功能：检测跨页面的文档元素（如表格、段落）并进行合并
# 数据集：OCRFlux-bench-cross (跨页文档)
# ============================================================================

# 步骤1：生成元素合并检测的测试数据
# 从跨页文档中提取需要合并的元素信息
python -m eval.gen_element_merge_detect_data /data/OCRFlux-bench-cross

# 步骤2：运行OCRFlux pipeline进行跨页元素合并
# --task merge_pages: 指定任务类型为页面合并
# 输入为JSON格式的页面结构化数据
python -m ocrflux.pipeline ./eval_element_merge_detect_result --task merge_pages --data /data/OCRFlux-bench-cross/jsons/*.json --model /data/OCRFlux-7B

# 步骤3：评估元素合并的准确性
# 检查是否正确识别和合并了跨页元素
python -m eval.eval_element_merge_detect ./eval_element_merge_detect_result --gt_file /data/OCRFlux-bench-cross/data.jsonl

# ============================================================================
# 任务3: table_to_html (表格转HTML)
# 功能：将表格图像转换为结构化的HTML表格
# 数据集：OCRFlux-pubtabnet-single (单个表格图像)
# ============================================================================

# 步骤1：运行OCRFlux pipeline进行表格识别和HTML转换
# 输入为表格图像（PNG格式），输出为HTML表格结构
python -m ocrflux.pipeline ./eval_table_to_html_result --task pdf2markdown --data /data/OCRFlux-pubtabnet-single/images/*.png --model /data/OCRFlux-7B

# 步骤2：评估生成的HTML表格结构准确性
# 比较表格的行列结构、单元格内容和合并情况
python -m eval.eval_table_to_html ./eval_table_to_html_result --gt_file /data/OCRFlux-pubtabnet-single/data.jsonl

# ============================================================================
# 任务4: html_table_merge (HTML表格合并)
# 功能：合并跨页面分割的表格，重建完整的表格结构
# 数据集：OCRFlux-pubtabnet-cross (跨页表格)
# ============================================================================

# 步骤1：生成HTML表格合并的测试数据
# 从跨页表格中提取需要合并的表格片段
python -m eval.gen_html_table_merge_data /data/OCRFlux-pubtabnet-cross

# 步骤2：运行OCRFlux pipeline进行跨页表格合并
# --task merge_tables: 指定任务类型为表格合并
# 输入为JSON格式的表格片段数据
python -m ocrflux.pipeline ./eval_html_table_merge_result --task merge_tables --data /data/OCRFlux-pubtabnet-cross/jsons/*.json --model /data/OCRFlux-7B

# 步骤3：评估表格合并的准确性
# 检查合并后的表格是否保持了正确的结构和内容
python -m eval.eval_html_table_merge ./eval_html_table_merge_result --gt_file /data/OCRFlux-pubtabnet-cross/data.jsonl


