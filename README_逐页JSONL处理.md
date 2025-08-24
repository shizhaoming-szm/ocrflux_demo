# OCRFlux 逐页JSONL处理功能说明

## 概述

OCRFlux新增了逐页JSONL处理功能，支持更精细的PDF处理流程。与原有的批量处理方式相比，新功能提供了以下优势：

- **逐页处理**：每页单独处理，避免大文档内存占用过高
- **JSONL格式**：结构化的页面数据，便于后续分析和处理
- **智能合并**：基于JSONL数据的相邻页面表格和段落合并
- **更好的错误处理**：单页失败不影响其他页面
- **详细的处理信息**：提供完整的处理过程和合并详情

## 功能特性

### 1. 三阶段处理流程

#### Stage 1: 逐页处理生成JSONL
- 逐页调用大模型进行OCR识别
- 将每页内容解析为结构化的JSONL格式
- 支持重试机制，提高处理成功率
- 分类识别文本、图像、表格等不同元素类型

#### Stage 2: 相邻页面合并检测
- 分析相邻页面的JSONL数据
- 智能检测需要合并的表格和段落
- 基于语义分析确定合并的必要性
- 提供合并置信度评估

#### Stage 3: 生成最终Markdown
- 整合所有页面的处理结果
- 应用检测到的合并操作
- 生成完整的Markdown文档
- 保留页面分隔信息便于调试

### 2. JSONL数据结构

每页的JSONL数据包含以下字段：

```json
{
  "page_number": 1,
  "language": "zh",
  "rotation": 0,
  "elements": [
    {
      "type": "text",
      "index": 0,
      "content": "页面文本内容",
      "is_mergeable": true
    },
    {
      "type": "table",
      "index": 1,
      "content": "<table>原始表格数据</table>",
      "html_content": "<table><tr><td>转换后的HTML</td></tr></table>",
      "is_mergeable": true
    },
    {
      "type": "image",
      "index": 2,
      "content": "<Image>图像描述</Image>",
      "description": "图像描述"
    }
  ],
  "raw_text": "原始识别文本"
}
```

### 3. 合并检测结果

合并检测返回的数据结构：

```json
{
  "has_merge": true,
  "merge_pairs": [
    {
      "type": "table",
      "page1_element_index": 2,
      "page2_element_index": 0,
      "merged_content": "<table>合并后的HTML表格</table>",
      "confidence": 0.95
    }
  ]
}
```

## 使用方法

### 1. 基本用法

```python
from vllm import LLM
from ocrflux.inference import parse_page_by_page_jsonl

# 初始化模型
llm = LLM(model="Qwen/Qwen2.5-VL-7B-Instruct", max_model_len=8192)

# 逐页处理PDF
result = parse_page_by_page_jsonl(llm, "./document.pdf", max_page_retries=2)

if result:
    # 保存最终Markdown
    with open("./output.md", "w", encoding="utf-8") as f:
        f.write(result["final_markdown"])
    
    # 保存详细数据
    import json
    with open("./output_data.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，共 {result['num_pages']} 页")
    print(f"失败页面: {result['fallback_pages']}")
    print(f"合并操作: {len(result['merged_elements'])} 个页面对")
else:
    print("处理失败")
```

### 2. 使用测试脚本

项目提供了完整的测试脚本 `test_jsonl_processing.py`：

```bash
# 基本测试
python test_jsonl_processing.py --pdf_path ./test.pdf

# 指定模型和重试次数
python test_jsonl_processing.py \
    --pdf_path ./document.pdf \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --max_retries 3 \
    --output_dir ./my_output
```

测试脚本会：
- 同时运行原始批量处理和新的逐页处理
- 对比两种方式的性能和结果
- 生成详细的对比报告
- 保存所有中间数据用于分析

### 3. 输出文件说明

运行后会生成以下文件：

```
test_output/
├── result_batch.md              # 原始批量处理结果
├── result_jsonl.md              # 逐页JSONL处理结果
├── result_jsonl_data.json       # 完整的处理数据
├── comparison_report.txt        # 对比报告
└── pages_jsonl/                 # 各页面的JSONL数据
    ├── page_1.json
    ├── page_2.json
    └── ...
```

## 参数说明

### parse_page_by_page_jsonl 函数参数

- `llm`: vLLM推理引擎实例
- `file_path`: PDF文件路径
- `max_page_retries`: 页面处理失败时的最大重试次数（默认0）

### 返回值字段

- `orig_path`: 原始文件路径
- `num_pages`: 总页数
- `page_jsonl_results`: 各页面的JSONL格式数据
- `merged_elements`: 合并后的元素信息
- `final_markdown`: 最终的Markdown文档
- `fallback_pages`: 解析失败的页面列表

## 性能对比

### 内存使用
- **批量处理**：所有页面同时加载到内存，大文档可能导致内存不足
- **逐页处理**：每次只处理一页，内存使用更稳定

### 处理时间
- **批量处理**：并行处理所有页面，速度较快
- **逐页处理**：串行处理，时间稍长，但提供更好的错误处理

### 错误恢复
- **批量处理**：单页失败可能影响整体结果
- **逐页处理**：单页失败不影响其他页面，支持重试机制

## 适用场景

### 推荐使用逐页JSONL处理的情况：

1. **大型文档**：页数较多的PDF文档
2. **内存受限**：运行环境内存有限
3. **需要详细分析**：需要分析每页的处理结果
4. **高可靠性要求**：需要确保每页都能正确处理
5. **后续处理**：需要基于结构化数据进行进一步分析

### 继续使用批量处理的情况：

1. **小型文档**：页数较少的PDF文档
2. **追求速度**：对处理时间有严格要求
3. **简单场景**：只需要最终的Markdown结果

## 注意事项

1. **模型要求**：需要使用支持视觉理解的大模型（如Qwen2.5-VL）
2. **内存配置**：虽然逐页处理减少了内存使用，但模型本身仍需要足够的GPU内存
3. **处理时间**：逐页处理通常比批量处理耗时更长
4. **合并准确性**：合并检测基于大模型的语义理解，可能存在误判
5. **文件格式**：目前支持PDF和常见图像格式

## 故障排除

### 常见问题

1. **页面处理失败**
   - 检查PDF文件是否损坏
   - 增加 `max_page_retries` 参数
   - 检查模型是否正确加载

2. **合并检测不准确**
   - 检查相邻页面的内容是否确实需要合并
   - 调整合并检测的提示词
   - 查看置信度评分

3. **内存不足**
   - 减少 `max_model_len` 参数
   - 使用更小的模型
   - 增加系统内存或GPU内存

4. **处理速度慢**
   - 使用更快的GPU
   - 减少重试次数
   - 考虑使用批量处理方式

### 调试技巧

1. **查看中间结果**：检查 `pages_jsonl/` 目录下的单页数据
2. **分析合并信息**：查看 `result_jsonl_data.json` 中的合并详情
3. **对比两种方式**：使用测试脚本对比批量和逐页处理的结果
4. **逐步调试**：先处理单页，确认无误后再处理整个文档

## 扩展开发

### 自定义合并逻辑

可以修改 `build_jsonl_merge_query` 函数来自定义合并检测逻辑：

```python
def custom_merge_query(page1_data, page2_data):
    # 自定义合并检测逻辑
    # 可以添加特定的业务规则
    pass
```

### 添加新的元素类型

在 `parse_page_elements_to_jsonl` 函数中添加新的元素类型识别：

```python
def parse_page_elements_to_jsonl(natural_text):
    # 添加新的元素类型识别逻辑
    if text.startswith("<custom>") and text.endswith("</custom>"):
        elements.append({
            "type": "custom",
            "index": i,
            "content": text,
            "is_mergeable": True
        })
```

### 自定义输出格式

修改 `build_final_markdown_from_jsonl` 函数来自定义最终输出格式：

```python
def custom_markdown_builder(page_jsonl_results, merged_elements, fallback_pages):
    # 自定义Markdown生成逻辑
    # 可以添加特定的格式要求
    pass
```

## 更新日志

### v0.2.0 (2024)
- 新增逐页JSONL处理功能
- 支持相邻页面智能合并
- 添加详细的测试脚本
- 提供完整的使用文档

---

如有问题或建议，请联系OCRFlux开发团队。