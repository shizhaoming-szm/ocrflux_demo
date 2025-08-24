# OCRFlux Stage 1 Demo 使用说明

这个demo演示了OCRFlux第一阶段的功能：将PDF页面或图片转换为Markdown格式。

## 配置信息

- **服务器地址**: `192.168.8.36:8004`
- **模型名称**: `OCRFlux-3B`
- **输出格式**: 为第二阶段(Element Merge Detect)准备的输入格式

## 功能特性

1. **多格式支持**: 支持PDF文件和图片文件（PNG、JPG等）
2. **多页处理**: 自动处理PDF的所有页面
3. **元素解析**: 将Markdown文本解析为结构化元素列表
4. **错误处理**: 包含完整的错误处理和重试机制
5. **Stage 2兼容**: 输出格式完全兼容第二阶段输入要求

## 使用方法

### 1. 基本使用

```python
import asyncio
from state1_exstruct import OCRFluxStage1Demo

# 创建demo实例
demo = OCRFluxStage1Demo(
    server_url="http://192.168.8.36:8004",
    model_name="OCRFlux-3B"
)

# 处理文档
result = await demo.process_document("your_file.pdf")
```

### 2. 直接运行demo

```bash
cd /path/to/OCRFlux/demo
python state1_exstruct.py
```

**注意**: 请确保在运行前将测试文件放在demo目录下，并修改`main()`函数中的`test_file`变量。

### 3. 处理单个页面

```python
# 处理PDF的第一页
page_result = await demo.process_page("document.pdf", 1)

# 处理图片文件
page_result = await demo.process_page("image.png", 1)
```

## 输出格式

### 完整文档处理结果

```json
{
  "orig_path": "test.pdf",
  "num_pages": 3,
  "page_to_markdown_result": {
    "1": [
      {
        "type": "heading",
        "index": 0,
        "content": "# 文档标题"
      },
      {
        "type": "text",
        "index": 1,
        "content": "这是一段文本内容..."
      },
      {
        "type": "table",
        "index": 2,
        "content": "<table>...</table>"
      }
    ]
  },
  "page_results": {
    "1": {
      "page_number": 1,
      "markdown_text": "# 文档标题\n\n这是一段文本内容...",
      "elements": [...],
      "element_count": 3
    }
  },
  "failed_pages": []
}
```

### 元素类型说明

- **heading**: 标题元素（以#开头）
- **text**: 普通文本段落
- **table**: HTML表格（`<table>...</table>`）
- **image**: 图像元素（`<Image>(x1,y1),(x2,y2)</Image>`）

## Stage 2 兼容性

输出的`page_to_markdown_result`字段完全兼容OCRFlux Stage 2的输入要求：

```python
# Stage 2可以直接使用这个格式
stage2_input = result["page_to_markdown_result"]
# stage2_input[page_number] = [element1, element2, ...]
```

## 错误处理

1. **API调用失败**: 自动重试和错误日志
2. **文件读取失败**: 返回None并记录错误信息
3. **页面处理失败**: 记录在`failed_pages`列表中
4. **网络超时**: 设置5分钟超时时间

## 依赖要求

```bash
pip install aiohttp pillow pypdf
```

确保OCRFlux项目的相关模块可以正常导入：
- `ocrflux.image_utils`
- `ocrflux.prompts`

## 注意事项

1. **服务器连接**: 确保`192.168.8.36:8004`服务器正常运行
2. **模型加载**: 确保服务器已加载`OCRFlux-3B`模型
3. **文件路径**: 使用绝对路径或确保文件在正确位置
4. **内存使用**: 大文档可能需要较多内存，建议分批处理
5. **网络稳定**: 确保网络连接稳定，避免API调用中断

## 示例输出

```
开始处理文档: test.pdf
文档共 2 页
正在处理页面 1...
页面 1 处理完成，识别到 5 个元素
正在处理页面 2...
页面 2 处理完成，识别到 3 个元素
文档处理完成，成功处理 2 页，失败 0 页

=== 处理结果 ===
文件: test.pdf
页面数: 2
成功页面: 2
失败页面: []

结果已保存到: stage1_result.json

=== 第一页示例 ===
元素数量: 5
Markdown内容:
# 文档标题

这是第一段文本内容，包含了文档的基本信息...
```

## 扩展开发

可以基于这个demo进行扩展开发：

1. **批量处理**: 添加多文件批量处理功能
2. **结果缓存**: 添加处理结果缓存机制
3. **进度监控**: 添加处理进度显示
4. **格式转换**: 添加其他输出格式支持
5. **质量评估**: 添加处理质量评估功能