# OCRFlux 项目架构与启动流程说明

## 项目概述

OCRFlux是一个基于视觉语言模型的文档理解和处理系统，专门用于将PDF文档转换为结构化的Markdown格式。系统采用三阶段处理流程，能够处理复杂的文档布局、表格识别和跨页元素合并。

## 核心功能

### 三阶段处理流程

1. **Stage 1: Page to Markdown (页面转Markdown)**
   - 将PDF页面转换为结构化的Markdown格式
   - 识别文本、图像、表格等文档元素
   - 保持原始文档的层次结构

2. **Stage 2: Element Merge Detect (元素合并检测)**
   - 检测跨页面的文档元素（表格、段落等）
   - 识别需要合并的元素边界
   - 为下一阶段的合并提供指导

3. **Stage 3: HTML Table Merge (HTML表格合并)**
   - 合并跨页面分割的表格
   - 重建完整的表格结构
   - 处理复杂的表格合并和单元格对齐

## 项目结构

```
OCRFlux/
├── ocrflux/                    # 核心模块目录
│   ├── pipeline.py             # 主处理流水线（核心入口）
│   ├── client.py               # 客户端调用接口
│   ├── inference.py            # 推理引擎和解析逻辑
│   ├── prompts.py              # 提示词构建和数据结构
│   ├── image_utils.py          # 图像处理工具函数
│   ├── work_queue.py           # 工作队列管理
│   ├── table_format.py         # 表格格式转换工具
│   ├── metrics.py              # 指标统计和监控
│   ├── check.py                # 依赖检查工具
│   └── server.sh               # VLLM服务器启动脚本
├── eval/                       # 评估脚本目录
│   ├── eval.sh                 # 主评估脚本
│   ├── parallel.py             # 并行处理工具
│   ├── eval_*.py               # 各任务评估脚本
│   └── gen_*.py                # 测试数据生成脚本
└── images/                     # 示例图像目录
```

## 启动流程

### 1. 环境准备

#### 依赖检查
```bash
# 检查系统依赖
python -m ocrflux.check
```

该命令会检查：
- **Poppler工具集**：用于PDF处理和页面渲染
- **VLLM库**：高性能大语言模型推理引擎
- **PyTorch**：深度学习框架（可选GPU支持）

#### 必要依赖
- Python 3.8+
- VLLM (需要单独安装)
- PyTorch (建议GPU版本)
- Poppler工具集
- 其他Python依赖（见requirements.txt）

### 2. 模型服务启动

#### 启动VLLM服务器
```bash
# 启动视觉语言模型服务
./ocrflux/server.sh <模型路径> <端口号>

# 示例
./ocrflux/server.sh /data/OCRFlux-7B 8000
```

服务器配置：
- `--max-model-len 8192`：支持长文档处理
- `--gpu_memory_utilization 0.8`：合理的GPU内存使用率

### 3. 文档处理

#### 基本使用
```bash
# PDF转Markdown
python -m ocrflux.pipeline <输出目录> --task pdf2markdown --data <PDF文件路径> --model <模型路径>

# 跨页元素合并
python -m ocrflux.pipeline <输出目录> --task merge_pages --data <JSON文件路径> --model <模型路径>

# 跨页表格合并
python -m ocrflux.pipeline <输出目录> --task merge_tables --data <JSON文件路径> --model <模型路径>
```

## 模块调用关系

### 核心调用链

```
pipeline.py (主入口)
    ├── client.py (客户端接口)
    │   └── inference.py (推理引擎)
    │       ├── prompts.py (提示词构建)
    │       └── image_utils.py (图像处理)
    ├── work_queue.py (任务队列)
    └── metrics.py (性能监控)
```

### 详细调用流程

1. **pipeline.py** 作为主入口：
   - 解析命令行参数
   - 初始化工作队列和指标收集器
   - 协调整个处理流程

2. **client.py** 提供统一接口：
   - 封装不同任务的处理逻辑
   - 管理与VLLM服务器的通信
   - 处理批量请求和重试机制

3. **inference.py** 执行核心推理：
   - 构建模型输入（文本+图像）
   - 调用视觉语言模型
   - 解析和后处理模型输出

4. **辅助模块**：
   - **prompts.py**：构建任务特定的提示词
   - **image_utils.py**：处理PDF页面图像
   - **table_format.py**：处理表格格式转换
   - **work_queue.py**：管理异步任务队列
   - **metrics.py**：收集性能指标

## 异步处理机制

### 工作队列系统
- 使用`asyncio`实现异步处理
- 支持并发处理多个文档
- 自动负载均衡和错误恢复

### 指标监控
- 实时统计处理速度（tokens/sec）
- 跟踪工作进程状态
- 提供性能分析数据

## 评估系统

### 四个评估任务
1. **page_to_markdown**：页面转换质量评估
2. **element_merge_detect**：元素合并检测准确性
3. **table_to_html**：表格识别准确性
4. **html_table_merge**：表格合并质量

### 评估流程
```bash
# 运行完整评估
./eval/eval.sh
```

每个任务包含：
1. 数据生成/准备
2. 模型推理
3. 结果评估

## 配置和优化

### 性能调优
- 调整VLLM服务器参数
- 优化并发处理数量
- 配置GPU内存使用

### 模型配置
- 支持不同的视觉语言模型
- 可配置的提示词模板
- 灵活的输出格式

## 故障排除

### 常见问题
1. **依赖缺失**：运行`python -m ocrflux.check`检查
2. **GPU内存不足**：调整`--gpu_memory_utilization`参数
3. **模型加载失败**：检查模型路径和格式
4. **处理速度慢**：增加并发数或优化硬件配置

### 日志和调试
- 查看详细的错误日志
- 使用指标监控定位性能瓶颈
- 启用调试模式获取更多信息

## 扩展开发

### 添加新任务
1. 在`prompts.py`中定义新的提示词
2. 在`client.py`中添加任务处理逻辑
3. 在`pipeline.py`中注册新任务

### 自定义模型
1. 确保模型兼容VLLM接口
2. 调整提示词格式
3. 测试和优化性能

这个架构设计确保了OCRFlux的高性能、可扩展性和易维护性，为文档理解任务提供了完整的解决方案。