# OCRFlux 服务器配置说明

## 概述

OCRFlux 现在支持通过参数配置服务器的IP地址和端口，使得系统更加灵活，可以连接到不同的推理服务器。

## 新增参数

### server_host
- **类型**: 字符串
- **默认值**: `'192.168.8.36'`
- **说明**: 推理服务器的IP地址或主机名
- **示例**: `'127.0.0.1'`, `'192.168.1.100'`, `'ai-server.company.com'`

### server_port
- **类型**: 整数
- **默认值**: `8004`
- **说明**: 推理服务器的端口号
- **示例**: `8004`, `8080`, `9000`

## 使用方法

### 1. 编程方式调用

```python
from control_page_proccess import run_with_config

config = {
    'workspace': './my_workspace',
    'task': 'pdf2markdown',
    'data': ['./document.pdf'],
    'workers': 4,
    'model': 'OCRFlux-3B',
    'port': 8004,
    'server_host': '192.168.1.100',  # 自定义服务器IP
    'server_port': 8005,             # 自定义服务器端口
    # ... 其他配置参数
}

await run_with_config(config)
```

### 2. 常见配置场景

#### 本地服务器
```python
config = {
    # ... 其他配置
    'server_host': '127.0.0.1',
    'server_port': 8004,
}
```

#### 远程服务器
```python
config = {
    # ... 其他配置
    'server_host': '192.168.8.36',
    'server_port': 8004,
}
```

#### 使用域名
```python
config = {
    # ... 其他配置
    'server_host': 'ai-server.company.com',
    'server_port': 80,
}
```

## 配置验证

系统会自动构建以下URL：
- **推理API**: `http://{server_host}:{server_port}/v1/chat/completions`
- **模型检测**: `http://{server_host}:{server_port}/v1/models`

## 注意事项

1. **网络连通性**: 确保客户端能够访问指定的服务器IP和端口
2. **防火墙设置**: 检查防火墙是否允许访问指定端口
3. **服务器状态**: 确保推理服务器已启动并正常运行
4. **端口冲突**: 避免使用已被占用的端口

## 测试配置

可以使用提供的测试脚本验证配置是否正确：

```bash
python test_server_config.py
```

## 示例文件

查看 `example_programmatic_usage.py` 文件中的完整示例，了解如何在不同场景下使用服务器配置参数。

## 兼容性

- 如果不指定 `server_host` 和 `server_port` 参数，系统将使用默认值
- 现有的配置文件无需修改即可继续使用
- 新参数完全向后兼容

## 故障排除

### 连接失败
1. 检查服务器IP和端口是否正确
2. 验证网络连通性：`ping {server_host}`
3. 检查端口是否开放：`telnet {server_host} {server_port}`
4. 查看服务器日志确认服务状态

### 配置错误
1. 确保IP地址格式正确
2. 确保端口号在有效范围内（1-65535）
3. 检查配置字典中的参数名称是否正确