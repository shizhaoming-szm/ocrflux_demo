# OCRFlux评估并行处理工具模块
# 该模块提供了带进度条的并行处理功能，用于加速评估脚本的执行
# 特别适用于大批量文档处理和模型推理任务

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=0):
    """
    带进度条的并行map函数实现
    
    该函数提供了Python内置map函数的并行版本，使用多进程池来加速处理大量数据。
    特别适用于OCRFlux评估中的批量文档处理、模型推理等CPU密集型任务。
    
    功能特点：
    - 多进程并行处理，充分利用多核CPU
    - 实时进度条显示，便于监控处理进度
    - 串行预处理，便于调试和错误捕获
    - 异常处理，确保单个任务失败不影响整体流程
    
    Args:
        array (array-like): 要处理的数据数组，每个元素将作为function的输入
        function (function): 要应用到数组元素的Python函数
                           通常是文档处理、模型推理等耗时操作
        n_jobs (int, default=16): 使用的进程数，建议设置为CPU核心数
                                 设置为1时退化为串行处理，便于调试
        use_kwargs (boolean, default=False): 是否将数组元素作为关键字参数传递
                                           True时，数组元素应为字典格式
        front_num (int, default=3): 并行处理前串行执行的迭代次数
                                   用于提前发现函数中的bug，避免大量进程同时失败
    
    Returns:
        list: 处理结果列表，格式为[function(array[0]), function(array[1]), ...]
              如果某个任务出现异常，对应位置将包含异常对象
    
    Example:
        # 并行处理文档列表
        documents = ['doc1.pdf', 'doc2.pdf', 'doc3.pdf']
        results = parallel_process(documents, process_document, n_jobs=8)
        
        # 使用关键字参数
        tasks = [{'file': 'doc1.pdf', 'model': 'model1'}, 
                {'file': 'doc2.pdf', 'model': 'model2'}]
        results = parallel_process(tasks, process_with_model, use_kwargs=True)
    """
    # 串行执行前几个迭代以提前发现bug
    # 这样可以避免在并行处理中大量进程同时失败
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    else:
        front = []
    
    # 如果设置n_jobs为1，则使用串行处理
    # 这对于基准测试和调试非常有用
    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    # 创建进程池执行器
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # 将数组元素提交给函数进行处理
        if use_kwargs:
            # 使用关键字参数模式：数组元素为字典，解包为函数参数
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            # 使用位置参数模式：数组元素直接作为函数参数
            futures = [pool.submit(function, a) for a in array[front_num:]]
        
        # 配置进度条参数
        kwargs = {
            'total': len(futures),    # 总任务数
            'unit': 'it',            # 单位显示
            'unit_scale': True,      # 自动缩放单位
            'leave': True            # 完成后保留进度条
        }
        
        # 显示任务完成进度
        # 使用as_completed确保按完成顺序更新进度条
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    # 从futures中获取结果
    # 注意：这里按提交顺序获取结果，保持输出顺序与输入一致
    for i, future in tqdm(enumerate(futures)):
        try:
            # 获取任务执行结果
            out.append(future.result())
        except Exception as e:
            # 如果任务执行失败，将异常对象添加到结果中
            # 这样可以继续处理其他任务，而不会因单个失败而中断
            out.append(e)
    
    # 返回串行处理结果 + 并行处理结果
    return front + out
