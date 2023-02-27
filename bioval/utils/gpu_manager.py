import pynvml

def get_available_gpu():
    """
    A function that returns the index of the GPU with the lowest combined metric of memory usage and utilization. 

    The function uses the pynvml library to retrieve the memory usage and utilization for each GPU in the system. The memory usage is expressed in MB and utilization in percent. 
    
    The combined metric is calculated as the sum of memory usage and half of utilization. The GPU with the lowest combined metric is considered to be the best GPU and its index is returned.

    Returns:
        int : index of the GPU with the lowest combined metric
        None if the system has no GPUs
    """
    try:
        pynvml.nvmlInit()
    except:
        return None

    best_gpu = 0
    lowest_metric = float('inf')
    gpu_count = pynvml.nvmlDeviceGetCount()
    if gpu_count == 0:
        return None

    for i in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        # Compute the combined metric
        metric = (meminfo.used / 1024**2) + util.gpu / 2
        if metric < lowest_metric:
            best_gpu = i
            lowest_metric = metric
        if util.gpu > 50 or meminfo.used / meminfo.total > 0.5:
            print(f"Warning: GPU {i} is already above 50% utilization or memory")


    pynvml.nvmlShutdown()

    return best_gpu
