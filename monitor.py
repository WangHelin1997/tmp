import torch
import time
import threading
import pynvml
import signal
import sys

# 初始化pynvml
pynvml.nvmlInit()

# 用于线程控制的标志
stop_threads = False

def keep_gpu_busy(device):
    global stop_threads
    print(f"Starting keep_gpu_busy on device: {device}")
    # Create a large tensor to perform operations on
    size = 30000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Keep the GPU busy with matrix multiplications
    while not stop_threads:
        _ = torch.mm(a, b)
        time.sleep(0.1)  # Adjust the sleep time if necessary to fine-tune the workload

def check_gpu_memory(threshold):
    """检查所有GPU的显存使用情况是否都小于给定阈值"""
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if info.used >= threshold:
            return False
    return True

def monitor_and_run(threshold):
    """监控GPU显存使用情况，并在满足条件时运行占用GPU的程序"""
    while not stop_threads:
        if check_gpu_memory(threshold):
            print("All GPUs have memory usage below the threshold. Starting GPU tasks...")
            # 自动识别可用的GPU设备数量
            device_count = torch.cuda.device_count()
            devices = [torch.device(f'cuda:{i}') for i in range(device_count)]
            print(f"Using devices: {devices}")

            # 在所有可用的GPU设备上运行keep_gpu_busy函数
            threads = []
            for device in devices:
                gpu_thread = threading.Thread(target=keep_gpu_busy, args=(device,))
                gpu_thread.daemon = True
                gpu_thread.start()
                threads.append(gpu_thread)

            # 等待所有线程结束
            for thread in threads:
                thread.join()
            break
        else:
            print("GPU memory usage is above the threshold. Waiting...")
        time.sleep(10)  # 等待一段时间后再次检查

def signal_handler(sig, frame):
    global stop_threads
    print('You pressed Ctrl+C!')
    stop_threads = True
    sys.exit(0)

if __name__ == "__main__":
    # 捕捉Ctrl+C信号
    signal.signal(signal.SIGINT, signal_handler)
    
    # 设置显存使用阈值 (例如 5000MB)
    memory_threshold = 5000 * 1024 * 1024  # 5000 MB 转换为字节

    # 开始监控并运行占用GPU的程序
    monitor_and_run(memory_threshold)

    # 清理pynvml
    pynvml.nvmlShutdown()
