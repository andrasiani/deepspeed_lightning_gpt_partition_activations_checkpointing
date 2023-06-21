import time

import psutil
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_info
from pynvml import *

def print_gpu_utilization(comment, device_index):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_index)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f" {comment} device idx {device_index} nvml GPU memory occupied: {info.used // 1024 ** 2} MB.")


class CUDACallback(Callback):

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()
        for device_index in range(0, torch.cuda.device_count()):
            print_gpu_utilization("on_train_epoch_start ", device_index=device_index)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx) -> None:
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20

        virt_mem = psutil.virtual_memory()
        virt_mem = round((virt_mem.used / (1024 ** 3)), 2)
        pl_module.log('Peak CUDA Memory (GiB)', max_memory / 1000, prog_bar=True, on_step=True, sync_dist=True)
        pl_module.log(f"Average Virtual memory (GiB)", virt_mem, prog_bar=True, on_step=True, sync_dist=True)

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time
        virt_mem = psutil.virtual_memory()
        virt_mem = round((virt_mem.used / (1024 ** 3)), 2)
        swap = psutil.swap_memory()
        swap = round((swap.used / (1024 ** 3)), 2)

        max_memory = trainer.strategy.reduce(max_memory)
        epoch_time = trainer.strategy.reduce(epoch_time)
        virt_mem = trainer.strategy.reduce(virt_mem)
        swap = trainer.strategy.reduce(swap)

        rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
        rank_zero_info(f"Average Peak CUDA memory {max_memory:.2f} MiB")
        rank_zero_info(f"Average Peak Virtual memory {virt_mem:.2f} GiB")
        rank_zero_info(f"Average Peak Swap memory {swap:.2f} Gib")

        for device_index in range(0, torch.cuda.device_count()):
            print_gpu_utilization("on_train_epoch_end ", device_index=device_index)