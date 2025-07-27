import gc
import torch
import numpy as np
import torch.distributed as dist
from typing import List
from datasets import Dataset, load_from_disk
from functools import partial

from .preprocessing import batch_preprocessing


class DownstreamTaskDatasetDDP:
    def __init__(self, config, dataset_name, split):
        self.split = split
        self.config = config
        self.dataset_name = dataset_name
        self.base_path = f"{config.data.base_path}/{dataset_name}"

        self.device_id = dist.get_rank() if torch.distributed.is_initialized() else 0
        self.total_device_number = dist.get_world_size() if torch.distributed.is_initialized() else 1
        self.epoch = 0
        
        self.max_context_len = config.data.max_context_len
        self.max_sequence_len = config.data.max_sequence_len
        
    def split_data_across_gpu(self, dt: List[str]):
        self.epoch += 1
        if self.split == "train":
            indexes = np.random.default_rng(seed=self.epoch).permutation(len(dt))
        else:
            indexes = np.arange(len(dt))
        
        start_ind = self.device_id * (len(dt) // self.total_device_number)
        end_ind = (self.device_id + 1) * (len(dt) // self.total_device_number)
        if (self.device_id + 1) == self.total_device_number:
            indexes = indexes[start_ind:]
        else:
            indexes = indexes[start_ind: end_ind]
        
        return Dataset.from_dict(dt[indexes])

    def load_data(self):
        path = f"{self.base_path}/{self.split}"
        dt = load_from_disk(path)
        dt = self.split_data_across_gpu(dt)

        if self.dataset_name == 'openwebtext':
            self.dt = dt.rename_column('text', 'text_trg')
        else:
            self.dt = dt.map(
                partial(
                    batch_preprocessing, split=self.split,
                    dataset_name=self.dataset_name, swap_cfg_coef=self.config.data.swap_cfg_coef
                ),
                batched=True,
                load_from_cache_file=False,
                num_proc=30,
                desc="Dataset preprocessing",
                batch_size=1000,
            )
        return self.dt
    
    def get_data(self):
        while True:
            yield self.load_data()
            del self.dt
            gc.collect()
