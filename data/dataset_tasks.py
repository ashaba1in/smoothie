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
        
    def split_data_across_gpu(self, dt):
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

        return dt.select(indexes)

    def load_data(self):
        path = f"{self.base_path}/{self.split}"
        if self.config.data.group_texts:
            path = f"{self.base_path}_grouped/{self.split}"

        dt = load_from_disk(path)
        if self.dataset_name == 'openwebtext' and self.split == 'test':
            dt = dt.select(range(5000))

        self.dt = self.split_data_across_gpu(dt)

        if self.dataset_name in ['openwebtext', 'wikipedia', 'rocstories']:
            if 'text' in self.dt.column_names:
                self.dt = self.dt.rename_column('text', 'text_trg')
            if 'target' in self.dt.column_names:
                self.dt = self.dt.rename_column('target', 'text_trg')
        else:
            self.dt = self.dt.map(
                partial(
                    batch_preprocessing, split=self.split,
                    dataset_name=self.dataset_name, swap_cfg_coef=self.config.data.swap_cfg_coef,
                    src_lang=self.config.data.src_lang, trg_lang=self.config.data.trg_lang,
                ),
                batched=True,
                load_from_cache_file=True,
                num_proc=30,
                desc="Dataset preprocessing",
                batch_size=1000,
                keep_in_memory=True
            )
        return self.dt
    
    def get_data(self):
        while True:
            yield self.load_data()
            del self.dt
            gc.collect()
