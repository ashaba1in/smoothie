from datasets import concatenate_datasets
from .dataset_tasks import DownstreamTaskDatasetDDP


def get_dataset_iter(config, dataset_name, split):
    dt = DownstreamTaskDatasetDDP(config, dataset_name, split)
    return dt.get_data()


class DatasetDDP:
    def __init__(self, config, split):
        self.split = split
        self.config = config
        self.datasets_config = config.data.datasets
        
        self.datasets = dict()
        for dataset_name in self.datasets_config.datasets_list:
            self.datasets[dataset_name] = get_dataset_iter(self.config, dataset_name, self.split)

    def load_data(self):
        datasets = []
        for dataset_name, dt_iter in self.datasets.items():
            datasets.append(next(dt_iter))    
        dt = concatenate_datasets(datasets)
        return dt

    def get_data(self):
        while True:
            yield self.load_data()
            