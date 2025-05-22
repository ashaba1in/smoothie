import os
import torch
from argparse import Namespace
from data.dataset import get_dataset_iter
from torch.utils.data import DataLoader
from create_config import create_config
from transformers import AutoTokenizer
from model.score_estimator import ScoreEstimatorEMB
from model.encoder import Encoder
from model.decoder import Decoder
from utils.ema_model import ExponentialMovingAverage
from diffusion_holder import DiffusionRunner
import matplotlib.pyplot as plt
from utils.util import convert_to_simplex, parse, set_seed
from tqdm.auto import tqdm
import torch.distributed as dist

if __name__ == '__main__':
    args = parse()

    config = create_config(args)

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    config.local_rank = rank
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()
    if dist.get_rank() == 0:
        print(config)
    seed = config.seed + dist.get_rank()
    set_seed(seed)

    diffusion = DiffusionRunner(config, eval=True)
