import torch
import random
import argparse
import numpy as np
from copy import deepcopy
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.functional import cross_entropy


def set_seed(seed: int = 0):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True


def dict_to_cuda(d):
    for key in d:
        d[key] = d[key].cuda(non_blocking=True)
    return d


def dict_to_tensor_cuda(d):
    for key in ["input_ids", "attention_mask", "token_type_ids"]:
        if key not in d:
            continue
        d[key] = torch.Tensor(d[key]).cuda(non_blocking=True)
    return d


def dict_to_tensors(d):
    for key in ["input_ids", "attention_mask", "token_type_ids"]:
        d[key] = torch.tensor(d[key])
    return d


def dict_to_device(d, device):
    return {k: v.to(device) for k, v in d.items()}


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def reduce_sum_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def masked_mean(tensor, mask):
    return torch.sum(tensor * mask[:, :, None], dim=[0, 1]) / torch.sum(mask)


def masked_std(tensor, mask):
    mean = masked_mean(tensor, mask)
    return torch.sqrt(torch.sum(tensor ** 2 * mask[:, :, None], dim=[0, 1]) / torch.sum(mask) - mean ** 2)


def parse_checkpoint_name(checkpoint_name):
    items = checkpoint_name.split("-")
    params = dict()
    for item in items:
        key, value = item.split("=")
        params[key] = value
    return params


def make_mask_wo_SEP_CLS(mask):
    mask = deepcopy(mask)
    mask.scatter_(dim=1, index=(mask.sum(dim=1) - 1).reshape(-1, 1), src=torch.zeros_like(mask))
    mask[:, 0] = 0
    return mask


def get_ravel_weights(model):
    ww = []
    for par in model.parameters():
        ww.append(par.detach().cpu().data.numpy().ravel())
    return np.concatenate(ww)


def get_ravel_grad(model):
    ww = []
    for par in model.parameters():
        ww.append(par.grad.detach().cpu().data.numpy().ravel())
    return np.concatenate(ww)


def bert_acc(targets, outputs, mask):
    if mask is None:
        mask = torch.ones(
            (targets.shape[0], targets.shape[1]),
            device=f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0",
            requires_grad=False,
            dtype=torch.int64,
        )
    pred_tokens = outputs.argmax(dim=-1)

    mask = deepcopy(mask)
    mask.scatter_(dim=1, index=(mask.sum(dim=1) - 1).reshape(-1, 1), src=torch.zeros_like(mask))
    mask[:, 0] = 0
    return torch.sum(mask * (targets == pred_tokens)) / torch.sum(mask)


def mse_loss(inputs, targets, mask=None):
    if mask is None:
        mask = torch.ones(
            (targets.shape[0], targets.shape[1]),
            device=f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0",
            requires_grad=False,
            dtype=torch.int64,
        )
    losses = torch.mean(torch.square(inputs - targets), dim=-1)
    losses = losses * mask
    loss = torch.sum(losses) / torch.sum(mask)
    return loss


def recon_loss(inputs, outputs, mask):
    if mask is None:
        mask = torch.ones(
            (inputs.shape[0], inputs.shape[1]),
            device=f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0",
            requires_grad=False,
            dtype=torch.int64,
        )
    losses = cross_entropy(
        input=inputs.reshape(-1, inputs.shape[-1]),
        target=outputs.reshape(-1),
        reduce=False,
    )
    losses = losses * mask.reshape(-1)
    loss = torch.sum(losses) / torch.sum(mask)
    return loss


def get_stat(z, mask):
    if mask is None:
        mask = torch.ones(
            (z.shape[0], z.shape[1]),
            device=f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0",
            requires_grad=False,
            dtype=torch.int64,
        )
    else:
        mask = make_mask_wo_SEP_CLS(mask)
    mean = masked_mean(z, mask)
    std = masked_std(z, mask)
    norm = torch.sum(torch.norm(z, dim=2) * mask) / torch.sum(mask)
    stat_dict = {
        "mean": torch.mean(mean),
        "std": torch.mean(std),
        "norm": norm
    }
    return stat_dict


@torch.no_grad()
def convert_to_simplex(input_embeddings, sigma_0, embeddings):
    # if input_embeddings is None:
    #     input_shape = (*token_ids.shape, -1)
    #     input_embeds = embeddings[token_ids.view(-1)]
    # else:
    input_shape = (*input_embeddings.shape[:-1], -1)
    input_embeds = input_embeddings.view(-1, input_embeddings.shape[-1])

    pairwise_dist = (
        (input_embeds ** 2).sum(-1).unsqueeze(1)
        - 2 * input_embeds @ embeddings.T
        + (embeddings ** 2).sum(-1).unsqueeze(0)
    )

    result = -pairwise_dist / (2 * sigma_0 ** 2)
    return result.view(*input_shape)



def parse():
    parser = argparse.ArgumentParser(description="Dataset arguments")
    parser.add_argument(
        "--dataset_name", type=str, default=None, 
        choices=[
            "rocstories", "wikipedia", "qqp", "xsum", "wiki_auto",
        ],
        required=False,
    )
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--swap_cfg_coef", type=float, default=0.)
    parser.add_argument("--scheduler", type=str, default='sd')
    parser.add_argument("--coef_d", type=float, default=9)
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--sigma_min", type=float, default=0.1)
    parser.add_argument("--sigma_max", type=float, default=20.0)
    parser.add_argument("--emb", type=bool, default=False, help='If set, train model on embeddings')
    parser.add_argument(
        "--emb_statistics_agg_type", type=str, default='features', choices=["features", "total"],
        help='Sets dimensions for aggregation of embeddings values'
    )
    parser.add_argument(
        "--embeddings_path", type=str, default=None, help='Path to file with embedding matrix'
    )
    parser.add_argument(
        "--cluster_diffusion", type=bool, default=False, help='If set, use clusterization-like noising'
    )
    parser.add_argument(
        "--encoder_name", type=str, default='bert-base-cased',
        choices=[
            "bert-base-cased",
            "t5-base",
            "roberta-base",
            "bart-base"
        ])
    parser.add_argument('--project_name', type=str, default='test')
    parser.add_argument('--decoder_name', type=str, default=None, help='Name of decoder file without .pt')
    return parser.parse_args()
