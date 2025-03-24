import functools
from collections import defaultdict
import numpy as np


def batch_preprocessing(batch, dataset_name, split, swap_cfg_coef):
    if dataset_name == "qqp":
        if split == "train":
            p_swap = 0.5
        else:
            p_swap = 0.

        new_batch = {
            "text_src": [],
            "text_trg": [],
        }
        for src, trg in zip(batch["question1"], batch["question2"]):
            if np.random.rand() < p_swap:
                src, trg = trg, src
            new_batch["text_src"].append(src)
            new_batch["text_trg"].append(trg)
    elif dataset_name == "xsum":
        new_batch = {
            "text_src": batch["document"],
            "text_trg": batch["summary"],
        }
    elif dataset_name == "wiki_auto":
        new_batch = {
            "text_src": batch["source"],
            "text_trg": batch["target"],
            "references": batch["references"],
        }
    elif dataset_name == "rocstories":
        new_batch = {
            "text_trg": batch["target"],
        }
    else:
        raise Exception(f"Unknown dataset: {dataset_name}")

    # CFG preprocessing
    if split == "train" and swap_cfg_coef:
        length = len(new_batch["text_src"])
        swaps = (np.random.rand(length) < swap_cfg_coef)
        new_batch["text_src"] = ["" if swaps[i] else src for i, src in enumerate(new_batch["text_src"])]
        
    return new_batch
