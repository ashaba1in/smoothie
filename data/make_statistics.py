import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.dataset import DatasetDDP
from model.encoder import Encoder
from create_config import create_config
from utils.util import parse


def get_loader(config, batch_size):
    train_dataset = next(DatasetDDP(
        config=config,
        split="train",
    ).get_data())  

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=20,
        pin_memory=True,
    )
    return train_loader


def compute_mean_std(
        config,
        encoder,
        tokenizer,
        model_name, 
):
    sum_ = None
    sqr_sum_ = None
    num = 0

    batch_size = 256

    train_loader = get_loader(
        config=config,
        batch_size=batch_size
    )
    T = tqdm(train_loader)

    for i, batch in enumerate(T):
        trg = tokenizer(
            batch['text_trg'],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=config.data.max_sequence_len,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_token_type_ids=False,
        ).to("cuda:0")     

        with torch.no_grad():
            output = encoder(
                    input_ids=trg["input_ids"],
                    attention_mask=trg["attention_mask"]
                )

        mask = 1 - trg["special_tokens_mask"]

        output = output * mask[:, :, None]
        cur_sum = torch.sum(output, dim=[0, 1])
        cur_sqr_sum = torch.sum(output ** 2, dim=[0, 1])
        cur_num = torch.sum(mask).item()

        sum_ = cur_sum if sum_ is None else cur_sum + sum_
        sqr_sum_ = cur_sqr_sum if sqr_sum_ is None else cur_sqr_sum + sqr_sum_
        num += cur_num

        mean = [m.item() for m in sum_[:3] / num]
        std = [s.item() for s in torch.sqrt(sqr_sum_[:3] / num - (sum_[:3] / num) ** 2)]
        mean = [f"{m:0.3f}" for m in mean]
        std = [f"{s:0.3f}" for s in std]
        T.set_description(f"mean: {mean}, std2: {std}")

    mean = sum_ / num
    std = torch.sqrt(sqr_sum_ / num - mean ** 2)

    if torch.isnan(mean).any() or torch.isnan(std).any():
        raise Exception("Nan in statistics")

    folder_path = ''.join([config.data.base_path, '/', config.data.datasets.datasets_list[0], "/statistics"])
    os.makedirs(folder_path, exist_ok=True)
    torch.save(mean, config.data.enc_gen_mean)
    torch.save(std, config.data.enc_gen_std)


if __name__ == "__main__":
    args = parse()
    config = create_config(args)
    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_link)
    
    encoder = Encoder(
        encoder_link=config.model.encoder_link,
        enc_normalizer=None,
        is_change_sp_tokens=False,
        emb=args.emb,
        embeddings_path=config.embeddings_path,
        emb_statistics_agg_type=config.emb_statistics_agg_type,
    ).eval()
    encoder = torch.nn.DataParallel(encoder).cuda()

    compute_mean_std(
        config,
        encoder,
        tokenizer, 
        model_name=config.model.encoder_name_hash,
    )
