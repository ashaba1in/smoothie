import os
import torch
import wandb
import numpy as np
import argparse
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils.util import dict_to_cuda
from model.decoder import Decoder
from data.dataset import get_dataset_iter
from model.encoder import Encoder
from create_config import create_config
from model.enc_normalizer import EncNormalizer
from diffusion_utils.dynamic import DynamicSDE
from utils.util import parse


def reconstruction_loss(target, prediction_scores, mask):
    if mask is None:
        return cross_entropy(
            input=prediction_scores.view(-1, prediction_scores.shape[-1]),
            target=target.view(-1),
        )

    ce_losses = cross_entropy(
        input=prediction_scores.view(-1, prediction_scores.shape[-1]),
        target=target.view(-1),
        reduce=False,
    )
    ce_losses = ce_losses * mask.reshape(-1)
    ce_loss = torch.sum(ce_losses) / torch.sum(mask)
    return ce_loss


def get_datasets(config):
    train_dataset = get_dataset_iter(config, dataset_name=config.decoder.dataset, split="train")
    test_dataset = get_dataset_iter(config, dataset_name=config.decoder.dataset, split="test")
    return train_dataset, test_dataset
    

def get_loaders(train_dataset, valid_dataset, batch_size):
    train_loader = DataLoader(
        next(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=30,
        pin_memory=False,
    )

    valid_loader = DataLoader(
        next(valid_dataset),
        batch_size=batch_size,
        num_workers=30,
        pin_memory=False,
    )

    return train_loader, valid_loader


def loss_step(batch, tokenizer, encoder, decoder, config, eval=False):
    trg = tokenizer(
            batch['text_trg'],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=config.decoder.max_sequence_len,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_token_type_ids=False,
        ).to("cuda:0")
    targets = trg["input_ids"]
    
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        latent = encoder(
            input_ids=targets,
            attention_mask=trg["attention_mask"]
        )
    
    if not eval:
        dynamic = DynamicSDE(config=config)

        if config.decoder.diffusion_forward:
            t = torch.randn(latent.shape[0], device=latent.device) * (config.decoder.T - config.decoder.eps) + config.decoder.eps
            latent = dynamic.marginal(latent, t)["x_t"]
        else:
            t = torch.randn(latent.shape[0], device=latent.device)
            eps = torch.randn_like(latent) * config.decoder.noise_sigma * t
            latent = latent + eps
    with torch.no_grad():
        if not config.emb:
            latent = encoder.module.enc_normalizer.denormalize(latent)

    if config.decoder.is_conditional:
        src = tokenizer(
            batch['text_src'],
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=config.decoder.max_sequence_len,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_token_type_ids=False,
        ).to("cuda:0")
        
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            src_latent = encoder(
                input_ids=src["input_ids"],
                attention_mask=src["attention_mask"]
            )
            if not config.emb:
                src_latent = encoder.module.enc_normalizer.denormalize(src_latent)
        src_mask = src["attention_mask"]
    else:
        src_latent = None
        src_mask = None
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = decoder(latent, cond_x=src_latent, cond_mask=src_mask)

    loss = reconstruction_loss(targets, logits, mask=None)
    
    tokens = logits.argmax(dim=-1)
    acc = torch.mean((targets == tokens) * 1.)

    return loss, acc


def save_checkpoint(model, config):
    os.makedirs(config.training.checkpoints_folder, exist_ok=True)

    model.eval()
    torch.save({"decoder": model.state_dict()}, config.decoder.decoder_path)
    print(f"Save model to: {config.decoder.decoder_path}")


def train(config, encoder, decoder, tokenizer):
    total_number_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Num params: {total_number_params}")

    batch_size = config.decoder.batch_size

    train_dataset, valid_dataset = get_datasets(config=config)

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=config.decoder.lr,
        weight_decay=config.decoder.weight_decay,
        betas=config.decoder.betas,
    )

    step = 0
    for _ in range(config.decoder.epochs): 
        train_loader, valid_loader = get_loaders(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            batch_size=batch_size
        )

        decoder.train()
        for batch in tqdm(train_loader):
            loss, acc = loss_step(
                batch=batch,
                tokenizer=tokenizer,
                encoder=encoder,
                decoder=decoder,
                config=config,
            )
       
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(),
                max_norm=config.decoder.max_norm
            )
            optimizer.step()

            wandb.log({f'train loss': loss.item()}, step=step)
            wandb.log({f'train accuracy': acc.item()}, step=step)

            step += 1

        
        decoder.eval()
        with torch.no_grad():
            total_loss = 0.
            total_acc = 0.
            total_num = 0.
            
            for batch in tqdm(valid_loader):
                loss, acc = loss_step(
                    batch=batch,
                    tokenizer=tokenizer,
                    encoder=encoder,
                    decoder=decoder,
                    config=config,
                    eval=True
                )
                total_loss += loss * len(batch['text_trg'])
                total_acc += acc * len(batch['text_trg'])
                total_num += len(batch['text_trg'])

            total_loss /= total_num
            total_acc /= total_num

            wandb.log({f'valid loss': total_loss.item()}, step=step)
            wandb.log({f'valid accuracy': total_acc.item()}, step=step)
        save_checkpoint(decoder, config)


def main():
    args = parse()

    config = create_config(args)
    if not config.emb:
        enc_normalizer = EncNormalizer(
            enc_mean_path=config.data.enc_gen_mean,
            enc_std_path=config.data.enc_gen_std,
        )
    else:
        enc_normalizer = None
    encoder = Encoder(
        config.model.encoder_link,
        enc_normalizer=enc_normalizer,
        is_change_sp_tokens=True,
        emb=config.emb,
        embeddings_path=config.embeddings_path,
        emb_statistics_agg_type=config.emb_statistics_agg_type,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(encoder.encoder_link)

    encoder = torch.nn.DataParallel(encoder).cuda()
    decoder = Decoder(decoder_config=config.decoder, diffusion_config=config.se_config).train().cuda()

    wandb.init(project=config.project_name, name=config.decoder.name, mode="online")
    train(config, encoder, decoder, tokenizer)


if __name__ == "__main__":
    main()
