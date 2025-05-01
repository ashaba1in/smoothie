import os
import wandb
import random
import numpy as np
import torch
import torch.nn.functional as F
import ml_collections
import torch.distributed as dist
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from ml_collections import ConfigDict
from typing import Optional, Union, Dict, Tuple
from tqdm.auto import trange
from tqdm import tqdm
from transformers import AutoTokenizer
from functools import partial
from timm.scheduler.cosine_lr import CosineLRScheduler
from collections import defaultdict
from torch.cuda.amp import GradScaler
import json
from copy import deepcopy
import heapq

from diffusion_utils.dynamic import DynamicSDE
from diffusion_utils.solvers import create_solver

from utils.ema_model import ExponentialMovingAverage
from utils.util import mse_loss, get_stat, reduce_tensor, set_seed, convert_to_simplex
from data.dataset import DatasetDDP, get_dataset_iter
from data.util import BatchEncoding

from model.score_estimator import ScoreEstimatorEMB
from model.encoder import Encoder
from model.enc_normalizer import EncNormalizer
from model.decoder import Decoder

from estimation_utils.util import gather_texts
from estimation_utils.metrics import compute_metric


class DiffusionRunner:
    def __init__(
            self,
            config: ConfigDict,
            eval: bool = False,
            run_wandb: bool = True
    ):
        self.config = config

        # Diffusion Encoder
        gen_cfg = config.model.encoder_link
        self.tokenizer = AutoTokenizer.from_pretrained(gen_cfg)
        if not config.emb:
            self.gen_enc_normalizer = EncNormalizer(
                enc_mean_path=self.config.data.enc_gen_mean,
                enc_std_path=self.config.data.enc_gen_std,
            )
        else:
            self.gen_enc_normalizer = None
        
        self.encoder = Encoder(
            gen_cfg,
            enc_normalizer=self.gen_enc_normalizer,
            is_change_sp_tokens=True,
            emb=config.emb,
            embeddings_path=config.embeddings_path,
            emb_statistics_agg_type=config.emb_statistics_agg_type,
            random_init=config.random_init_embeddings
        ).eval().cuda()

        # Decoder
        # self.decoder = Decoder(
        #     decoder_config=config.decoder,
        #     diffusion_config=config.se_config
        # )
        # if not config.training.train_embeddings:
        #     self.restore_decoder()
        #     self.decoder.eval()

        # self.decoder = self.decoder.cuda()

        # Score estimator
        self.se_config = deepcopy(config.se_config)
        self.se_config.use_self_cond = config.use_self_cond
        self.score_estimator = ScoreEstimatorEMB(
            config=self.se_config
        ).cuda()

        self.ddp_score_estimator = self.score_estimator
        if self.config.ddp:
            self.ddp_score_estimator = torch.nn.parallel.DistributedDataParallel(
                self.score_estimator,
                device_ids=[config.local_rank],
                broadcast_buffers=False,
            )

        # Number of parameters
        self.config.params_number = ml_collections.ConfigDict()
        self.config.params_number.score_estimator = sum(p.numel() for p in self.score_estimator.parameters() if p.requires_grad)
        # self.config.params_number.decoder = sum(p.numel() for p in self.decoder.parameters())
        self.config.params_number.generative_encoder = sum(p.numel() for p in self.encoder.parameters())

        self.device = next(self.score_estimator.parameters()).device

        # Dynamic
        self.dynamic = DynamicSDE(config=config)
        self.diff_eq_solver = create_solver(config)(
            dynamic=self.dynamic,
            score_fn=partial(self.calc_score, model=self.ddp_score_estimator),
            ode_sampling=config.training.ode_sampling
        )

        self.train_datasets_iter = DatasetDDP(
            split="train",
            config=config,
        ).get_data()
        self.train_dataset = None
        
        try:
            self.valid_dataset = next(DatasetDDP(split="validation", config=config).get_data())
        except Exception:
            self.valid_dataset = next(DatasetDDP(split="test", config=config).get_data())

        # Checkpoint loading
        self.ema = ExponentialMovingAverage(self.score_estimator.parameters(), self.config.model.ema_rate)

        # Checkpoint utils
        self.all_checkpoints = []
        self.last_checkpoint = None
        self.tracked_test_metric = dict()  # step -- metric value

        if self.config.ddp and dist.get_rank() == 0 and run_wandb:
            wandb.init(
                project=self.config.project_name,
                name=self.config.training.checkpoints_prefix,
                config=dict(self.config),
            )

        if eval:
            self.restore_parameters(self.device)
            self.score_estimator.eval()
            self.estimate("test")
        else:
            self.set_optimizer()
            self.set_scheduler()
            self.set_grad_scaler()
            self.step = 0
            
            if self.load_checkpoint():
                for group in self.optimizer.param_groups:
                    group['weight_decay'] = self.config.optim.weight_decay
                    group['lr'] = self.config.optim.lr

                self.scheduler.base_values = [self.config.optim.lr for _ in self.optimizer.param_groups]
                self.scheduler.lr_min = self.config.optim.lr

                if self.config.is_conditional:
                    self.estimate("validation")
                self.estimate("test")
                self.validate()

    def restore_parameters(self, device: Optional[torch.device] = None) -> None:
        prefix_folder = os.path.join(self.config.training.checkpoints_folder, self.config.training.checkpoints_prefix)

        checkpoint_names = list(os.listdir(prefix_folder))
        checkpoint_names = [str(t).replace(".pth", "") for t in checkpoint_names]
        checkpoint_names = [int(t) for t in checkpoint_names if t.isdigit()]

        if not checkpoint_names:
            return False

        name = self.config.training.checkpoint_name
        if not name:
            name = max(checkpoint_names)
        checkpoint_name = f"{prefix_folder}/{name}.pth"
        load = torch.load(checkpoint_name)
        self.step = load["step"]
        self.ema.load_state_dict(load["ema"])
        self.switch_to_ema()
        
    def save_checkpoint(self, last: bool = False) -> None:
        if not dist.get_rank() == 0:
            return

        if not os.path.exists(self.config.training.checkpoints_folder):
            os.makedirs(self.config.training.checkpoints_folder)
            
        prefix_folder = os.path.join(self.config.training.checkpoints_folder, self.config.training.checkpoints_prefix)
        if not os.path.exists(prefix_folder):
            os.makedirs(prefix_folder)

        if last:
            prefix = 'last'
        else:
            prefix = str(self.step)

        save_path = os.path.join(prefix_folder, prefix + ".pth")

        if self.config.higher_better:
            item = (self.tracked_test_metric[self.step], save_path)
        else:
            item = (-self.tracked_test_metric[self.step], save_path)

        if last:
            self.__save_checkpoint(save_path)
            heapq.heappush(self.all_checkpoints, item)
            return

        if self.config.save_top_k is not None and len(self.all_checkpoints) >= self.config.save_top_k:
            heap_smallest = self.all_checkpoints[0]
            if heap_smallest[0] < item[0]:
                self.__remove_checkpoint(heap_smallest[1])
                heapq.heappop(self.all_checkpoints)
                heapq.heappush(self.all_checkpoints, item)
        else:
            heapq.heappush(self.all_checkpoints, item)

        self.__save_checkpoint(save_path)
        if self.last_checkpoint is not None and self.last_checkpoint not in self.all_checkpoints:
            self.__remove_checkpoint(self.last_checkpoint[1])

        self.last_checkpoint = item

    def __remove_checkpoint(self, save_path):
        os.remove(save_path)
    
    def __save_checkpoint(self, save_path):
        torch.save(
            {
                "model": self.score_estimator.state_dict(),
                "ema": self.ema.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.grad_scaler.state_dict(),
                "step": self.step,
                "encoder": self.encoder.state_dict(),
                # "decoder": self.decoder.state_dict(),
            },
            save_path
        )
        print(f"Save model to: {save_path}")
        
    def load_checkpoint(self) -> int:
        prefix_folder = os.path.join(self.config.training.checkpoints_folder, self.config.training.checkpoints_prefix)

        if not os.path.exists(prefix_folder):
            return False

        checkpoint_names = list(os.listdir(prefix_folder))
        checkpoint_names = [str(t).replace(".pth", "") for t in checkpoint_names]
        checkpoint_names = [int(t) for t in checkpoint_names if t.isdigit()]

        if not checkpoint_names:
            return False
            
        name = self.config.training.checkpoint_name
        if not name:
            name = max(checkpoint_names)
        checkpoint_name = f"{prefix_folder}/{name}.pth"

        load = torch.load(checkpoint_name, map_location="cpu")

        self.ema.load_state_dict(load["ema"])
        self.ema.cuda()
        self.score_estimator.load_state_dict(load["model"])
        self.optimizer.load_state_dict(load["optimizer"])
        self.scheduler.load_state_dict(load["scheduler"])
        self.grad_scaler.load_state_dict(load["scaler"])
        
        self.step = load["step"]
        if dist.get_rank() == 0:
            print(f"Checkpoint is loaded {checkpoint_name}")
        return True

    # def restore_decoder(self):
    #     decoder_path = self.config.decoder.decoder_path
    #     self.decoder.load_state_dict(torch.load(decoder_path)["decoder"])

    def switch_to_ema(self) -> None:
        ema = self.ema
        score_model = self.score_estimator
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())

    def switch_back_from_ema(self) -> None:
        ema = self.ema
        score_model = self.score_estimator
        ema.restore(score_model.parameters())

    def set_optimizer(self) -> None:
        parameters = self.score_estimator.parameters()
        if self.config.training.train_embeddings:
            parameters = list(parameters) + [self.encoder.embeddings]  # + list(self.decoder.parameters())
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
            betas=(self.config.optim.beta_1, self.config.optim.beta_2),
            eps=self.config.optim.eps,
        )
        self.warmup = self.config.optim.linear_warmup
        self.grad_clip_norm = self.config.optim.grad_clip_norm
        self.optimizer = optimizer

    def set_scheduler(self) -> None:
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.config.training.training_iters,
            lr_min=self.config.optim.min_lr,
            warmup_lr_init=self.config.optim.warmup_lr,
            warmup_t=self.config.optim.linear_warmup,
            cycle_limit=1,
            t_in_epochs=False,
        )

    def set_grad_scaler(self) -> None:
        self.grad_scaler = GradScaler()

    def collate_fn(self, batch):
        texts_trg = [t["text_trg"] for t in batch]
        tok_trg = self.tokenizer(
            texts_trg,
            add_special_tokens=self.config.data.add_special_tokens,
            padding='max_length',
            truncation=True,
            max_length=self.config.data.max_sequence_len,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        if self.config.is_conditional:
            texts_src = [t["text_src"] for t in batch]
            tok_src = self.tokenizer(
                texts_src,
                add_special_tokens=self.config.data.add_special_tokens,
                padding=True,
                truncation=True,
                max_length=self.config.data.max_context_len,
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False,
            )

            new_batch = {
                "text_src": texts_src,
                "input_ids_src": tok_src["input_ids"],
                "attention_mask_src": tok_src["attention_mask"],
                "text_trg": texts_trg,
                "input_ids_trg": tok_trg["input_ids"],
                "attention_mask_trg": tok_trg["attention_mask"],
            }
            if "references" in batch[0]:
                new_batch["text_references"] = [t["references"] for t in batch]

            new_batch = BatchEncoding(new_batch)
        else:
            new_batch = BatchEncoding({
                "text_trg": texts_trg,
                "input_ids_trg": tok_trg["input_ids"],
                "attention_mask_trg": tok_trg["attention_mask"],
            })
        return new_batch

    def set_train_data_generator(self) -> None:
        del self.train_dataset
        self.train_dataset = next(self.train_datasets_iter)
        print("Dataset length:", len(self.train_dataset))

        self.train_loader = DataLoader(
            self.train_dataset,
            num_workers=30,
            batch_size=self.config.training.batch_size_per_gpu,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def set_valid_data_generator(self) -> None:
        self.valid_loader = DataLoader(
            self.valid_dataset,
            num_workers=20,
            batch_size=self.config.validation.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def log_metric(self, metric_name: str, loader_name: str, value: Union[float, torch.Tensor, wandb.Image]):
        if dist.is_initialized() and dist.get_rank() == 0:
            wandb.log({f'{metric_name}/{loader_name}': value}, step=self.step)

    def optimizer_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        # loss.backward()
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        
        grad_norm = torch.sqrt(
            sum([torch.sum(t.grad ** 2) for t in self.score_estimator.parameters() if t.requires_grad])
        )

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.score_estimator.parameters(),
                max_norm=self.grad_clip_norm
            )

        self.log_metric('lr', 'train', self.optimizer.param_groups[0]['lr'])
        # self.optimizer.step()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        # My custom strategy
        scale = self.grad_scaler._scale.item()
        max_scale = 2 ** 30
        min_scale = 1
        scale = np.clip(scale, min_scale, max_scale)
        self.grad_scaler.update(new_scale=scale)

        self.ema.update(self.score_estimator.parameters())
        self.scheduler.step_update(self.step)
        return grad_norm

    def sample_time(self, batch_size: int, eps: float = 1e-5):
        return torch.cuda.FloatTensor(batch_size).uniform_() * (self.dynamic.T - eps) + eps
    
    def train(self) -> None:
        self.set_valid_data_generator()

        self.train_range = trange(self.step + 1, self.config.training.training_iters + 1)
        self.train_range_iter = iter(self.train_range)

        while True:
            self.set_train_data_generator()
            self.ddp_score_estimator.train()
            self.train_epoch()

            if self.step >= self.config.training.training_iters:
                break

        self.score_estimator.eval()
        self.save_checkpoint(last=True)
        self.switch_to_ema()

    def train_epoch(self):
        for _, batch in enumerate(self.train_loader):
            if self.step >= self.config.training.training_iters:
                return
            _ = next(self.train_range_iter)

            loss_dict, stat_dict = self.train_step(batch)

            if self.step % self.config.training.eval_freq == 0:
                if self.config.is_conditional: 
                    self.estimate("validation")
                self.estimate("test")
                self.validate()
            
            if self.step % self.config.training.checkpoint_freq == 0:
                self.save_checkpoint()

            if self.step % self.config.training.accum_batch_steps == 0:
                self.train_range.set_description(
                    f"loss_x_0: {loss_dict['loss_x_0'].item():0.4f}, "
                    f"grad_norm: {stat_dict['grad_norm'].item():0.4f}, "
                )

    def train_step(self, batch):
        self.step += 1

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                batch = batch.to(f"cuda:{dist.get_rank()}")

                if self.config.is_conditional:
                    src_x = self.encoder(**{
                        "input_ids": batch["input_ids_src"],
                        "attention_mask": batch["attention_mask_src"]
                    })
                else:
                    src_x = None

                trg_x = self.encoder(**{
                    "input_ids": batch["input_ids_trg"], 
                    "attention_mask": batch["attention_mask_trg"]
                })

        loss_dict, stat_dict = self.calc_loss(clean_x=trg_x, cond_x=src_x, batch=batch)

        if self.step % self.config.training.accum_batch_steps == 0:
            stat_dict["grad_norm"] = self.optimizer_step(loss_dict['total_loss'])
            stat_dict["scale_factor"] = torch.Tensor([self.grad_scaler._scale])

        if self.step % 10 == 0:
            stat_dict["weight_norm"] = torch.sqrt(
                sum([torch.sum(t.data ** 2) for t in self.score_estimator.parameters()]))

            for k, v in loss_dict.items():
                self.log_metric(k, 'train', v.item())

            for k, v in stat_dict.items():
                self.log_metric("statistics", k, v.item())

        return loss_dict, stat_dict

    @torch.no_grad()
    def validate(self) -> None:
        self.set_valid_data_generator()
        prev_mode = self.ddp_score_estimator.training

        self.ddp_score_estimator.eval()
        self.switch_to_ema()

        valid_loss: Dict[str, torch.Tensor] = dict()
        valid_count = torch.Tensor([0.0])

        for batch in self.valid_loader:
            batch = batch.to(f"cuda:{dist.get_rank()}")
            if self.config.is_conditional:
                src_x = self.encoder(**{
                    "input_ids": batch["input_ids_src"],
                    "attention_mask": batch["attention_mask_src"]
                })
            else:
                src_x = None

            trg_x = self.encoder(**{
                "input_ids": batch["input_ids_trg"], 
                "attention_mask": batch["attention_mask_trg"]
            })

            loss_dict, _ = self.calc_loss(clean_x=trg_x, cond_x=src_x, batch=batch)
            for k, v in loss_dict.items():
                if k in valid_loss:
                    valid_loss[k] += v.item() * trg_x.size(0)
                else:
                    valid_loss[k] = torch.Tensor([v.item() * trg_x.size(0)])
            valid_count += trg_x.size(0)

        valid_count = reduce_tensor(valid_count.cuda())
        for k, v in valid_loss.items():
            valid_loss[k] = reduce_tensor(valid_loss[k].cuda())

        for k, v in valid_loss.items():
            valid_loss[k] = v / valid_count
        for k, v in valid_loss.items():
            self.log_metric(k, 'valid_loader', v)

        # --------- Eval per timestep metrics ------------

        clean_x = self.encoder(**{
            "input_ids": batch["input_ids_trg"],
            "attention_mask": batch["attention_mask_trg"]
        })
        target = clean_x.clone()
        if self.config.cluster_diffusion:
            clean_x = convert_to_simplex(
                input_embeddings=clean_x,
                sigma_0=self.config.dynamic.sigma_min,
                embeddings=self.encoder.embeddings,
            )
        noise = torch.randn_like(clean_x)
        x_0_self_cond = torch.zeros_like(target)
        per_t_losses = []
        mean_id_probs_t = []
        accuracies = []
        ts = torch.linspace(0, self.dynamic.T, self.dynamic.N)
        for t in ts:
            timesteps = torch.empty(size=(clean_x.shape[0],), device=clean_x.device).fill_(t)
            marg_forward = self.dynamic.marginal(clean_x, timesteps, noise=noise)
            x_t = marg_forward['x_t']

            if self.config.cluster_diffusion:
                model_input = torch.softmax(x_t, dim=-1) @ self.encoder.embeddings
            else:
                model_input = x_t
            # model prediction
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                x_0 = self.ddp_score_estimator(
                    x_t=model_input, time_t=timesteps, cond=src_x,
                    cond_mask=batch.get("attention_mask_src"),
                    x_0_self_cond=x_0_self_cond
                )

            # MSE losses
            loss = mse_loss(target, x_0, mask=None)

            per_t_losses.append(loss.item())
            # pred_tokens = self.decoder(x_0, cond_x=src_x, cond_mask=batch.get("attention_mask_src")).argmax(-1)
            pred_tokens = self.decode(x_0)
            mask = batch['attention_mask_trg'].bool()
            accuracies.append((pred_tokens[mask] == batch["input_ids_trg"][mask]).float().mean().item())
            if self.config.cluster_diffusion:
                probs_t = torch.softmax(x_t, dim=-1)
                id_probs_t = probs_t.gather(2, batch["input_ids_trg"].unsqueeze(-1))
                mean_id_probs_t.append(id_probs_t.mean().item())

        dir_path = f'plots/{self.config.training.checkpoints_prefix}'
        os.makedirs(dir_path, exist_ok=True)
        fig = plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(ts, per_t_losses)
        plt.xlabel('timestep')
        plt.title('Reconstruction loss')
        plt.grid()

        plt.subplot(1, 3, 2)
        plt.plot(ts, accuracies)
        plt.xlabel('timestep')
        plt.title('Accuracies')
        plt.grid()

        if self.config.cluster_diffusion:
            plt.subplot(1, 3, 3)
            plt.plot(ts, mean_id_probs_t)
            plt.xlabel('timestep')
            plt.title('Mean id_prob$_t$')
            plt.yscale('log')
            plt.axhline(1 / len(self.tokenizer), c='r')
            plt.grid()

        fig.savefig(f'{dir_path}/per_t_loss_{self.step}.png', dpi=fig.dpi)

        self.switch_back_from_ema()
        self.ddp_score_estimator.train(prev_mode)

    def predict_x_0_unconditional(
        self,
        model,
        x_t, t,
        attention_mask=None,
        x_0_self_cond=None
    ) -> torch.Tensor:
        texts_src = ["" for _ in range(x_t.shape[0])]
        tok_src = self.tokenizer(
            texts_src,
            add_special_tokens=self.config.data.add_special_tokens,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_context_len,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
        ).to(f"cuda:{dist.get_rank()}")
        src_x = self.encoder(
            input_ids=tok_src["input_ids"],
            attention_mask=tok_src["attention_mask"]
        )

        x_0 = model(
            x_t=x_t, 
            time_t=t, 
            cond=src_x,
            attention_mask=attention_mask, 
            cond_mask=tok_src["attention_mask"],
            x_0_self_cond=x_0_self_cond
        )
        return x_0

    def calc_score(
            self,
            model,
            x_t, t,
            cond=None,
            attention_mask=None,
            cond_mask=None,
            x_0_self_cond=None
    ) -> Dict[str, torch.Tensor]:
        """
        x_0 - prediction x_0(x_t, t)
        eps = (x_t - sqrt(alpha_t) * x_0) / std
        score = (-x_t + sqrt(alpha_t) * x_0) / std**2
        """
        params = self.dynamic.marginal_params(t)
        if self.config.cluster_diffusion:
            # x_t is [bs, seq_len, V]
            model_input = torch.softmax(x_t, dim=-1) @ self.encoder.embeddings
        else:
            # x_t is [bs, seq_len, hidden_size]
            model_input = x_t

        model_prediction = model(
            x_t=model_input, time_t=t, cond=cond,
            attention_mask=attention_mask, cond_mask=cond_mask,
            x_0_self_cond=x_0_self_cond
        )

        if not model.training and self.config.validation.cfg_coef and self.config.is_conditional:
            null_pred = self.predict_x_0_unconditional(
                model, x_t=model_input, t=t, attention_mask=attention_mask, x_0_self_cond=x_0_self_cond
            )
            model_prediction = model_prediction + self.config.validation.cfg_coef * (model_prediction - null_pred)

        if self.config.cluster_diffusion:
            x_0 = convert_to_simplex(
                input_embeddings=model_prediction,
                sigma_0=self.config.dynamic.sigma_min,
                embeddings=self.encoder.embeddings,
            )
        else:
            x_0 = model_prediction

        eps_theta = (x_t - params["mu"] * x_0) / params["std"]
        score = -eps_theta / params["std"]
        return {
            "score": score,
            "x_0": x_0,
            "eps_theta": eps_theta,
            "latent_pred": model_prediction,
        }

    def calc_loss(
            self,
            clean_x,
            cond_x,
            batch=None,
            eps: float = 1e-5,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        mask = None

        target = clean_x.clone()
        if self.config.cluster_diffusion:
            clean_x = convert_to_simplex(
                input_embeddings=clean_x.detach(),
                sigma_0=self.config.dynamic.sigma_min,
                embeddings=self.encoder.embeddings,
            )

        # Noising
        batch_size = clean_x.size(0)

        t = self.sample_time(batch_size, eps=eps)
        x_t = self.dynamic.marginal(clean_x, t)['x_t']

        loss = 0
        loss_dict = dict()
        x_0_self_cond = torch.zeros_like(target, dtype=target.dtype)
        apply_self_cond = self.config.use_self_cond and random.random() > 0.5
        if self.config.training.step_unrolled or apply_self_cond:
            if self.config.cluster_diffusion:
                model_input = torch.softmax(x_t, dim=-1) @ self.encoder.embeddings
            else:
                model_input = x_t

            if self.config.training.step_unrolled:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    x_0 = self.ddp_score_estimator(
                        x_t=model_input, time_t=t, cond=cond_x,
                        attention_mask=mask,
                        cond_mask=batch.get("attention_mask_src"),
                        x_0_self_cond=x_0_self_cond
                    )
                first_loss_x_0 = mse_loss(target, x_0, mask)
                loss = loss + first_loss_x_0
                loss_dict['first_loss_x_0'] = first_loss_x_0

                new_clean_x = x_0.detach()
                if self.config.cluster_diffusion:
                    new_clean_x = convert_to_simplex(
                        input_embeddings=new_clean_x,
                        sigma_0=self.config.dynamic.sigma_min,
                        embeddings=self.encoder.embeddings,
                    )
                x_t = self.dynamic.marginal(new_clean_x, t)['x_t']
            elif apply_self_cond:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    with torch.no_grad():
                        x_0_self_cond = self.ddp_score_estimator(
                            x_t=model_input, time_t=t, cond=cond_x,
                            attention_mask=mask,
                            cond_mask=batch.get("attention_mask_src"),
                            x_0_self_cond=x_0_self_cond
                        ).detach()

        if self.config.cluster_diffusion:
            model_input = torch.softmax(x_t, dim=-1) @ self.encoder.embeddings
        else:
            model_input = x_t
        # model prediction
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            x_0 = self.ddp_score_estimator(
                x_t=model_input, time_t=t, cond=cond_x,
                attention_mask=mask, 
                cond_mask=batch.get("attention_mask_src"),
                x_0_self_cond=x_0_self_cond
            )

        # MSE losses
        loss_x_0 = mse_loss(target, x_0, mask)
        loss = loss + loss_x_0

        loss_dict['loss_x_0'] = loss_x_0

        if self.config.training.train_embeddings:
            # x_T loss to regularize embeddings
            t = t.fill_(self.dynamic.T)
            mu_T = self.dynamic.marginal_params(t)['mu']
            x_T = mu_T * clean_x
            if self.config.cluster_diffusion:
                model_input_T = torch.softmax(x_T, dim=-1) @ self.encoder.embeddings
                tT_loss = mse_loss(model_input_T, self.encoder.embeddings.mean(0, keepdim=True).detach())
            else:
                tT_loss = torch.mean(x_T**2)
            loss_dict['tT_loss'] = tT_loss

            # Decoder loss to learn decoder and prevent embeddings from collapsing
            # 2 ways of computing decoder loss
            # 1) Pros: decoder is tuned for model's predictions. Cons: predictions might be not good, because t is big.
            # if self.config.cluster_diffusion:
            #     input_embeddings = torch.softmax(clean_x, dim=-1) @ self.encoder.embeddings
            # else:
            #     input_embeddings = clean_x
            # decoder_input = input_embeddings + (x_0 - input_embeddings).detach()
            # 2) Pros: input is not that noisy. Cons: decoder is not tuned to model as good.
            t = torch.randn_like(t) * self.config.decoder.T
            decoder_x_t = self.dynamic.marginal(clean_x, t)["x_t"]
            if self.config.cluster_diffusion:
                decoder_input = torch.softmax(decoder_x_t, dim=-1) @ self.encoder.embeddings
            else:
                decoder_input = decoder_x_t
            # logits = self.decoder(decoder_input, cond_x=cond_x, cond_mask=batch.get("attention_mask_src"))
            logits = self.decode(decoder_input)  # self.decode returns TOKENS!
            nll_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), batch['input_ids_trg'].view(-1))

            acc = (logits.argmax(dim=-1) == batch['input_ids_trg']).float().mean()
            loss_dict['accuracy'] = acc
            loss_dict['nll_loss'] = nll_loss

            loss = loss + self.config.training.x_T_coef * tT_loss + self.config.training.nll_coef * nll_loss

        loss_dict['total_loss'] = loss

        with torch.no_grad():
            stat_dict = {}
            if self.config.cluster_diffusion:
                D_0_dict = get_stat(clean_x, mask)
                for key in D_0_dict:
                    stat_dict[f"D_0_{key}"] = D_0_dict[key]

            target_dict = get_stat(target, mask)
            for key in target_dict:
                stat_dict[f"clean_x_{key}"] = target_dict[key]
    
            x_0_dict = get_stat(x_0.detach(), mask)
            for key in x_0_dict:
                stat_dict[f"x_0_{key}"] = x_0_dict[key]
    
            mask = batch["attention_mask_trg"]
            target_dict_SPT = get_stat(target, mask)
            for key in target_dict_SPT:
                stat_dict[f"clean_x_woSPT_{key}"] = target_dict_SPT[key]
    
            x_0_dict_SPT = get_stat(x_0, mask)
            for key in x_0_dict_SPT:
                stat_dict[f"x_0_woSPT_{key}"] = x_0_dict_SPT[key]

        return loss_dict, stat_dict

    @torch.no_grad()
    def generate_text_conditional(self, dataset_name: str, split: str):
        dt = next(get_dataset_iter(self.config, dataset_name, split=split))
        loader = DataLoader(
            dt,
            num_workers=20,
            batch_size=self.config.validation.batch_size,
            collate_fn=self.collate_fn,
        )

        result_dict = {
            "GEN": [],
            "TRG": []
        }
        if self.config.is_conditional:
            result_dict["SRC"] = []

        for batch in loader:
            if dist.is_initialized():
                batch = batch.to(f"cuda:{dist.get_rank()}")
            else:
                batch = batch.to(f"cuda:0")
            if self.config.is_conditional:
                src_x = self.encoder(**{
                    "input_ids": batch["input_ids_src"],
                    "attention_mask": batch["attention_mask_src"]
                })
            else:
                src_x = None

            gen_text = self.generate_text_batch(
                batch_size=len(batch["text_trg"]),
                cond_x=src_x,
                attention_mask=None,
                cond_mask=batch.get("attention_mask_src")
            )[0]

            # if dataset_name not in self.config.data.datasets.downstream_tasks:
            result_dict["TRG"] += self.tokenizer.batch_decode(batch["input_ids_trg"], skip_special_tokens=True)
            # else:
            #     if "text_references" in batch:
            #         result_dict["TRG"] += batch["text_references"]
            #     else:
            #         result_dict["TRG"] += batch["text_trg"]
                
            result_dict["GEN"] += gen_text
            if self.config.is_conditional:
                result_dict["SRC"] += batch["text_src"]

            if len(result_dict["TRG"]) >= (self.config.validation.num_gen_texts // dist.get_world_size()):
                break

        return result_dict

    @torch.no_grad()
    def generate_text_batch(self, batch_size, cond_x=None, attention_mask=None, cond_mask=None, eps_t=0.0, x=None):
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()

        pred_embeddings = self.pred_embeddings(
            batch_size=batch_size,
            attention_mask=attention_mask,
            cond_x=cond_x,
            cond_mask=cond_mask,
            eps_t=eps_t,
            x=x
        )

        # output = self.pred_logits(pred_embeddings, cond_x=cond_x, cond_mask=cond_mask)
        # tokens = output.argmax(dim=-1)
        tokens = self.decode(pred_embeddings)

        end_tokens = []
        if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
            end_tokens.append(self.tokenizer.vocab[self.tokenizer.eos_token])
        if hasattr(self.tokenizer, 'sep_token') and self.tokenizer.sep_token is not None:
            end_tokens.append(self.tokenizer.vocab[self.tokenizer.sep_token])

        tokens = tokens.detach().cpu().tolist()
        tokens_list = []
        for seq in tokens:
            id = 0
            while id < len(seq) and seq[id] not in end_tokens:
                id += 1
            tokens_list.append(seq[0:id])

        text = self.tokenizer.batch_decode(tokens_list, skip_special_tokens=True)
        return text, pred_embeddings

    # @torch.no_grad()
    # def pred_logits(self, pred_embeddings, cond_x=None, cond_mask=None):
    #     if not self.config.emb:
    #         pred_embeddings = self.gen_enc_normalizer.denormalize(pred_embeddings)
    #         if self.config.decoder.is_conditional and cond_x is not None:
    #             cond_x = self.gen_enc_normalizer.denormalize(cond_x)
    #
    #     output = self.decoder(pred_embeddings, cond_x=cond_x, cond_mask=cond_mask)
    #     return output

    def decode(self, pred_embeddings):
        logits = convert_to_simplex(
            input_embeddings=pred_embeddings,
            sigma_0=self.config.dynamic.sigma_min,
            embeddings=self.encoder.embeddings,
        )
        tokens = logits.argmax(-1)
        return tokens

    @torch.no_grad()
    def pred_embeddings(
            self,
            batch_size,
            cond_x=None,
            cond_mask=None,
            attention_mask=None,
            eps_t=0.0,
            x=None
    ) -> torch.Tensor:
        self.score_estimator.eval()

        if self.config.cluster_diffusion:
            shape = (
                batch_size,
                self.config.data.max_sequence_len,
                self.encoder.encoder.config.vocab_size
            )
        else:
            shape = (
                batch_size,
                self.config.data.max_sequence_len,
                self.encoder.encoder.config.hidden_size
            )
        with torch.no_grad():
            if x is None:
                x = self.dynamic.prior_sampling(shape).to(self.device)
            x_0_self_cond = torch.zeros(
                *shape[:-1], self.encoder.encoder.config.hidden_size, dtype=x.dtype, device=x.device
            )

            timesteps = torch.linspace(self.dynamic.T, eps_t, self.dynamic.N + 1, device=self.device)
            for idx in tqdm(range(self.dynamic.N)):
                t = timesteps[idx]
                next_t = timesteps[idx + 1]

                input_t = t * torch.ones(shape[0], device=self.device)
                next_input_t = next_t * torch.ones(shape[0], device=self.device)

                output = self.diff_eq_solver.step(
                    x_t=x, t=input_t, next_t=next_input_t,
                    cond=cond_x,
                    cond_mask=cond_mask,
                    attention_mask=attention_mask,
                    x_0_self_cond=x_0_self_cond,
                )

                x = output["x"]
                x_0_self_cond = output["latent_pred"]

            pred_embeddings = output["latent_pred"]

        return pred_embeddings

    @torch.no_grad()
    def estimate(self, split: str):
        self.score_estimator.eval()
        self.ddp_score_estimator.eval()
        self.switch_to_ema()
        
        result_dict = dict()
        
        # Generation
        for dataset_name in self.config.data.datasets.datasets_list:
            if dist.is_initialized():
                seed = self.config.seed + self.step + dist.get_rank()
            else:
                seed = self.config.seed + self.step
            set_seed(seed)
            result_dict[dataset_name] = self.generate_text_conditional(dataset_name, split=split)

        # Gathering
        if dist.is_initialized():
            for dataset_name in result_dict:
                for key in result_dict[dataset_name]:
                    result_dict[dataset_name][key] = gather_texts(result_dict[dataset_name][key])
                if dataset_name not in self.config.data.datasets.downstream_tasks:
                    for key in result_dict[dataset_name]:
                        result_dict[dataset_name][key] = result_dict[dataset_name][key][:self.config.validation.num_gen_texts]

        # Logging
        result_list = dict()

        for dataset_name in result_dict:
            keys = list(result_dict[dataset_name].keys())
            result_list[dataset_name] = []
            for ind in range(len(result_dict[dataset_name][keys[0]])):
                result_list[dataset_name].append(
                    {key: result_dict[dataset_name][key][ind] for key in keys}
                )

        if not dist.is_initialized() or dist.get_rank() == 0:
            if not os.path.exists(self.config.validation.texts_path):
                os.makedirs(self.config.validation.texts_path)

            prefix_folder = os.path.join(self.config.validation.texts_path, self.config.training.checkpoints_prefix)
            if not os.path.exists(prefix_folder):
                os.makedirs(prefix_folder)

            file_name = f"{self.step}-N={self.config.dynamic.N}-len={len(result_list)}.json"
            save_path = os.path.join(prefix_folder, file_name)
            json.dump(result_list, open(save_path, "w"), indent=4)
            print(f"Texts are saved in {save_path}")

        # Metrics
        metrics_dict = dict()
        for dataset_name in self.config.data.datasets.datasets_list:
            texts_src = result_dict[dataset_name].get("SRC")
            texts_trg = result_dict[dataset_name]["TRG"]
            texts_gen = result_dict[dataset_name]["GEN"]

            metrics_dict[dataset_name] = dict()

            for metric_name in self.config.data.datasets.metrics[dataset_name]["metrics"]:
                metrics_dict[dataset_name][metric_name] = compute_metric(
                    metric_name,
                    predictions=texts_gen,
                    references=texts_trg,
                    sources=texts_src
                )

        ## Metrics Logging
        if not dist.is_initialized() or dist.get_rank() == 0:
            for dataset_name in self.config.data.datasets.datasets_list:
                print("-----", f"{dataset_name}-{split}", "-----")
                for metric_name in self.config.data.datasets.metrics[dataset_name]["metrics"]:
                    value = metrics_dict[dataset_name][metric_name]
                    if isinstance(value, dict):
                        for key in value:
                            print(f"{key}: {value[key]:0.5f}")
                            self.log_metric(metric_name=f"{dataset_name}-{split}", loader_name=key, value=value[key])
                    else:
                        print(f"{metric_name}: {value:0.5f}")
                        self.log_metric(metric_name=f"{dataset_name}-{split}", loader_name=metric_name, value=value)

        if split == "test":
            self.tracked_test_metric[self.step] = metrics_dict[self.config.tracked_dataset][self.config.tracked_metric]
        
        if dist.is_initialized():
            seed = self.config.seed + self.step + dist.get_rank()
        else:
            seed = self.config.seed + self.step
        set_seed(seed)

        self.switch_back_from_ema()
        self.ddp_score_estimator.train()
        self.score_estimator.train()
