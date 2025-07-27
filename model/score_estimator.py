from copy import deepcopy

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from transformers import AutoModel
from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput, \
    apply_chunking_to_forward

from utils.util import convert_to_simplex


def get_extended_attention_mask(attention_mask, dtype):
    if attention_mask is None:
        return None
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


class BertBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.condition_type = config.condition_type if config.is_conditional else None
        if self.condition_type == 'cross-attention':
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = self_attention_outputs[0]

        if self.condition_type == 'cross-attention' and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                hidden_states=attention_output,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            attention_output = cross_attention_outputs[0]

        outputs = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


TransformerBlock = BertBlock


class TransformerEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.use_self_cond = config.use_self_cond
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.input_blocks = torch.nn.ModuleList(
            [TransformerBlock(config) for _ in range(0, self.num_hidden_layers // 2)]
        )
        self.output_blocks = torch.nn.ModuleList(
            [TransformerBlock(config) for _ in range(0, self.num_hidden_layers // 2)]
        )
        self.time_layers = torch.nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(0, self.num_hidden_layers)]
        )
        if self.use_self_cond and config.self_cond_type != 'tess':
            self.self_cond_layers = torch.nn.ModuleList(
                [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(0, self.num_hidden_layers)]
            )

        self.condition_type = config.condition_type
        self.max_sequence_len = config.max_sequence_len

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            emb_t=None,
            cond=None,
            cond_mask=None,
            x_0_self_cond=None,
    ):
        x_input_list = []

        for i, block in enumerate(self.input_blocks):
            x_input_list.append(x)
            time_emb = self.time_layers[i](emb_t)
            if self.condition_type == 'concatenation':
                # don't add time embeddings to condition
                time_emb = time_emb.repeat(1, x.shape[1], 1)
                time_emb[:, self.max_sequence_len:] = 0

            x = x + time_emb
            if self.use_self_cond and x_0_self_cond is not None:
                x += self.self_cond_layers[i](x_0_self_cond)
            x = block(
                hidden_states=x,
                attention_mask=attention_mask,
                encoder_hidden_states=cond,
                encoder_attention_mask=cond_mask
            )

        for i, block in enumerate(self.output_blocks):
            ind = i + self.num_hidden_layers // 2
            time_emb = self.time_layers[ind](emb_t)
            if self.condition_type == 'concatenation':
                # don't add time embeddings to condition
                time_emb = time_emb.repeat(1, x.shape[1], 1)
                time_emb[:, self.max_sequence_len:] = 0
            x = x + x_input_list.pop() + time_emb
            if self.use_self_cond and x_0_self_cond is not None:
                x += self.self_cond_layers[ind](x_0_self_cond)
            x = block(
                hidden_states=x,
                attention_mask=attention_mask,
                encoder_hidden_states=cond,
                encoder_attention_mask=cond_mask
            )

        return x


class ConditionEncoder(nn.Module):
    def __init__(self, config, num_hidden_layers):
        super().__init__()

        arch_config = deepcopy(config)
        arch_config.is_conditional = False
        self.blocks = torch.nn.ModuleList(
            [BertBlock(arch_config) for _ in range(num_hidden_layers)]
        )

    def forward(self, x, attention_mask=None):
        for _, block in enumerate(self.blocks):
            x = block(
                hidden_states=x,
                attention_mask=attention_mask,
            )
        return x


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ScoreEstimatorEMB(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.use_self_cond = config.use_self_cond
        self.config = config
        hidden_layer_dim = self.config.hidden_size
        self._hidden_layer_dim = hidden_layer_dim
        self.time_emb = torch.nn.Sequential(
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_layer_dim * 2, hidden_layer_dim)
        )

        self.encoder = TransformerEncoder(config)
        if config.is_conditional and config.condition_encoder == 'transformer':
            self.condition_encoder = ConditionEncoder(config, num_hidden_layers=6)

        if self.use_self_cond and config.self_cond_type != 'tess':
            self.self_condition_encoder = ConditionEncoder(config, num_hidden_layers=3)

        self.condition_type = config.condition_type if config.is_conditional else None
        if self.condition_type == 'concatenation':
            self.sequence_embeddings = torch.nn.Embedding(2, self._hidden_layer_dim)

        if self.condition_type == 'concatenation':
            self._max_position_embeddings = self.config.max_sequence_len + self.config.max_context_len
        else:
            self._max_position_embeddings = self.config.max_sequence_len

        self.register_buffer("position_ids", torch.arange(self._max_position_embeddings).expand((1, -1)))
        self.position_embeddings = torch.nn.Embedding(self._max_position_embeddings, self._hidden_layer_dim)

        if config.self_cond_type == 'tess':
            self.embeddings = AutoModel.from_pretrained('bert-base-cased').embeddings.word_embeddings.weight.data

        if self.config.predict_tokens:
            self.head = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def forward(
            self,
            x_t: torch.Tensor,
            time_t: Optional[torch.Tensor] = None,
            cond=None,
            attention_mask=None,
            cond_mask=None,
            x_0_self_cond=None,
    ):
        assert time_t is not None

        if attention_mask is None:
            attention_mask = torch.ones(*x_t.shape[:-1], device=x_t.device)

        attention_mask = get_extended_attention_mask(
            attention_mask=attention_mask,
            dtype=x_t.dtype
        )
        if cond_mask is not None:
            cond_mask = get_extended_attention_mask(
                attention_mask=cond_mask,
                dtype=x_t.dtype
            )

        if self.use_self_cond:
            if self.config.self_cond_type == 'tess':
                self_cond_D = convert_to_simplex(
                    input_embeddings=x_0_self_cond,
                    sigma_0=self.config.sigma_min,
                    embeddings=self.embeddings.to(x_0_self_cond.device),
                )
                x_t = 0.5 * (x_t + torch.softmax(self_cond_D, dim=-1) @ self.embeddings.to(self_cond_D.device))
                x_0_self_cond = None
            else:
                x_0_self_cond = self.self_condition_encoder(x_0_self_cond)

        emb_t = timestep_embedding(time_t, self._hidden_layer_dim)
        hidden_t = self.time_emb(emb_t)
        hidden_t = hidden_t[:, None, :]

        if self.config.is_conditional:
            if self.config.condition_encoder == 'transformer':
                cond = self.condition_encoder(cond, cond_mask)

            if self.condition_type == 'concatenation':
                x_t = torch.cat((
                    x_t + self.sequence_embeddings(torch.tensor(0, device=x_t.device)),
                    cond + self.sequence_embeddings(torch.tensor(1, device=x_t.device))
                ), dim=-2)
                attention_mask = torch.cat((attention_mask, cond_mask), dim=-1)

        seq_length = x_t.size(1)
        position_ids = self.position_ids[:, :seq_length]
        emb_pos = self.position_embeddings(position_ids)

        hidden_state = x_t + emb_pos

        output = self.encoder(
            x=hidden_state,
            attention_mask=attention_mask,
            emb_t=hidden_t,
            cond=cond,
            cond_mask=cond_mask,
            x_0_self_cond=x_0_self_cond,
        )
        if self.config.is_conditional and self.condition_type == 'concatenation':
            output = output[:, :self.config.max_sequence_len].contiguous()

        if self.config.predict_tokens:
            output = self.head(output)

        return output
