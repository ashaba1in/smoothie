import torch
import torch.nn as nn
from copy import deepcopy
from .score_estimator import BertBlock, get_extended_attention_mask, ConditionEncoder


class Decoder(nn.Module):
    def __init__(self, decoder_config, diffusion_config):
        super().__init__()

        self.num_hidden_layers = decoder_config.num_hidden_layers

        arch_config = deepcopy(diffusion_config)
        arch_config.is_conditional = decoder_config.is_conditional
        arch_config.condition_type = decoder_config.condition_type
        self.blocks = torch.nn.ModuleList(
            [BertBlock(arch_config) for _ in range(0, self.num_hidden_layers)]
        )
        self.fc = nn.Linear(arch_config.hidden_size, arch_config.vocab_size)

        if decoder_config.condition_encoder == 'transformer':
            self.condition_encoder = ConditionEncoder(diffusion_config, num_hidden_layers=3)
        else:
            self.condition_encoder = lambda *x: x[0]

    def forward(self, x, cond_x=None, cond_mask=None):
        extended_cond_mask = get_extended_attention_mask(cond_mask, x.dtype)
        if cond_x is not None:
            cond_x = self.condition_encoder(cond_x, extended_cond_mask)

        for _, block in enumerate(self.blocks):
            x = block(
                hidden_states=x,
                attention_mask=None,
                encoder_hidden_states=cond_x,
                encoder_attention_mask=extended_cond_mask
            )
        x = self.fc(x)
        return x
