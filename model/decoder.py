import torch
import torch.nn as nn
from copy import deepcopy
from .score_estimator import BertBlock


class Decoder(nn.Module):
    def __init__(self, decoder_config, diffusion_config):
        super().__init__()

        self.num_hidden_layers = decoder_config.num_hidden_layers

        arch_config = deepcopy(diffusion_config)
        arch_config.is_conditional = decoder_config.is_conditional
        arch_config.condition_type = decoder_config.condition_type
        # arch_config.is_decoder = decoder_config.is_conditional
        arch_config.is_decoder = False
        self.blocks = torch.nn.ModuleList(
            [BertBlock(arch_config) for _ in range(0, self.num_hidden_layers)]
        )
        self.fc = nn.Linear(arch_config.hidden_size, arch_config.vocab_size)

    def forward(self, x, cond_x=None, cond_mask=None):
        extended_cond_mask = self.get_extended_attention_mask(cond_mask)
        for _, block in enumerate(self.blocks):
            x = block(
                hidden_states=x,
                attention_mask=None,
                encoder_hidden_states=cond_x,
                encoder_attention_mask=extended_cond_mask
            )
        x = self.fc(x)
        return x        

    def get_extended_attention_mask(self, attention_mask):
        if attention_mask is None:
            return None
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.iinfo(attention_mask.dtype).min
        return extended_attention_mask
