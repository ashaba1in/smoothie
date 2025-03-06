import torch
import numpy as np
from transformers import AutoModel, T5EncoderModel, AutoTokenizer


class Encoder(torch.nn.Module):
    def __init__(self, encoder_link, enc_normalizer, is_change_sp_tokens=True, emb=False):
        super().__init__()
        self.emb = emb
        self.encoder_link = encoder_link
        if "bert" in encoder_link.lower():
            self.encoder = AutoModel.from_pretrained(self.encoder_link)
            self.embeddings = self.encoder.embeddings.word_embeddings.weight.data.cpu()
        elif "roberta" in encoder_link.lower():
            self.encoder = AutoModel.from_pretrained(self.encoder_link)
            self.embeddings = self.encoder.embeddings.word_embeddings.weight.data.cpu()
        elif "t5" in encoder_link.lower():
            self.encoder = T5EncoderModel.from_pretrained(self.encoder_link)
            self.embeddings = self.encoder.encoder.embed_tokens.weight.data.cpu()
        elif "bart" in encoder_link.lower():
            self.encoder = AutoModel.from_pretrained(self.encoder_link).encoder
            self.embeddings = self.encoder.embed_tokens.weight.data.cpu()
        else:
            raise Exception("Unknown encoder name. Add encoder to ./model/encoder.py")
        
        if self.emb:
            if 'bert' in encoder_link:
                used_ids, unused_ids = self.get_used_ids(encoder_link=encoder_link)
            else:
                used_ids = torch.arange(start=0, end=self.embeddings.shape[0], device=self.embeddings.device)
                unused_ids = []
        
            self.emb_mean = torch.mean(self.embeddings[used_ids, :], dim=0, keepdim=True)
            self.emb_std = torch.std(self.embeddings[used_ids, :], dim=0, keepdim=True)
            self.embeddings[unused_ids, :] *= torch.inf
            self.embeddings = self.embeddings.cuda()

        self.enc_normalizer = enc_normalizer
        self.is_change_sp_tokens = is_change_sp_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_link)
        self.register_buffer("zero_emb", torch.zeros((self.encoder.config.hidden_size)))

    def forward(self, input_ids, attention_mask):
        if self.emb:
            return (self.embeddings[input_ids] - self.emb_mean.cuda()[None, :, :]) / self.emb_std.cuda()[None, :, :]
        
        sequence_output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        if self.enc_normalizer is not None:
            sequence_output = self.enc_normalizer.normalize(sequence_output)
        
        if self.is_change_sp_tokens:
            for sp_token_id in self.tokenizer.all_special_ids:
                if sp_token_id == self.tokenizer.pad_token_id:
                    sequence_output[input_ids == sp_token_id] = self.zero_emb.type(sequence_output.dtype)
                else:
                    sequence_output[input_ids == sp_token_id] = self._normalize_emb(self.embeddings[sp_token_id]).type(sequence_output.dtype).to(self.encoder.device)
        
        return sequence_output
    
    def _normalize_emb(self, x):
        return x / torch.norm(x) * np.sqrt(x.shape[-1])

    @staticmethod
    def get_used_ids(encoder_link: str) -> tuple[list[int], list[int]]:
        """Function to get ids to filter unused ids of BERT"""
        vocab = AutoTokenizer.from_pretrained(encoder_link).vocab
        used_ids = []
        unused_ids = []
        for key in vocab.keys():
            if '[unused' in key:
                unused_ids.append(vocab[key])
            else:
                used_ids.append(vocab[key])

        return used_ids, unused_ids
