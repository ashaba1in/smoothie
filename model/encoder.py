import os
import torch
from transformers import AutoModel, AutoTokenizer


class Encoder(torch.nn.Module):
    def __init__(self, encoder_name='bert-base-cased', emb_statistics_agg_type='features', t5_encoder=True):
        super().__init__()
        self.encoder_name = encoder_name
        if 'glove' in encoder_name:
            embeddings = torch.load(os.path.join(encoder_name, 'embeddings.pt'))
        else:
            model = AutoModel.from_pretrained(encoder_name)
            if encoder_name == 'bert-base-cased':
                embeddings = model.embeddings.word_embeddings.weight.data
            elif encoder_name == 'gpt2':
                # padding token
                model.resize_token_embeddings(model.wte.num_embeddings + 1)
                model.wte.weight.data[-1] = 0
                embeddings = model.wte.weight.data
            elif encoder_name == 'google-t5/t5-base':
                if t5_encoder:
                    embeddings = model.encoder.embed_tokens.weight.data
                else:
                    embeddings = model.decoder.embed_tokens.weight.data
            else:
                raise NotImplementedError(f"Tokenizer {encoder_name} is not supported")

        used_ids, unused_ids = self.get_used_ids(encoder_name=encoder_name)
        if emb_statistics_agg_type == 'features':
            dim = 0
        elif emb_statistics_agg_type == 'total':
            dim = (0, 1)
        else:
            raise Exception("Unknown embedding aggregation type, support only ['features', 'total']")

        emb_mean = torch.mean(embeddings[used_ids, :], dim=dim, keepdim=True)
        emb_std = torch.std(embeddings[used_ids, :], dim=dim, keepdim=True)
        embeddings.data = (embeddings.data - emb_mean) / emb_std
        self.register_buffer("embeddings", embeddings)

    def forward(self, input_ids):
        print(self.embeddings.device)
        return self.embeddings[input_ids]

    @staticmethod
    def get_used_ids(encoder_name: str) -> tuple[list[int], list[int]]:
        """Function to get ids to filter unused ids of BERT"""
        vocab = AutoTokenizer.from_pretrained(encoder_name).vocab
        used_ids = []
        unused_ids = []
        for key in vocab.keys():
            if '[unused' in key:
                unused_ids.append(vocab[key])
            else:
                used_ids.append(vocab[key])

        return used_ids, unused_ids
