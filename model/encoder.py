import torch
from transformers import AutoModel, AutoTokenizer


class Encoder(torch.nn.Module):
    def __init__(self, encoder_name='bert-base-cased', emb_statistics_agg_type='features', t5_encoder=True):
        super().__init__()
        self.encoder_name = encoder_name
        model = AutoModel.from_pretrained(self.encoder_name)
        if encoder_name == 'bert-base-cased':
            self.embeddings = model.embeddings.word_embeddings.weight.cpu()
        if encoder_name == 'gpt2':
            self.embeddings = model.wte.weight.cpu()
        if encoder_name == 't5-base':
            if t5_encoder:
                self.embeddings = model.encoder.embed_tokens.weight.cpu()
            else:
                self.embeddings = model.decoder.embed_tokens.weight.cpu()

        used_ids, unused_ids = self.get_used_ids(encoder_name=encoder_name)
        if emb_statistics_agg_type == 'features':
            self.dim = 0
        elif emb_statistics_agg_type == 'total':
            self.dim = (0, 1)
        else:
            raise Exception("Unknown embedding aggregation type, support only ['features', 'total']")

        self.emb_mean = torch.mean(self.embeddings[used_ids, :], dim=self.dim, keepdim=True)
        self.emb_std = torch.std(self.embeddings[used_ids, :], dim=self.dim, keepdim=True)
        self.embeddings.data = (self.embeddings.data - self.emb_mean) / self.emb_std
        self.embeddings.cuda()

    def forward(self, input_ids):
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
