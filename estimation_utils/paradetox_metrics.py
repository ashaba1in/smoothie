import os
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import tqdm
from nltk.tokenize import TreebankWordTokenizer
from nltk.translate.bleu_score import sentence_bleu
from torch.nn.modules.distance import CosineSimilarity
from transformers import RobertaForSequenceClassification, RobertaModel, RobertaTokenizer


def compute_bleu_paradetox(predictions, references):
    bleu_sim = 0
    counter = 0
    for i in range(len(references)):
        if len(references[i]) > 3 and len(predictions[i]) > 3:
            bleu_sim += sentence_bleu([references[i]], predictions[i])
            counter += 1
    if counter == 0:
        return 0
    return float(bleu_sim / counter)


def compute_j_score(predictions, sources):
    fl_results = fluency(predictions=predictions)
    sta_results = style_accuracy(predictions=predictions)
    sim_results = content_preservation(sources, predictions)
    return {
        "sta": np.mean(sta_results),
        "sim": np.mean(sim_results),
        "fl": np.mean(fl_results),
        "j-score": np.mean(fl_results * sta_results * sim_results),
    }


def fluency(predictions):
    # https://github.com/D3Mlab/diffu-detox/blob/471119f8814a8176d62cf8f4074743fe5d6d1a88/evaluation/evaluation_detox/metric_tools/fluency.py#L18
    batch_size = 128
    results = []

    tokenizer = RobertaTokenizer.from_pretrained('cointegrated/roberta-large-cola-krishna2020')
    model = RobertaForSequenceClassification.from_pretrained('cointegrated/roberta-large-cola-krishna2020').cuda()

    for i in tqdm.tqdm(range(0, len(predictions), batch_size)):
        batch = tokenizer(predictions[i:i + batch_size], return_tensors='pt', padding=True)
        for key in batch.keys():
            batch[key] = batch[key].to('cuda')
        with torch.no_grad():
            outputs = model(**batch)['logits']
        result = outputs.argmax(1).float().data.tolist()
        results.extend([1 - item for item in result])
    return np.array(results)


def style_accuracy(predictions):
    batch_size = 128
    results = []

    tokenizer = RobertaTokenizer.from_pretrained("SkolkovoInstitute/roberta_toxicity_classifier")
    model = RobertaForSequenceClassification.from_pretrained(
        "SkolkovoInstitute/roberta_toxicity_classifier"
    )

    for i in tqdm.tqdm(range(0, len(predictions), batch_size)):
        batch = tokenizer(predictions[i : i + batch_size], return_tensors="pt", padding=True)
        with torch.no_grad():
            result = model(**batch)["logits"].argmax(1).float().data.tolist()
        results.extend([1 - item for item in result])

    accuracy = np.mean(results)
    return accuracy


def content_preservation(sources, predictions):
    assert len(sources) == len(predictions)

    batch_size = 128
    sim_model = SimilarityEvaluator()

    sim_scores = []

    for i in tqdm.tqdm(range(0, len(sources), batch_size)):
        sim_scores.extend(
            sim_model.find_similarity(sources[i : i + batch_size], predictions[i : i + batch_size])
        )

    return np.array(sim_scores)


class SimilarityEvaluator:
    def __init__(
        self,
        model_path="/home/alsh689h/workspaces/horse/alsh689h-base/smoothie/sim.pt",
        tokenizer_path="/home/alsh689h/workspaces/horse/alsh689h-base/smoothie/sim.sp.30k.model",
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.tok = TreebankWordTokenizer()
        model = torch.load(self.model_path, weights_only=False)
        state_dict = model["state_dict"]
        vocab_words = model["vocab_words"]
        args = model["args"]
        # turn off gpu
        self.model = WordAveraging(args, vocab_words)
        self.model.load_state_dict(state_dict, strict=True)
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.tokenizer_path)
        self.model.eval()

    def make_example(self, sentence):
        sentence = sentence.lower()
        sentence = " ".join(self.tok.tokenize(sentence))
        sentence = self.sp.EncodeAsPieces(sentence)
        wp1 = Example(" ".join(sentence))
        wp1.populate_embeddings(self.model.vocab)
        return wp1

    def find_similarity(self, s1, s2):
        with torch.no_grad():
            s1 = [self.make_example(x) for x in s1]
            s2 = [self.make_example(x) for x in s2]
            wx1, wl1, wm1 = self.model.torchify_batch(s1)
            wx2, wl2, wm2 = self.model.torchify_batch(s2)
            scores = self.model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
            return [x.item() for x in scores]

    def find_similarity_batched(self, inputs, preds, batch_size=32):
        assert len(inputs) == len(preds)
        sim_scores = []
        for i in range(0, len(inputs), batch_size):
            sim_scores.extend(
                self.find_similarity(inputs[i : i + batch_size], preds[i : i + batch_size])
            )
        return np.array(sim_scores)

    def embed_texts(self, texts, batch_size=128):
        result = []
        for i in range(0, len(texts), batch_size):
            wx, wl, wm = self.model.torchify_batch(
                [self.make_example(x) for x in texts[i : i + batch_size]]
            )
            with torch.no_grad():
                tensors = torch.nn.functional.normalize(self.model.encode(wx, wm, wl))
            result.append(tensors.cpu().numpy())
        return np.concatenate(result)


class ParaModel(nn.Module):
    def __init__(self, args, vocab):
        super(ParaModel, self).__init__()

        self.args = args
        self.vocab = vocab
        self.gpu = args.gpu

        self.cosine = CosineSimilarity()

    def compute_mask(self, lengths):
        lengths = lengths.cpu()
        max_len = torch.max(lengths)
        range_row = torch.arange(0, max_len).long()[None, :].expand(lengths.size()[0], max_len)
        mask = lengths[:, None].expand_as(range_row)
        mask = range_row < mask
        mask = mask.float()
        if self.gpu >= 0:
            mask = mask.cuda()
        return mask

    def torchify_batch(self, batch):
        max_len = 0
        for i in batch:
            if len(i.embeddings) > max_len:
                max_len = len(i.embeddings)

        batch_len = len(batch)

        np_sents = np.zeros((batch_len, max_len), dtype="int32")
        np_lens = np.zeros((batch_len,), dtype="int32")

        for i, ex in enumerate(batch):
            np_sents[i, : len(ex.embeddings)] = ex.embeddings
            np_lens[i] = len(ex.embeddings)

        idxs, lengths, masks = (
            torch.from_numpy(np_sents).long(),
            torch.from_numpy(np_lens).float().long(),
            self.compute_mask(torch.from_numpy(np_lens).long()),
        )

        if self.gpu >= 0:
            idxs = idxs.cuda()
            lengths = lengths.cuda()
            masks = masks.cuda()

        return idxs, lengths, masks

    def scoring_function(self, g_idxs1, g_mask1, g_lengths1, g_idxs2, g_mask2, g_lengths2):
        g1 = self.encode(g_idxs1, g_mask1, g_lengths1)
        g2 = self.encode(g_idxs2, g_mask2, g_lengths2)
        return self.cosine(g1, g2)


class WordAveraging(ParaModel):
    def __init__(self, args, vocab):
        super().__init__(args, vocab)

        self.vocab = vocab
        self.embedding = nn.Embedding(len(self.vocab), self.args.dim)

        if args.gpu >= 0:
            self.cuda()

    def encode(self, idxs, mask, lengths):
        word_embs = self.embedding(idxs)
        word_embs = word_embs * mask[:, :, None]
        g = word_embs.sum(dim=1) / lengths[:, None].float()
        return g


def lookup(words, w):
    w = w.lower()
    if w in words:
        return words[w]


class Example(object):
    def __init__(self, sentence):
        self.sentence = sentence.strip().lower()
        self.embeddings = []
        self.representation = None

    def populate_embeddings(self, words):
        sentence = self.sentence.lower()
        arr = sentence.split()
        for i in arr:
            emb = lookup(words, i)
            if emb:
                self.embeddings.append(emb)
        if len(self.embeddings) == 0:
            self.embeddings.append(words["UUUNKKK"])


def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res
