import torch
from evaluate import load
from nltk.util import ngrams
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import spacy
import numpy as np
import torch.nn.functional as F
from .paradetox_metrics import compute_bleu_paradetox, compute_j_score


def compute_metric(metric_name, predictions, references, sources=None, **kwargs):
    if metric_name == "mauve":
        return compute_mauve(predictions=predictions, references=references)
    elif metric_name == "div":
        return compute_diversity(all_texts_list=predictions)['diversity']
    elif metric_name.startswith("div"):
        return distinct_n_grams(texts=predictions, n=int(metric_name[-1]))
    elif metric_name == "mem":
        return compute_memorization(all_texts_list=predictions, human_references=references)
    elif metric_name.startswith("rouge"):
        return compute_rouge(predictions=predictions, references=references)[metric_name]
    elif metric_name == "bert-score":
        return compute_bert_score(predictions=predictions, references=references)
    elif metric_name == "bleu":
        return compute_bleu(predictions=predictions, references=references)
    elif metric_name == "sari":
        return compute_sari(sources=sources, predictions=predictions, references=[[x] for x in references])
    elif metric_name == "ppl":
        return compute_ppl(predictions=predictions)
    elif metric_name == "j_score":
        return compute_j_score(predictions=predictions, sources=sources)
    elif metric_name == "bleu_paradetox":
        return compute_bleu_paradetox(predictions=predictions, references=references)
    else:
        raise Exception(f"Unknown metric: {metric_name}")


def filter_empty_texts(predictions, references):
    pred_list = []
    ref_list = []
    for i in range(len(predictions)):
        if predictions[i] and references[i]:
            pred_list.append(predictions[i])
            ref_list.append(references[i])
    return pred_list, ref_list


@torch.no_grad()
def compute_ppl(predictions, model_id='gpt2-large'):
    torch.cuda.empty_cache()

    predictions = [p for p in predictions if p]
    if len(predictions) == 0:
        return 0

    eval_tokenizer = AutoTokenizer.from_pretrained(model_id)
    if eval_tokenizer.pad_token is None:
        eval_tokenizer.pad_token = eval_tokenizer.eos_token
        eval_tokenizer.pad_token_id = eval_tokenizer.eos_token_id

    eval_model = AutoModelForCausalLM.from_pretrained(model_id).eval().cuda()

    samples = eval_tokenizer(
        predictions,
        return_tensors='pt',
        return_token_type_ids=False,
        return_attention_mask=True,
        truncation=True,
        padding=True,
        max_length=eval_tokenizer.model_max_length,
    )
    attn_mask = samples['attention_mask']
    samples = samples['input_ids']

    batch_size = min(8, samples.shape[0])
    num_batches = samples.shape[0] // batch_size

    nll_sum = 0
    n_tokens = 0
    for i in range(num_batches):
        _samples = samples[i*batch_size:(i+1)*batch_size].cuda()
        _attn_mask = attn_mask[i*batch_size:(i+1)*batch_size].cuda()
        logits = eval_model(_samples, attention_mask=_attn_mask)[0]
        logits = logits.transpose(-1, -2)

        nlls = F.cross_entropy(logits[..., :-1], _samples[..., 1:], reduction='none')

        nll_sum += nlls[_attn_mask[..., :-1].bool()].sum()
        n_tokens += _attn_mask[..., :-1].sum()

    ppl = torch.exp(nll_sum / n_tokens).item()

    return ppl


def compute_mauve(predictions, references, model_id='gpt2-large'):
    torch.cuda.empty_cache()

    mauve = load("mauve")
    assert len(predictions) == len(references)

    predictions, references = filter_empty_texts(predictions, references)
    print('Len predictions for MAUVE:', len(predictions))
    if len(predictions) == 0:
        return 0

    results = mauve.compute(
        predictions=predictions, references=references,
        featurize_model_name=model_id, device_id=0, verbose=False
    )

    return results.mauve


def compute_diversity(all_texts_list):
    ngram_range = [2, 3, 4]

    tokenizer = spacy.load("en_core_web_sm").tokenizer
    token_list = []
    for sentence in all_texts_list:
        token_list.append([str(token) for token in tokenizer(sentence)])
    ngram_sets = {}
    ngram_counts = defaultdict(int)

    metrics = {}
    for n in ngram_range:
        ngram_sets[n] = set()
        for tokens in token_list:
            ngram_sets[n].update(ngrams(tokens, n))
            ngram_counts[n] += len(list(ngrams(tokens, n)))
        metrics[f'{n}gram_repitition'] = 0 if ngram_counts[n] == 0 else (1 - len(ngram_sets[n]) / ngram_counts[n])
    diversity = 1
    for val in metrics.values():
        diversity *= (1 - val)
    metrics['diversity'] = diversity
    return metrics


def distinct_n_grams(texts, n=1):
    """Computes the average distinct n-grams of the generated texts.
    Args:
        texts (list of str): representing the generated texts.
        n (int): n-gram length
    """
    dist = []
    for text in texts:
        total_words = len(text.split())
        n_grams = set(ngrams(text.split(), n))
        if total_words == 0:
            dist.append(0)
        else:
            dist.append(len(n_grams) / total_words)
    return np.nanmean(dist)


def compute_memorization(all_texts_list, human_references, n=4):
    tokenizer = spacy.load("en_core_web_sm").tokenizer
    unique_four_grams = set()
    for sentence in human_references:
        unique_four_grams.update(ngrams([str(token) for token in tokenizer(sentence)], n))

    total = 0
    duplicate = 0
    for sentence in all_texts_list:
        four_grams = list(ngrams([str(token) for token in tokenizer(sentence)], n))
        total += len(four_grams)
        for four_gram in four_grams:
            if four_gram in unique_four_grams:
                duplicate += 1

    return duplicate / total


def compute_rouge(predictions, references):
    torch.cuda.empty_cache()

    rouge = load('rouge')
    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    return result

def compute_sari(sources, predictions, references):
    torch.cuda.empty_cache()

    sari = load('sari')
    result = sari.compute(sources=sources, predictions=predictions, references=references)
    return result['sari']

def compute_bert_score(predictions, references):
    torch.cuda.empty_cache()

    bertscore = load("bertscore", module_type="metric")
    results = bertscore.compute(predictions=predictions, references=references, model_type='microsoft/deberta-xlarge-mnli', lang='en', verbose=True)
    # https://github.com/Shark-NLP/DiffuSeq/blob/f78945d79de5783a4329695c0adb1e11adde31bf/scripts/eval_seq2seq.py#L128C48-L128C115
    return np.mean(results["f1"])


def compute_bleu(predictions, references, max_order=4, smooth=False):
    torch.cuda.empty_cache()

    from .bleu import compute_bleu as bleu
    tokenizer_mbert = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    if isinstance(references[0], str):
        references = [[ref] for ref in references]

    references = [[tokenizer_mbert.tokenize(item) for item in ref] for ref in references]
    predictions = [tokenizer_mbert.tokenize(item) for item in predictions]

    results = bleu(reference_corpus=references, translation_corpus=predictions, max_order=max_order, smooth=smooth)
    return results[0]
