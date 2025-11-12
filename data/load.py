import json
from functools import partial

from transformers import AutoTokenizer
import torch
import itertools

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
import argparse
import os


def tokenize(texts, tokenizer, do_group_texts=False, max_seq_len=None):
    if do_group_texts:
        tokens = tokenizer(
            texts['text'],
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False
        )
    else:
        tokens = tokenizer(
            texts['text'],
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            max_length=max_seq_len,
        )
    return tokens

def group_texts(examples, max_seq_len, sep_token):
    # Concatenate all texts.
    input_ids = examples['input_ids']
    for i in range(len(input_ids)):
        input_ids[i].append(sep_token)

    concatenated_examples = [sep_token] + list(itertools.chain(*input_ids))
    concatenated_examples = torch.tensor(concatenated_examples)
    reminder = len(concatenated_examples) % max_seq_len
    if reminder > 0:
        concatenated_examples = concatenated_examples[:-reminder]

    grouped_input_ids = concatenated_examples.view(-1, max_seq_len)

    return {'input_ids': grouped_input_ids}


def download_qqp(dataset_path):
    dt = load_dataset("glue", "qqp")
    dt = dt.filter(lambda x: x["label"] == 1)
    dt = dt.remove_columns(["label", "idx"])
    dt = concatenate_datasets([dt["train"], dt["validation"]])
    
    dt = dt.train_test_split(test_size=0.2, seed=0)
    dt_train = dt["train"]
    dt = dt["test"].train_test_split(test_size=0.5, seed=0)

    dt = DatasetDict(
        {
            "train": dt_train,
            "validation": dt["train"],
            "test": dt["test"],
        }
    )
    dt.save_to_disk(dataset_path)


def download_xsum(dataset_path):
    dt = load_dataset("EdinburghNLP/xsum")
    dt.save_to_disk(dataset_path)


def download_paradetox(dataset_path):
    dt = load_dataset("s-nlp/paradetox")["train"]
    dt = dt.train_test_split(test_size=0.2, seed=0)
    dt_train = dt["train"]
    dt = dt["test"].train_test_split(test_size=0.5, seed=0)

    dt = DatasetDict(
        {
            "train": dt_train,
            "validation": dt["train"],
            "test": dt["test"],
        }
    )
    dt.save_to_disk(dataset_path)


def download_rocstory():
    def preprocess(batch):
        targets = []
        size = len(batch["storyid"])
        for i in range(size):
            text = " ".join([batch[f"sentence{k}"][i] for k in range(1, 6)])
            targets.append(text)
        return {"text": targets}

    dt = load_dataset("wza/roc_stories")
    dt = dt["train"]
    dt = dt.map(
        preprocess,
        batched=True,
        num_proc=30,
        desc="Loading...",
        remove_columns=dt.column_names,
    )
    dataset = dt.train_test_split(test_size=5000, seed=0)

    return dataset


def download_openwebtext():
    train_dt = load_dataset('openwebtext', split='train[:-100000]', cache_dir='~/.cache/huggingface/datasets/')
    test_dt = load_dataset('openwebtext', split='train[-100000:]', cache_dir='~/.cache/huggingface/datasets/')

    dataset = DatasetDict({'train': train_dt, 'test': test_dt})

    return dataset


def download_wikipedia():
    train_dt = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split='train[:-100000]', cache_dir='~/.cache/huggingface/datasets/'
    )
    test_dt = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split='train[-100000:]', cache_dir='~/.cache/huggingface/datasets/'
    )

    dataset = DatasetDict({'train': train_dt, 'test': test_dt})

    return dataset


def process_diffuseq_dataset(dataset_path):
    splits = ['train', 'valid', 'test']
    dataset = {split: {'src': [], 'trg': []} for split in splits}

    for split in splits:
        with open(f'{dataset_path}/{split}.jsonl', 'r') as file:
            # Iterate over each line in the file
            for line in file:
                # Strip any leading/trailing whitespace (including newlines)
                line = line.strip()
                # Parse the JSON string into a dictionary
                data_dict = json.loads(line)
                dataset[split]['src'].append(data_dict['src'])
                dataset[split]['trg'].append(data_dict['trg'])

    for k, v in dataset.items():
        dataset[k] = Dataset.from_dict(v)
    dataset = DatasetDict(dataset)
    dataset['validation'] = dataset['valid']
    del dataset['valid']
    dataset.save_to_disk(dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset arguments")
    parser.add_argument(
        "--dataset_name", type=str, default=None, 
        choices=[
            "rocstories", "paradetox", "qqp", "xsum", "quasar_t", "newsela_auto", "openwebtext", 'wikipedia'
        ],
        required=True,
    )
    parser.add_argument(
        "--dataset_path", type=str, default=''.join([os.getcwd(), '/datasets/']),
        required=False,
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default='bert-base-cased',
    )
    parser.add_argument(
        "--max_seq_len", type=int, required=False, help='Max sequence length in tokens'
    )
    parser.add_argument(
        "--group_texts", action='store_true', help='Concatenate all texts in a one long string'
    )
    parser.add_argument(
        "--tokenize", action='store_true', help='Concatenate all texts in a one long string'
    )

    args = parser.parse_args()

    save_path = args.dataset_path + args.dataset_name
    if args.dataset_name == "rocstories":
        dataset = download_rocstory()
    elif args.dataset_name == "openwebtext":
        dataset = download_openwebtext()
    elif args.dataset_name == "wikipedia":
        dataset = download_wikipedia()
    elif args.dataset_name == "paradetox":
        download_paradetox(save_path)
    elif args.dataset_name == "qqp":
        download_qqp(save_path)
    elif args.dataset_name == "xsum":
        download_xsum(save_path)
    elif args.dataset_name == "quasar_t" or args.dataset_name == "newsela_auto":
        process_diffuseq_dataset(save_path)

    if args.dataset_name in ['rocstories', 'openwebtext', 'wikipedia']:
        if args.tokenize:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
            tokenized_dataset = dataset.map(
                partial(tokenize, tokenizer=tokenizer, do_group_texts=args.group_texts, max_seq_len=args.max_seq_len),
                batched=True,
                num_proc=16,
                desc='Tokenizing',
                remove_columns=dataset.column_names
            )
            if args.group_texts:
                sep_token = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
                tokenized_dataset = tokenized_dataset.map(
                    partial(args.group_texts, max_seq_len=args.max_seq_len, sep_token=sep_token),
                    batched=True,
                    num_proc=16,
                    desc='Grouping'
                )

            suffix = '_grouped' if args.group_texts else ''
            tokenized_dataset.save_to_disk(save_path + suffix)
        else:
            dataset.save_to_disk(save_path)
