import json

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
import argparse
import os


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


def download_rocstory(dataset_path):
    def preprocess(batch):
        targets = []
        size = len(batch["storyid"])
        for i in range(size):
            text = " ".join([batch[f"sentence{k}"][i] for k in range(1, 6)])
            targets.append(text)
        return {"target": targets}

    dt = load_dataset("wza/roc_stories")
    dt = dt["train"]
    dt = dt.map(
        preprocess,
        batched=True,
        num_proc=30,
        desc="Loading...",
        remove_columns=dt.column_names,
    )
    dt = dt.train_test_split(test_size=5000, seed=0)
    dt.save_to_disk(dataset_path)


def download_openwebtext(dataset_path):
    train_dt = load_dataset('openwebtext', split='train[:-100000]', cache_dir='~/.cache/huggingface/datasets/')
    tes_dt = load_dataset('openwebtext', split='train[-100000:]', cache_dir='~/.cache/huggingface/datasets/')

    dataset = DatasetDict({'train': train_dt, 'test': tes_dt})
    dataset.save_to_disk(dataset_path)

def download_wikipedia(dataset_path):
    train_dt = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split='train[:-100000]', cache_dir='~/.cache/huggingface/datasets/'
    )
    tes_dt = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split='train[-100000:]', cache_dir='~/.cache/huggingface/datasets/'
    )

    dataset = DatasetDict({'train': train_dt, 'test': tes_dt})
    dataset.save_to_disk(dataset_path)

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
            "rocstories", "qqp", "xsum", "quasar_t", "newsela_auto", "openwebtext", 'wikipedia'
        ],
        required=True,
    )
    parser.add_argument(
        "--dataset_path", type=str, default=''.join([os.getcwd(), '/datasets/']),
        required=False,
    )

    args = parser.parse_args()

    if not os.path.isfile(args.dataset_path + args.dataset_name):
        if args.dataset_name == "rocstories":
            download_rocstory(args.dataset_path + args.dataset_name)
        if args.dataset_name == "openwebtext":
            download_openwebtext(args.dataset_path + args.dataset_name)
        if args.dataset_name == "wikipedia":
            download_wikipedia(args.dataset_path + args.dataset_name)
        elif args.dataset_name == "qqp":
            download_qqp(args.dataset_path + args.dataset_name)
        elif args.dataset_name == "xsum":
            download_xsum(args.dataset_path + args.dataset_name)
        elif args.dataset_name == "quasar_t" or args.dataset_name == "newsela_auto":
            process_diffuseq_dataset(args.dataset_path + args.dataset_name)
