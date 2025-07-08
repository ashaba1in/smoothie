## Smoothie: Smoothing Diffusion on Token Embeddings for Text Generation

Paper: https://arxiv.org/pdf/2505.18853

## Requirements

* Python libraries: See [requirements.txt](./requirements.txt) for exact library dependencies. You can use the following commands with Miniconda3 to create and activate your Python environment:
  - `conda create --name smoothie python=3.9`
  - `conda activate smoothie`
  - `conda install pip`
  - `pip install -r requirements.txt`
  - `python -m spacy download en`

## Dataset loading

For Newsela-Auto and Quasar-T datasets you first need to download files `train.json`, `valid.json` and `test.json` from [DiffuSeq github](https://github.com/Shark-NLP/DiffuSeq/tree/main) and put them in the `./datasets/` folder.

Then you should run the following command:
```
python -m data.load --dataset_name=dataset_name
```

For any other dataset used in the paper, you can run the command above without downloading anything.

The `'dataset_name'` is one of the following:
 - `'rocstories'`
 - `'qqp'`
 - `'xsum'`
 - `'newsela-auto'`
 - `'quasar_t`


## Diffusion training

To train basic Smoothie setup, run

```
torchrun --nproc_per_node=n train_diffusion.py --dataset_name dataset_name --smooth_diffusion
```

This script will train Smoothie model used in the paper.

## Diffusion evaluation

To evaluate the trained model, run

```
torchrun --nproc_per_node=n evaluate_diffusion.py --dataset_name dataset_name --smooth_diffusion --checkpoints_name checkpoints_name"
```

where `checkpoints_name` is a name of the folder with saved checkpoint. By default, it is `smoothie-{dataset_name}`
