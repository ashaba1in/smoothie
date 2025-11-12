import ml_collections
import os
from transformers import AutoConfig
import torch


def create_config(args):
    config = ml_collections.ConfigDict()

    if args.charisma:
        config.work_dir = os.getcwd()
    else:
        config.work_dir = "/data/horse/ws/alsh689h-base/smoothie"

    training = config.training = ml_collections.ConfigDict()
    training.accum_batch_steps = 1
    training.training_iters = args.training_iters * training.accum_batch_steps
    training.checkpoint_freq = 25_000 * training.accum_batch_steps
    training.eval_freq = 25_000 * training.accum_batch_steps
    training.batch_size = args.batch_size // training.accum_batch_steps
    training.ode_sampling = False
    training.checkpoints_folder = os.path.join(config.work_dir, "checkpoints")
    training.checkpoint_name = ""
    training.x_0_variance = args.x_0_variance

    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.
    optim.linear_warmup = 5000 * training.accum_batch_steps
    optim.lr = args.lr
    optim.min_lr = args.lr
    optim.warmup_lr = 1e-8
    optim.weight_decay = args.wd
    optim.beta_1 = 0.9
    optim.beta_2 = 0.98
    optim.eps = 1e-6

    validation = config.validation = ml_collections.ConfigDict()
    validation.batch_size = training.batch_size
    validation.num_gen_texts = args.num_gen_texts
    validation.texts_path = os.path.join(config.work_dir, "generated_texts")
    validation.cfg_coef = 0.

    dynamic = config.dynamic = ml_collections.ConfigDict()
    dynamic.scheduler = args.scheduler
    dynamic.N = args.num_gen_steps
    dynamic.beta_min = 0.1
    dynamic.beta_max = 20
    dynamic.ode_sampling = False
    dynamic.coef_d = args.coef_d
    dynamic.delta = args.delta
    dynamic.sigma_min = args.sigma_min
    dynamic.sigma_max = args.sigma_max
    dynamic.simplex_value = args.simplex_value

    model = config.model = ml_collections.ConfigDict()
    model.ema_rate = 0.9999
    model.downstream_task = ""
    model.prediction = "x_0"
    model.loss = "L_x_0"
    model.encoder_name = args.encoder_name
    model.conditional_encoder_name = model.encoder_name
    model.encoder_name_hash = model.encoder_name.replace("/", "-")
    model.conditional_encoder_name_hash = model.conditional_encoder_name.replace("/", "-")
    model.t5_encoder = True

    data = config.data = ml_collections.ConfigDict()
    data.datasets = create_datasets_config(args)
    data.base_path = os.path.join(config.work_dir, "/datasets")
    if args.max_sequence_len is None:
        data.max_sequence_len = get_sequence_len(data.datasets.datasets_list[0])
    else:
        data.max_sequence_len = args.max_sequence_len
    if args.max_context_len is None:
        data.max_context_len = get_context_len(data.datasets.datasets_list[0])
    else:
        data.max_context_len = args.max_context_len
    data.add_special_tokens = args.add_special_tokens
    data.path = ""
    data.swap_cfg_coef = args.swap_cfg_coef
    data.enc_gen_mean = f"{data.base_path}/{data.datasets.datasets_list[0]}/statistics/encodings-{model.encoder_name_hash}-mean.pt"
    data.enc_gen_std = f"{data.base_path}/{data.datasets.datasets_list[0]}/statistics/encodings-{model.encoder_name_hash}-std.pt"
    data.group_texts = args.group_texts

    config.finetuning = False
    config.seed = 0
    config.ddp = True
    config.use_self_cond = args.use_self_cond
    if 'rocstories' in data.datasets.datasets_list or \
        'wikipedia' in data.datasets.datasets_list or \
        'openwebtext' in data.datasets.datasets_list:
        config.is_conditional = False
    else:
        config.is_conditional = True

    config.emb_statistics_agg_type = args.emb_statistics_agg_type
    config.smooth_diffusion = args.smooth_diffusion
    config.tess_diffusion = args.tess_diffusion
    config.predict_tokens = args.predict_tokens
    config.clamp = args.clamp

    decoder = config.decoder = create_decoder_config()
    if args.training_epochs is not None:
        decoder.epochs = args.training_epochs
    decoder.dataset = data.datasets.datasets_list[0]
    decoder.name = args.decoder_name if args.decoder_name is not None else f"decoder-{model.encoder_name_hash}-transformer"
    decoder.name += decoder.suffix
    decoder.is_conditional = config.is_conditional
    decoder.condition_type = 'cross-attention'
    decoder.condition_encoder = args.decoder_condition_encoder
    decoder.decoder_path = f"{data.base_path}/{data.datasets.datasets_list[0]}/{decoder.name}.pth"

    config.se_config = create_se_config(model.encoder_name)
    config.se_config.is_conditional = config.is_conditional
    config.se_config.vocab_size = AutoConfig.from_pretrained(model.encoder_name).vocab_size
    config.se_config.use_self_cond = config.use_self_cond
    config.se_config.self_cond_type = args.self_cond_type
    config.se_config.predict_tokens = config.predict_tokens

    device_name = torch.cuda.get_device_name(0)
    if 'H100' in device_name or 'A100' in device_name or 'V100' in device_name:
        config.se_config._attn_implementation = 'sdpa'
    else:
        config.se_config._attn_implementation = 'eager'
    config.se_config.max_sequence_len = data.max_sequence_len
    config.se_config.max_context_len = data.max_context_len
    config.se_config.condition_type = args.condition_type
    config.se_config.condition_encoder = args.condition_encoder
    config.se_config.sigma_min = args.sigma_min
    config.se_config.model_type = args.model_type

    config.project_name = args.project_name
    config.neptune_api_token = open('neptune_token.txt', 'r').read()

    config.timesteps = "linear"

    if args.checkpoints_name is None:
        pref = ""
        if args.smooth_diffusion:
            pref = f"smoothie-"
        if args.tess_diffusion:
            pref = f"tess-"
        training.checkpoints_prefix = f"{pref}{data.datasets.datasets_list[0]}"
        if args.run_name:
            training.checkpoints_prefix += f"-{args.run_name}"
    else:
        training.checkpoints_prefix = args.checkpoints_name

    config.eval = False

    config.tracked_dataset = data.datasets.datasets_list[0]
    config.tracked_metric = data.datasets.metrics[config.tracked_dataset]["tracked_metric"]
    config.higher_better = False if config.tracked_metric == 'ppl' else True
    config.save_top_k = 2

    return config


def create_se_config(encoder_name):
    se_config = AutoConfig.from_pretrained(encoder_name)
    se_config.attention_head_size = se_config.hidden_size // se_config.num_attention_heads
    se_config.rope_theta = 1000000.0
    se_config.layer_norm_eps = 1e-05
    return se_config


def create_datasets_config(args):
    config = ml_collections.ConfigDict()
    config.downstream_tasks = ["paradetox", "qqp", "xsum", "newsela_auto", "quasar_t"]
    if args.dataset_name is None:
        config.datasets_list = ["rocstories"]
    else:
        config.datasets_list = [args.dataset_name]
    config.metrics = {
        "rocstories": {"metrics": ["mauve", "div", "ppl"],
                       "tracked_metric": "mauve"},
        "wikipedia": {"metrics": ["mauve", "div", "ppl"],
                      "tracked_metric": "mauve"},
        "openwebtext": {
            "metrics": ["div", "ppl", "mauve"],
            "tracked_metric": "mauve"
        },
        "paradetox": {
            "metrics": ["bleu_paradetox", "j_score", "ppl"],
            "tracked_metric": "j_score",
        },
        "qqp": {
            "metrics": ["bleu", "bert-score", "rougeL", "div1", "div4"],
            "tracked_metric": "bert-score",
        },
        "xsum": {
            "metrics": ["bleu", "bert-score", "rouge1", "rouge2", "rougeL"],
            "tracked_metric": "rougeL",
        },
        "wiki_auto": {
            "metrics": ["bleu", "bert-score", "rouge1", "rouge2", "rougeL"],
            "tracked_metric": "bert-score",
        },
        "newsela_auto": {
            "metrics": ["sari", "bleu", "bert-score", "rougeL"],
            "tracked_metric": "sari",
        },
        "quasar_t": {
            "metrics": ["bleu", "bert-score", "rougeL", "div1", "div4"],
            "tracked_metric": "bert-score",
        },
    }
    return config


def create_decoder_config():
    config = ml_collections.ConfigDict()

    config.max_sequence_len = 128
    config.noise_sigma = 0.5
    config.lr = 1e-4
    config.betas = (0.9, 0.98)
    config.weight_decay = 0.001
    config.batch_size = 64
    config.epochs = 2
    config.max_norm = 1.0
    config.is_conditional = False
    config.dataset = ""
    config.T = 0.15
    config.eps = 0.0
    config.diffusion_forward = False
    config.suffix = ""
    config.num_hidden_layers = 3

    return config


def get_sequence_len(dataset_name):
    data = {
        "wikipedia": 256,
        "rocstories": 80,
        "openwebtext": 1024,
        "paradetox": 40,
        "qqp": 50,
        "xsum": 64,
        "wiki_auto": 100,
        "newsela_auto": 64,
        "quasar_t": 50,
    }
    return data[dataset_name]


def get_context_len(dataset_name):
    data = {
        "wikipedia": 256,
        "rocstories": 80,
        "openwebtext": 1024,
        "paradetox": 40,
        "qqp": 50,
        "xsum": 512,
        "wiki_auto": 100,
        "newsela_auto": 64,
        "quasar_t": 100,
    }
    return data[dataset_name]
