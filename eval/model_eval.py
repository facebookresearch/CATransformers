"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

"""
In this code, we provide functionality to evaluate the accuracy of pruned models against MSCOCO and Imagenet1k. 
We also pretrain the pruned models then evaluate against MSCOCO

Evaluation code is borrowed from https://github.com/mlfoundations/datacomp/blob/main/eval_utils/wds_eval.py
Licensed under MIT License
"""

import os, sys
import argparse

import torch
from clip_benchmark.datasets.builder import build_dataset
from clip_benchmark.metrics import zeroshot_classification as zsc
from clip_benchmark.metrics import zeroshot_retrieval as zsr
import open_clip
import math
import time

# from pruning import prune_model
from eval.pruning import prune_model
from eval.model_constants import calculate_flop, orig_models, MAX_EPOCH

import os
import subprocess


class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        super().__init__()
        self._dataset = hf_dataset
        self.transform = (lambda x: x) if transform is None else transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int):
        return (
            self.transform(self._dataset[index]["image"]),
            self._dataset[index]["caption"],
        )

def parse_args(parser):
    parser.add_argument(
        "--model-arch",
        type=str,
        required=False,
        default='ViT-B-16',
        help="base model architecture",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        required=False,
        default='datacomp_xl_s13b_b90k',
        help="base pretrained model",
    )
    parser.add_argument(
        "--text-layers",
        type=float,
        required=False,
        default=1,
        help="new num of layers",
    )
    parser.add_argument(
        "--text-embed-dim",
        type=float,
        required=False,
        help="new embedding dimension (must be a multiple of orig_dim / 8)",
    )
    parser.add_argument(
        "--text-ffn-dim",
        type=float,
        required=False,
        default=1,
        help="new FFN dimension (must be a multiple of orig_dim / 8)",
    )
    parser.add_argument(
        "--text-head-num",
        type=float,
        required=False,
        default=1,
        help="new num attention head",
    )
    parser.add_argument(
        "--vision-layers",
        type=float,
        required=False,
        default=1,
        help="new num of layers",
    )
    parser.add_argument(
        "--vision-embed-dim",
        type=float,
        required=False,
        default=1,
        help="new embedding dimension (must be a multiple of orig_dim / 8)",
    )
    parser.add_argument(
        "--vision-ffn-dim",
        type=float,
        required=False,
        default=1,
        help="new ffn dimension (must be a multiple of orig_dim / 8)",
    )
    parser.add_argument(
        "--vision-head-num",
        type=float,
        required=False,
        default=1,
        help="new num attention head",
    )
    parser.add_argument(
        "--load-checkpoint",
        required=False,
        default=None,
        help="path to load pretrained model from a checkoiunt",
    )
    return parser


def eval_zeroShotClassification(task,
    model,
    transform, 
    model_arch="ViT-B-16",
    data_root=None,
    dataset_len=None,
    batch_size=64,
    num_workers=4,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
     # Load data
    dataset, dataloader = create_webdataset(
        task, transform, data_root, dataset_len, batch_size, num_workers, task_type='zeroshot_classification'
    )

    zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
    classnames = dataset.classes if hasattr(dataset, "classes") else None
    assert (
        zeroshot_templates is not None and classnames is not None
    ), "Dataset does not support classification"

    # Evaluate
    classifier = zsc.zero_shot_classifier(
        model,
        open_clip.get_tokenizer(model_arch),
        classnames,
        zeroshot_templates,
        device,
    )
    logits, target = zsc.run_classification(
        model, classifier, dataloader, device, amp=False
    )

    # Compute metrics
    acc1, acc5 = zsc.accuracy(logits, target, topk=(1, 5))
    metrics = {
        "acc1": acc1,
        "acc5": acc5,
    }
    return metrics

def eval_retreival(task,
    model,
    transform, 
    model_arch="ViT-B-16",
    data_root=None,
    dataset_len=None,
    batch_size=64,
    num_workers=4,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = open_clip.get_tokenizer(model_arch) 
    # Load data
    dataset, dataloader = create_webdataset(
        task, transform, data_root, dataset_len, batch_size, num_workers, task_type='retrieval'
    )
    metrics = zsr.evaluate(
        model, dataloader, tokenizer, recall_k_list=[1, 5, 10], device=device
    )
    metrics["mean_recall@1"] = 0.5 * (
        metrics["text_retrieval_recall@1"] + metrics["image_retrieval_recall@1"]
    )
    return metrics

def create_model(model_arch, pretrained, text_layer, text_embedding_dim, text_ffn_dim, text_head_num, vision_layer, vision_embedding_dim, vision_ffn_dim, vision_head_num, load_checkpoint=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model, _, transform = open_clip.create_model_and_transforms(model_arch, pretrained=pretrained)
    model.eval()
    model = prune_model(model, transform, model_arch, pretrained, int(text_layer), int(text_embedding_dim), int(text_ffn_dim), int(text_head_num), int(vision_layer), int(vision_embedding_dim), int(vision_ffn_dim), int(vision_head_num))
    if load_checkpoint != None:
        print("loading model from checkpoint: " + load_checkpoint)
        open_clip.load_checkpoint(model, load_checkpoint)
    model.eval()
    print(model)
    model = model.to(device)

    param_size = 0
    num_parameters = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        num_parameters +=param.nelement()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    print('num Param: {:.3f}M'.format(num_parameters/ 1024**2))
    model_size = '{:.3f}MB'.format(size_all_mb)
    
    return model, transform, model_size

def create_webdataset(
    task, transform, data_root=None, dataset_len=None, batch_size=64, num_workers=4, task_type='zeroshot_classification'
):
    # data_folder = f"wds_{task.replace('/','-')}_test"
    # if data_root is None:
    #     data_root = f"https://huggingface.co/datasets/djghosh/{data_folder}/tree/main"
    # else:
    #     data_root = os.path.join(data_root, data_folder)

    data_folder = f"wds_{task.replace('/','-')}"
    data_root = f"https://huggingface.co/datasets/clip-benchmark/{data_folder}/tree/main"
    max_retries = 10
    retry_delay = 60  # seconds
    home_dir = os.getcwd()
    cache_directory = f"{home_dir}/dataset/{data_folder}"
    os.makedirs(cache_directory, exist_ok=True)
    for attempt in range(max_retries):
        try:
            dataset = build_dataset(
                dataset_name=f"wds/{task}",
                root=data_root,
                transform=transform,
                split="test",
                download=False,
                task=task_type,
                wds_cache_dir=cache_directory
            )
            break
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise Exception("All attempts failed")

    if dataset_len:
        dataset = dataset.with_length((dataset_len + batch_size - 1) // batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset.batched(batch_size),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
    )
    return dataset, dataloader


if __name__ == "__main__":
    # prunes and evaluates the pruned model directly.
    parser = argparse.ArgumentParser(description="Webdataset evaluation script.")
    parser = parse_args(parser)
    args = parser.parse_args()

    checkpoint = None
    if args.load_checkpoint:
        checkpoint = args.load_checkpoint

    model, transform, model_size = create_model(model_arch=args.model_arch, pretrained=args.pretrained, text_layer=args.text_layers, text_embedding_dim=args.text_embed_dim, text_ffn_dim=args.text_ffn_dim, text_head_num=args.text_head_num,
        vision_layer=args.vision_layers, vision_embedding_dim=args.vision_embed_dim, vision_ffn_dim=args.vision_ffn_dim, vision_head_num=args.vision_head_num, load_checkpoint=checkpoint)

    metric_retrieval = eval_retreival(task='mscoco_captions', model=model, transform=transform, model_arch=args.model_arch)

    print(f"MSCOCO Eval Metrics: {metric_retrieval}")


# finetune on the mscoco dataset and evaluate the model on mscoco
def train_and_eval(model_config, model_arch, pretrained):

    text_layer = model_config["num_hidden_layers"] 
    text_ffn_dim = model_config["intermediate_size"]
    text_embedding_dim = model_config["hidden_size"] 
    text_head_num = model_config["num_attn_heads"] 
    vision_layer = model_config["vision_num_hidden_layers"]
    vision_ffn_dim = model_config["vision_intermediate_size"] 
    vision_embedding_dim = model_config["vision_hidden_size"]
    vision_head_num = model_config["vision_num_attn_heads"]

    checkpoint_name = f"model_eval_{model_arch}_{pretrained}_{text_layer}_{text_embedding_dim}_{text_ffn_dim}_{text_head_num}_{vision_layer}_{vision_embedding_dim}_{vision_ffn_dim}_{vision_head_num}"
    home_dir = os.getcwd()
    dataset_dir = os.path.join(home_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_location = f"{home_dir}/dataset/train2014.csv"

    # check if model config already trained befores
    checkpoint_path = f"{home_dir}/logs/{checkpoint_name}/checkpoints/epoch_1.pt"
    if os.path.exists(checkpoint_path):
        print("checkpoint already exists!")
    else:
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} GPUs")

        if device_count == 0:
            print(f"ERROR: CLIP training requires at least one GPU, exiting...")
            sys.exit(1)

        command = f"torchrun --nproc_per_node={device_count} {home_dir}/open_clip_custom/src/open_clip_train/main.py --dataset-type='csv' --train-data={dataset_location} \
        --batch-size 64     --lr 1e-5     --wd 0.1     --epochs=1    --workers=32      --model={model_arch}   --pretrained={pretrained} \
        --text-layers {text_layer} --text-embed-dim {text_embedding_dim} --text-ffn-dim {text_ffn_dim} --text-head-num {text_head_num} \
        --vision-layers {vision_layer} --vision-embed-dim {vision_embedding_dim} --vision-ffn-dim {vision_ffn_dim} --vision-head-num {vision_head_num} --name {checkpoint_name} --scale-flops"
        # Specify the working directory
        working_directory = home_dir
        # Run the command using subprocess
        subprocess.run(command, shell=True, cwd=working_directory)
    
    orig_model_configs = orig_models[model_arch]
    orig_model_flops = calculate_flop(orig_model_configs["text_layer"],orig_model_configs["text_embedding_dim"], orig_model_configs["text_ffn_dim"], orig_model_configs["text_head_num"],
                                        orig_model_configs["vision_layer"], orig_model_configs["vision_embedding_dim"], orig_model_configs["vision_ffn_dim"], orig_model_configs["vision_head_num"])

    pruned_model_flops = calculate_flop(text_layer, text_embedding_dim, text_ffn_dim, text_head_num, 
                                          vision_layer, vision_embedding_dim, vision_ffn_dim, vision_head_num)
    
    flop_ratio = orig_model_flops / pruned_model_flops

    total_epochs = math.ceil(1* flop_ratio)
    if(total_epochs > MAX_EPOCH):
        total_epochs = MAX_EPOCH

    # Eval model 
    model, transform, model_size = create_model(model_arch=model_arch,pretrained=pretrained,
        text_layer=text_layer, text_embedding_dim=text_embedding_dim, text_ffn_dim=text_ffn_dim, text_head_num=text_head_num,
        vision_layer=vision_layer, vision_embedding_dim=vision_embedding_dim, vision_ffn_dim=vision_ffn_dim, vision_head_num=vision_head_num, load_checkpoint=f"{home_dir}/logs/{checkpoint_name}/checkpoints/epoch_{total_epochs}.pt"
    )

    max_retries = 10
    retry_delay = 60  # seconds
    for attempt in range(max_retries):
        try:
            metric_retrieval = eval_retreival(task='mscoco_captions', model=model, transform=transform, model_arch=model_arch)
            break
        except Exception as e:
            print(f"Attempt at evaluation {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise Exception("All attempts at evaluation failed")

    print(f"MSCOCO Eval Metrics: {metric_retrieval}")
    
    return metric_retrieval, model_size

# Helper function on evaluating a model against the MSCOCO dataset
# checkpoint loads a trained checkpoint of the model, if not specified will evaluate the directly pruned model.
def eval_only(model_config, checkpoint=None, model_arch='ViT-B-16', pretrained='datacomp_xl_s13b_b90k'):

    text_layer = model_config["num_hidden_layers"] 
    text_ffn_dim = model_config["intermediate_size"]
    text_embedding_dim = model_config["hidden_size"] 
    text_head_num = model_config["num_attn_heads"] 
    vision_layer = model_config["vision_num_hidden_layers"]
    vision_ffn_dim = model_config["vision_intermediate_size"] 
    vision_embedding_dim = model_config["vision_hidden_size"]
    vision_head_num = model_config["vision_num_attn_heads"]

    # Eval model 
    model, transform, model_size = create_model(model_arch=model_arch,pretrained=pretrained,
        text_layer=text_layer, text_embedding_dim=text_embedding_dim, text_ffn_dim=text_ffn_dim, text_head_num=text_head_num,
        vision_layer=vision_layer, vision_embedding_dim=vision_embedding_dim, vision_ffn_dim=vision_ffn_dim, vision_head_num=vision_head_num,
        load_checkpoint=checkpoint
    )
    max_retries = 10
    retry_delay = 60  # seconds
    for attempt in range(max_retries):
        try:
            metric_retrieval = eval_retreival(task='mscoco_captions', model=model, transform=transform, model_arch=model_arch)
            break
        except Exception as e:
            print(f"Attempt at evaluation {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise Exception("All attempts at evaluation failed")
    # metric_zsc = eval_zeroShotClassification(task='imagenet1k', model=model, transform=transform)
    metric_zsc= {}

    print(f"MSCOCO Eval Metrics: {metric_retrieval}")
    print(f"ImageNet1k Eval Metrics: {metric_zsc}")
    
    return metric_retrieval, metric_zsc, model_size


# Helper function on evaluating a model against the imagenet1K
# checkpoint loads a trained checkpoint of the model, if not specified will evaluate the directly pruned model.
def eval_imagenet(model_config, checkpoint=None, model_arch='ViT-B-16', pretrained='datacomp_xl_s13b_b90k'):

    text_layer = model_config["num_hidden_layers"] 
    text_ffn_dim = model_config["intermediate_size"]
    text_embedding_dim = model_config["hidden_size"] 
    text_head_num = model_config["num_attn_heads"] 
    vision_layer = model_config["vision_num_hidden_layers"]
    vision_ffn_dim = model_config["vision_intermediate_size"] 
    vision_embedding_dim = model_config["vision_hidden_size"]
    vision_head_num = model_config["vision_num_attn_heads"]

    # Eval model 
    model, transform, model_size = create_model(model_arch=model_arch,pretrained=pretrained,
        text_layer=text_layer, text_embedding_dim=text_embedding_dim, text_ffn_dim=text_ffn_dim, text_head_num=text_head_num,
        vision_layer=vision_layer, vision_embedding_dim=vision_embedding_dim, vision_ffn_dim=vision_ffn_dim, vision_head_num=vision_head_num,
        load_checkpoint=checkpoint
    )
    metric_zsc = eval_zeroShotClassification(task='imagenet1k', model=model, transform=transform, model_arch=model_arch)

    print(f"ImageNet1k Eval Metrics: {metric_zsc}")
    
    return metric_zsc

def measure_energy(model_config, checkpoint=None, model_arch='ViT-B-16', pretrained='datacomp_xl_s13b_b90k'):
    text_layer = model_config["num_hidden_layers"] 
    text_ffn_dim = model_config["intermediate_size"]
    text_embedding_dim = model_config["hidden_size"] 
    text_head_num = model_config["num_attn_heads"] 
    vision_layer = model_config["vision_num_hidden_layers"]
    vision_ffn_dim = model_config["vision_intermediate_size"] 
    vision_embedding_dim = model_config["vision_hidden_size"]
    vision_head_num = model_config["vision_num_attn_heads"]

    # Eval model 
    model, transform, model_size = create_model(model_arch=model_arch,pretrained=pretrained,
        text_layer=text_layer, text_embedding_dim=text_embedding_dim, text_ffn_dim=text_ffn_dim, text_head_num=text_head_num,
        vision_layer=vision_layer, vision_embedding_dim=vision_embedding_dim, vision_ffn_dim=vision_ffn_dim, vision_head_num=vision_head_num,
        load_checkpoint=checkpoint
    )
    tokenizer =  open_clip.get_tokenizer(model_arch)
    import torch
    from PIL import Image
    import requests
    from zeus.monitor import ZeusMonitor

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Set the device to the first available GPU
    # torch.cuda.set_device(0)
    image = Image.open(requests.get(url, stream=True).raw)
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    text = tokenizer(["a diagram"])
    text = text.to(device)

    monitor = ZeusMonitor(gpu_indices=[0])

    # Measure total time and energy within the window.
    monitor.begin_window("inference")
    for e in range(1000):

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    measurement = monitor.end_window("inference")
    print(f"Entire inference: {measurement.time} s, {measurement.total_energy} J")
    return (measurement.time / 1000, measurement.total_energy / 1000)











