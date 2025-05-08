"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

# coding=utf-8
import evaluate
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision import datasets, transforms
import torch.nn as nn
import os

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    ViTImageProcessor, 
    ViTForImageClassification,
    ViTConfig,
    DataCollatorWithPadding,
)
from torch.optim import Adam
from torch.utils.data import Dataset
from eval import model_constants

# Define task name and keys
task_name = "mrpc"
task_to_keys = {"mrpc": ("sentence1", "sentence2")}

def train_and_eval(model_config, model_name):
    if(model_name in model_constants.vit_models):
        return train_and_eval_vit(model_config, model_name)
    else:
        return train_and_eval_language(model_config, model_name)

# finetune and evaluate language models
def train_and_eval_language(model_config, model_name):

    # Set the seed for PyTorch
    seed = 0
    torch.manual_seed(seed)
    raw_datasets = load_dataset("nyu-mll/glue", task_name)
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)


    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        finetuning_task=task_name,
        token='',
        hidden_size=model_config['hidden_size'],
        intermediate_size=model_config['intermediate_size'],
        num_attention_heads=model_config['num_attn_heads']
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token='')
    if (model_name == "meta-llama/Llama-2-7b-hf" or model_name== "meta-llama/Meta-Llama-3.1-8B"):
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_config(
        config=config,
    )
    orig_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, token=''
    )
    print(model)

    state_dict = orig_model.state_dict()
    new_state_dict = modify_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict, strict=False)

    prune_layers = []
    original_num_layers = orig_model.config.num_hidden_layers
    num_layers_to_prune = original_num_layers - model_config['num_hidden_layers']
    for i in range(1, num_layers_to_prune + 1):
        prune_layers.append(i)

    if (model_name == "meta-llama/Llama-2-7b-hf"):
        model.model.layers = torch.nn.ModuleList([layer for i, layer in enumerate(model.model.layers) if i not in prune_layers])
    elif (model_name == "google-bert/bert-base-uncased"):
        model.bert.encoder.layer = torch.nn.ModuleList([layer for i, layer in enumerate(model.bert.encoder.layer) if i not in prune_layers])
    else:
        model.model.layers = torch.nn.ModuleList([layer for i, layer in enumerate(model.model.layers) if i not in prune_layers])
    
    print(model)


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

    # Preprocess dataset
    sentence1_key, sentence2_key = task_to_keys[task_name]
    def preprocess_function(examples):
        texts = (examples[sentence1_key], examples[sentence2_key])
        result = tokenizer(*texts, padding=True, max_length=512, truncation=True)
        result["labels"] = examples["label"]
        return result
    
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    # Create data loaders
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]
    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=32)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=32)

    # Train model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # bert model needs to be finetuned for more epochs to get better results
    if model_name == "google-bert/bert-base-uncased":
        total_epoch = 15
    else:
        total_epoch = 5

    for epoch in range(total_epoch):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            batch.to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")

    # Evaluate model
    model.eval()
    metric = evaluate.load("glue", task_name)
    for batch in eval_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(predictions=predictions, references=labels)
    eval_metric = metric.compute()
    print(f"Evaluation Metric: {eval_metric}")
    return(eval_metric["accuracy"], model_size)

# finetune and evaluate vision (vit) models
def train_and_eval_vit(model_config, model_name):
    
    # get the model with the right shape 
    # weights are randomly initialized 
    config = ViTConfig.from_pretrained(model_name, 
                                       hidden_size=model_config['hidden_size'],
                                       intermediate_size=model_config['intermediate_size'],
                                        num_attention_heads=model_config['num_attn_heads'])
    model = ViTForImageClassification(config=config)

    # get the original model with pretrained weights
    orig_model= ViTForImageClassification.from_pretrained(
        model_name)

    # Load the weights from the pretrained model into the new model
    # dimensions in the back are pruned
    state_dict = orig_model.state_dict()
    new_state_dict = modify_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict, strict=False)

    # Prune layers from the front (keep layer 0)
    prune_layers = []
    original_num_layers = orig_model.config.num_hidden_layers
    num_layers_to_prune = original_num_layers - model_config['num_hidden_layers']
    for i in range(1, num_layers_to_prune + 1):
        prune_layers.append(i)

    model.vit.encoder.layer = torch.nn.ModuleList([layer for i, layer in enumerate(model.vit.encoder.layer) if i not in prune_layers])

    print(model)

    # calculate model size
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

    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    home_dir = os.getcwd()
    dataset_dir = os.path.join(home_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    train_dataset = datasets.CIFAR10(root=f"{dataset_dir}/cifar10_", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=f"{dataset_dir}/cifar10_", train=False, download=True, transform=transform)
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    # Move model to device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} GPUs")
    # Move model to device
    if device_count > 1:
        device = torch.device("cuda:0")
        model = nn.DataParallel(model, device_ids=list(range(device_count)))
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Define training loop
    def train(model, device, loader, optimizer):
        model.train()
        total_loss = 0
        with tqdm(loader, desc="Training") as pbar:
            for batch in pbar:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = torch.nn.CrossEntropyLoss()(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    # Define evaluation loop
    def evaluate(model, device, loader):
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.logits, dim=1)  # Extract logits from ImageClassifierOutput
                correct += (predicted == labels).sum().item()
        accuracy = correct / len(loader.dataset)
        return accuracy
    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(1):
        print(f"Epoch {epoch+1}")
        train_loss = train(model, device, train_loader, optimizer)
        # print(f"Train loss: {train_loss:.4f}")
    # Evaluate model on test set
    test_accuracy = evaluate(model, device, test_loader)
    print(f"Test accuracy: {test_accuracy:.4f}")

    return test_accuracy, model_size

# prune pretrained state_dict from the back of each dimension
def modify_state_dict(state_dict, model):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "embeddings" in key:
                attr_path = key.split('.')
                attr = model
                for path in attr_path:
                    attr = getattr(attr, path)
                if len(value.shape) == 3:  # 3D tensor 
                    new_value = value[:attr.shape[0], :attr.shape[1], :attr.shape[2]]
            elif 'weight' in key:
                # handle weights
                attr_path = key.split('.')
                attr = model
                for path in attr_path:
                    attr = getattr(attr, path)
                if len(value.shape) == 1:  # 1D tensor (e.g., linear layer bias)
                    new_value = value[:attr.shape[0]]
                elif len(value.shape) == 2:  # 2D tensor (e.g., linear layer weight)
                    new_value = value[:attr.shape[0], :attr.shape[1]]
                elif len(value.shape) == 3:  # 3D tensor 
                    new_value = value[:attr.shape[0], :attr.shape[1], :attr.shape[2]]
                elif len(value.shape) == 4:  # 4D tensor (e.g., convolutional layer weight)
                    new_value = value[:attr.shape[0], :attr.shape[1], :attr.shape[2], :attr.shape[3]]
                else:
                    raise ValueError(f"Unsupported weight shape: {value.shape}")
                
                new_state_dict[key] = new_value
            
            elif 'bias' in key:
                # handle biases
                attr_path = key.split('.')
                attr = model
                for path in attr_path:
                    attr = getattr(attr, path)
                if len(value.shape) == 1:  # 1D tensor (e.g., linear layer bias)
                    new_value = value[:attr.shape[0]]
                else:
                    raise ValueError(f"Unsupported bias shape: {value.shape}")
                
                new_state_dict[key] = new_value
            
            else:
                # other parameters (e.g., batch norm)
                new_state_dict[key] = value
        
        return new_state_dict
