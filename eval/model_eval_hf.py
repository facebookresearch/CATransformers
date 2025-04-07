# coding=utf-8
import evaluate
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
    get_scheduler,
)
from eval import model_constants

# Define task name and keys
task_name = "mrpc"
task_to_keys = {"mrpc": ("sentence1", "sentence2")}

def train_and_eval(model_config, model_name):

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
        token='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',
        hidden_size=model_config['hidden_size'],
        num_hidden_layers=model_config['num_hidden_layers'], 
        intermediate_size=model_config['intermediate_size'],
        num_attention_heads=model_config['num_attn_heads']
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token='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')
    if (model_name == "meta-llama/Llama-2-7b-hf"):
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_config(
        config=config,
    )
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
    eval_dataset = processed_datasets["validation"]
    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=32)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=32)

    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    for epoch in range(5):
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