"""
In this file we initialize the required files to run MoPE-CLIP based pruning for the number of heads and FFN dimension 
only needs to be ran one for each model architecture
"""

from pruning import trim_attn_head_idx, trim_ffn_idx
from model_eval import eval_retreival, eval_zeroShotClassification
from configurations import MODEL_ARCH, PRETRAINED
import csv, json, os
import torch
import open_clip
import copy

text_ffn_ranking = []
vision_ffn_ranking = [] 
text_head_ranking = []
vision_head_ranking = []

def init_mope(model_arch='ViT-B-16', pretrained='datacomp_xl_s13b_b90k'):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model, _, transform = open_clip.create_model_and_transforms(model_arch, pretrained=pretrained)
    model.eval()
    print(model)
    model = model.to(device)
    home_dir = os.getcwd()
    save_location = f"{home_dir}/eval/mope/{model_arch}_{pretrained}"

    # for every layer in text model, evaluate it's importance (12)
    rank_attn_heads(model, transform, 'text', save_location)
    rank_attn_heads(model, transform, 'vision', save_location)
    rank_ffn_dim(model, transform, 'text', save_location)
    rank_ffn_dim(model, transform, 'vision', save_location)

    return text_ffn_ranking, vision_ffn_ranking, text_head_ranking, vision_head_ranking

def rank_attn_heads(model, transform,  v_or_t="text", save_location="eval/mope"):
    layer_blocks = model.transformer.resblocks if v_or_t == "text" else model.visual.transformer.resblocks
    ranking_list = text_head_ranking if v_or_t == "text" else vision_head_ranking
    importance_scores = []
    num_layer = len(layer_blocks)

    filename = f"{save_location}/ranking_head_{v_or_t}.csv"
    print(f"saving to {filename}")
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Check if the file is empty
        if csvfile.tell() == 0:
            # Write the header row
            writer.writerow(['Text or Vision', 'Layer Number', 'Head Index Number', 'Mean_accuracy', 'mean_recall@1', 'text_retrieval_recall@1', 'image_retrieval_recall@1', 'ImageNet Acc1'])
        
        for i, m in enumerate(layer_blocks):
            num_attn_heads = m.attn.num_heads
            original_attn = m.attn
            importance_scores = []
            for head_idx in range(num_attn_heads):
                m.attn = trim_attn_head_idx(copy.deepcopy(original_attn), head_idx)
                # evaluate the pruned model 
                accuracy_retreival,  accuracy_zeroShot = evaluate_model(model, transform)
                average_acc = (accuracy_retreival['mean_recall@1'] + accuracy_zeroShot['acc1']) / 2
                writer.writerow([v_or_t, i, head_idx, average_acc, accuracy_retreival['mean_recall@1'], 
                                 accuracy_retreival['text_retrieval_recall@1'], accuracy_retreival['image_retrieval_recall@1'], 
                                 accuracy_zeroShot['acc1']])
                #rank 
                importance_scores.append((head_idx, average_acc))
            importance_scores = sorted(importance_scores, key=lambda x: x[1], reverse=True)
            print(importance_scores)
            ranking_list.append((i, importance_scores))
            # make sure, the full attention head is back for the layer
            m.attn = original_attn

    with open(f"{save_location}/ranking_head_{v_or_t}.json", "w") as f:
    # Write the list to the JSON file
        json.dump(ranking_list, f)

def rank_ffn_dim(model, transform,  v_or_t="text", save_location="eval/mope"):
    num_blocks = 8 # fix at 12.5 search granularity
    layer_blocks = model.transformer.resblocks if v_or_t == "text" else model.visual.transformer.resblocks
    ranking_list = text_ffn_ranking if v_or_t == "text" else vision_ffn_ranking
    
    importance_scores = []
    filename = f"{save_location}/ranking_ffn_{v_or_t}.csv"
    print(f"saving to {filename}")
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Check if the file is empty
        if csvfile.tell() == 0:
            # Write the header row
            writer.writerow(['Text or Vision', 'Layer Number', 'Block Index Number', 'Mean_accuracy', 'mean_recall@1', 'text_retrieval_recall@1', 'image_retrieval_recall@1', 'ImageNet Acc1'])
        
        for i, m in enumerate(layer_blocks):
            original_mlp = m.mlp
            importance_scores = []
            for ffn_idx in range(num_blocks):
                m.mlp = trim_ffn_idx(copy.deepcopy(original_mlp), ffn_idx, block_num=num_blocks)
                # evaluate the pruned model 
                accuracy_retreival,  accuracy_zeroShot = evaluate_model(model, transform)
                average_acc = (accuracy_retreival['mean_recall@1'] + accuracy_zeroShot['acc1']) / 2
                writer.writerow([v_or_t, i, ffn_idx, average_acc, accuracy_retreival['mean_recall@1'], 
                                 accuracy_retreival['text_retrieval_recall@1'], accuracy_retreival['image_retrieval_recall@1'], 
                                 accuracy_zeroShot['acc1']])
                #rank 
                importance_scores.append((ffn_idx, average_acc))
            importance_scores = sorted(importance_scores, key=lambda x: x[1], reverse=True)
            ranking_list.append((i, importance_scores))
            print(importance_scores)
            # make sure, the full attention head is back for the layer
            m.mlp = original_mlp
    with open(f"{save_location}/ranking_ffn_{v_or_t}.json", "w") as f:
    # Write the list to the JSON file
        json.dump(ranking_list, f)
            
        

def evaluate_model(model, transform):
    print(model)
    accuracy_retreival = eval_retreival("mscoco_captions", model, transform)
    accuracy_zeroShot = eval_zeroShotClassification("ImageNet1k", model, transform)
    return accuracy_retreival, accuracy_zeroShot


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model_arch=MODEL_ARCH
    pretrained=PRETRAINED
    model, _, transform = open_clip.create_model_and_transforms(model_arch, pretrained=pretrained)
    model.eval()
    print(model)
    model = model.to(device)
    home_dir = os.getcwd()
    save_location = f"{home_dir}/eval/mope/{model_arch}_{pretrained}"
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    rank_ffn_dim(model, transform, 'text', save_location=save_location)
    rank_attn_heads(model, transform, 'text', save_location=save_location)
    rank_attn_heads(model, transform, 'vision', save_location=save_location)
    rank_ffn_dim(model, transform, 'vision', save_location=save_location)




