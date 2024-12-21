from pruning import trim_attn_head_idx, trim_ffn_idx
from model_eval import eval_retreival, eval_zeroShotClassification
import csv, json
import torch
import open_clip_custom
import copy

text_ffn_ranking = []
vision_ffn_ranking = [] 
text_head_ranking = []
vision_head_ranking = []

def init_importance(model, transform):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model, _, transform = open_clip_custom.create_model_and_transforms('ViT-B-16', pretrained='datacomp_xl_s13b_b90k')
    model.eval()
    print(model)
    model = model.to(device)


    # for every layer in text model, evaluate it's importance (12)
    rank_attn_heads(model, transform, 'text')
    rank_attn_heads(model, transform, 'vision')
    rank_ffn_dim(model, transform, 'text')
    rank_ffn_dim(model, transform, 'vision')

    return text_ffn_ranking, vision_ffn_ranking, text_head_ranking, vision_head_ranking

# def rank_text_layers(model): 
#     importance_scores = []
#     num_layer = len(model.transformer.resblocks)
#     original_resblock = model.transformer.resblocks
#     for i in num_layer:
#         model.transformer.resblocks = pruning.trim_specific_layer(original_resblock, i)
#         # evaluate the pruned model 
#         result = evaluate_model(model)
#         #rank 
#         importance_scores.append((i, result))
#     importance_scores = sorted(importance_scores, key=lambda x: x[1], reverse=True)
#     text_layer_ranking = importance_scores

# def rank_vision_layers(model): 
#     importance_scores = []
#     num_layer = len(model.visual.transformer.resblocks)
#     original_resblock = model.visual.transformer.resblocks
#     for i in num_layer:
#         model.visual.transformer.resblocks = pruning.trim_specific_layer(original_resblock, i)
#         # evaluate the pruned model 
#         result = evaluate_model(model)
#         #rank 
#         importance_scores.append((i, result))
#     importance_scores = sorted(importance_scores, key=lambda x: x[1], reverse=True)
#     vision_layer_ranking = importance_scores

def rank_attn_heads(model, transform,  v_or_t="text"):
    layer_blocks = model.transformer.resblocks if v_or_t == "text" else model.visual.transformer.resblocks
    ranking_list = text_head_ranking if v_or_t == "text" else vision_head_ranking
    importance_scores = []
    num_layer = len(layer_blocks)

    filename = f"ranking_head_{v_or_t}.csv"
    print(f"saving to {filename}")
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Check if the file is empty
        if csvfile.tell() == 0:
            # Write the header row
            writer.writerow(['Text or Vision', 'Layer Number', 'Head Index Number', 'Mean_accuracy', 'mean_recall@1', 'text_retrieval_recall@1', 'image_retrieval_recall@1', 'ImageNet Acc1'])
        
        for i, m in enumerate(layer_blocks):
            if i < 11:
                continue
            num_attn_heads = m.attn.num_heads
            original_attn = m.attn
            importance_scores = []
            for head_idx in range(num_attn_heads):
                if head_idx < 10:
                    continue
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

    with open(f"ranking_head_{v_or_t}.json", "w") as f:
    # Write the list to the JSON file
        json.dump(ranking_list, f)

def rank_ffn_dim(model, transform,  v_or_t="text"):
    num_blocks = 8 # fix at 12.5 search granularity
    layer_blocks = model.transformer.resblocks if v_or_t == "text" else model.visual.transformer.resblocks
    ranking_list = text_ffn_ranking if v_or_t == "text" else vision_ffn_ranking
    
    importance_scores = []
    filename = f"ranking_ffn_{v_or_t}.csv"
    print(f"saving to {filename}")
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Check if the file is empty
        if csvfile.tell() == 0:
            # Write the header row
            writer.writerow(['Text or Vision', 'Layer Number', 'Block Index Number', 'Mean_accuracy', 'mean_recall@1', 'text_retrieval_recall@1', 'image_retrieval_recall@1', 'ImageNet Acc1'])
        
        for i, m in enumerate(layer_blocks):
            if i != 7:
                continue
            original_mlp = m.mlp
            importance_scores = []
            for ffn_idx in range(num_blocks):
                if ffn_idx < 5:
                    continue
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
    with open(f"ranking_ffn_{v_or_t}.json", "w") as f:
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
    model, _, transform = open_clip_custom.create_model_and_transforms('ViT-B-16', pretrained='datacomp_xl_s13b_b90k')
    model.eval()
    print(model)
    model = model.to(device)

    rank_ffn_dim(model, transform, 'text')
    rank_attn_heads(model, transform, 'text')
    rank_attn_heads(model, transform, 'vision')
    rank_ffn_dim(model, transform, 'vision')




