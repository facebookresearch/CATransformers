import os
from pathlib import Path
import argparse
import datasets

import mobileclip
from torchvision import transforms
import torch
from clip_benchmark.datasets.builder import build_dataset
from clip_benchmark.metrics import zeroshot_classification as zsc
from clip_benchmark.datasets.builder import image_captions_collate_fn
from clip_benchmark.metrics import zeroshot_retrieval as zsr
import open_clip
import math
from torch.nn.utils import prune
import torch.nn as nn
import json

from transformers import CLIPProcessor, CLIPModel, CLIPConfig, CLIPVisionConfig,CLIPTextConfig
from eval.multiheaded_attention_custom import MultiheadAttentionSuper

text_ffn_ranking = []
vision_ffn_ranking = [] 
text_head_ranking = []
vision_head_ranking = []


def prune_model(model, transform, text_layer, text_embedding_dim, text_ffn_dim, text_head_num, vision_layer, vision_embedding_dim, vision_ffn_dim, vision_head_num, training=False):

    global text_ffn_ranking 
    global vision_ffn_ranking
    global text_head_ranking 
    global vision_head_ranking
    # Open the JSON file for reading

    mope_dir = os.path.join(Path(__file__).parent.absolute(), "mope/")
    text_ffn_path = os.path.join(mope_dir, "ranking_ffn_text.json")
    text_head_path = os.path.join(mope_dir, "ranking_head_text.json")
    vision_ffn_path = os.path.join(mope_dir, "ranking_ffn_vision.json")
    vision_head_path = os.path.join(mope_dir, "ranking_head_vision.json")
    

    with open(text_ffn_path, "r") as f:
        # Load the JSON data from the file
        text_ffn_ranking = json.load(f)
        # print(text_ffn_ranking)
    with open(vision_ffn_path, "r") as f:
    # Load the JSON data from the file
        vision_ffn_ranking = json.load(f)
        # print(vision_ffn_ranking)
    with open(text_head_path, "r") as f:
    # Load the JSON data from the file
        text_head_ranking = json.load(f)
        # print(text_head_ranking)
    with open(vision_head_path, "r") as f:
    # Load the JSON data from the file
        vision_head_ranking = json.load(f)
        # print(vision_head_ranking)

    if (text_ffn_dim != 1):
        model.transformer.resblocks = trim_ffn_mope(model.transformer.resblocks, text_ffn_dim, block_num=8, v_or_t='text')
    if (vision_ffn_dim != 1):
        model.visual.transformer.resblocks = trim_ffn_mope(model.visual.transformer.resblocks, vision_ffn_dim, block_num=8, v_or_t='vision')
    if(text_head_num != 1):
        model = trim_attn_head_mope(model, text_head_num, v_or_t='text', training=training)
    if(vision_head_num != 1):
        model.visual = trim_attn_head_mope(model.visual, vision_head_num, v_or_t='vision', training=training)
    if (text_embedding_dim != 1):
        model = trim_embed_text (model, text_embedding_dim, training=training)
    if (vision_embedding_dim != 1):
        model.visual = trim_embed_vision(model.visual, vision_embedding_dim, training=training)
    if (text_layer != 1):
        model.transformer.resblocks = trim_layers(model.transformer.resblocks, text_layer)
    if (vision_layer != 1):
        model.visual.transformer.resblocks = trim_layers_importance(model.visual.transformer.resblocks, vision_layer, v_or_t='vision')
    # if (text_ffn_dim != 1):
    #     model.transformer.resblocks = trim_ffn_back(model.transformer.resblocks, text_ffn_dim)
    # if (vision_ffn_dim != 1):
    #     model.visual.transformer.resblocks = trim_ffn_back(model.visual.transformer.resblocks, vision_ffn_dim)
    # if(text_head_num != 1):
    #     model = trim_num_heads_text(model, text_head_num, training=training)
    # if(vision_head_num != 1):
    #     model.visual = trim_num_heads_vision(model.visual, vision_head_num, training=training)
    # if (text_embedding_dim != 1):
    #     model = trim_embed_text (model, text_embedding_dim, training=training)
    # if (vision_embedding_dim != 1):
    #     model.visual = trim_embed_vision(model.visual, vision_embedding_dim, training=training)
    # if (text_layer != 1):
    #     model.transformer.resblocks = trim_layers(model.transformer.resblocks, text_layer)
    # if (vision_layer != 1):
    #     model.visual.transformer.resblocks = trim_layers(model.visual.transformer.resblocks, vision_layer)
    return model


def trim_layers(model, percentage_layers):
    num_layers = len(model)
    new_num_layers = int(num_layers * percentage_layers) #round down 

    # drop layers in the middle 
    front_layers = int(new_num_layers / 2)
    back_layers = int(0-(new_num_layers - front_layers))
    model = model[:front_layers] + model[back_layers:]
    return model

def trim_specific_layer(model, layer_idx):
    num_layers = len(model)
    print(f"number of layers = {num_layers}")
    new_model = model[:layer_idx] + model[layer_idx + 1:]
    return new_model


def trim_layers_importance(model, percentage_layers, v_or_t='text'):
    importance_scores = []
    src = torch.rand(1, 10, model[0].attn.out_proj.out_features) if v_or_t=='text' else torch.rand(1, 10, model[0].attn.out_proj.out_features)
    for i, m in enumerate(model):
        out = m(src)
        # mse = torch.mean((src - out) ** 2)
        # mae = torch.mean(torch.abs(src - out))
        cosine_similarity = torch.nn.functional.cosine_similarity(src, out).mean()
        importance_scores.append((i, cosine_similarity))
        src = out

    importance_scores = sorted(importance_scores, key=lambda x: x[1], reverse=True)
    


    num_layers = len(model)
    new_num_layers = int(num_layers * percentage_layers)
    remove_idx_list = []

    for i in range(int(num_layers - new_num_layers)):
        index, _ = importance_scores[i]
        remove_idx_list.append(index)

    # mask = torch.ones(num_layers, dtype=torch.bool)
    # mask[remove_idx_list] = False
    # print(mask)

    # model = model[mask]

    remove_idx_list = sorted(remove_idx_list, reverse=True)
    print(remove_idx_list)
    for indx in remove_idx_list:

        # checking whether the corresponding iterator index is less than the list length
        if indx < num_layers:

            # removing element by index using pop() function
            model = model[:indx] + model[indx+1:]
    # # drop layers in the middle 
    # front_layers = math.ceil(new_num_layers / 2)
    # back_layers = int(0-(new_num_layers - front_layers))
    # model = model[:front_layers] + model[back_layers:]
    return model


def trim_embed_vision(model, percentage_embed, training=False):
    
    embed_dim = model.conv1.out_channels
        
    new_embed_dim = int(embed_dim * percentage_embed)

    model.positional_embedding.embedding_dim = new_embed_dim
    model.positional_embedding.data = nn.Parameter(model.positional_embedding.data[:, :new_embed_dim])

    model.class_embedding.embedding_dim = new_embed_dim
    model.class_embedding.data = nn.Parameter(model.class_embedding.data[:new_embed_dim])


    model.conv1.out_channels = new_embed_dim
    model.conv1.weight = nn.Parameter(model.conv1.weight[:new_embed_dim, :])
    # model.conv1.bias = nn.Parameter(model.conv1.bias[:new_embed_dim])
    
    model.ln_pre.normalized_shape = (new_embed_dim,)
    model.ln_pre.weight = nn.Parameter(model.ln_pre.weight[:new_embed_dim])
    model.ln_pre.bias = nn.Parameter(model.ln_pre.bias[:new_embed_dim])

    model.ln_post.normalized_shape = (new_embed_dim,)
    model.ln_post.weight = nn.Parameter(model.ln_post.weight[:new_embed_dim])
    model.ln_post.bias = nn.Parameter(model.ln_post.bias[:new_embed_dim])

    model.proj.data = nn.Parameter(model.proj.data[:new_embed_dim, :])

    for m in model.transformer.resblocks:
        m.ln_1.normalized_shape = (new_embed_dim,)
        m.ln_1.weight = nn.Parameter(m.ln_1.weight[:new_embed_dim])
        m.ln_1.bias = nn.Parameter(m.ln_1.bias[:new_embed_dim])
        qkv_dim = m.attn.out_proj.in_features 

        # create a new MultiheadAttention module with the desired embedding dimension
        new_multihead_attention = MultiheadAttentionSuper(super_embed_dim=new_embed_dim, is_encoder=True, num_heads=m.attn.num_heads, qkv_dim=qkv_dim, batchFirst=training)

        # new_multihead_attention.load_state_dict(m.attn.state_dict())
        new_multihead_attention.in_proj_weight = nn.Parameter(m.attn.in_proj_weight.data[:, :new_embed_dim])
        new_multihead_attention.in_proj_bias = nn.Parameter(m.attn.in_proj_bias.data[:])

        new_multihead_attention.out_proj.out_features = new_embed_dim
        new_multihead_attention.out_proj.weight = nn.Parameter(m.attn.out_proj.weight[:new_embed_dim, :])
        new_multihead_attention.out_proj.bias = nn.Parameter(m.attn.out_proj.bias[:new_embed_dim])

        m.attn.embed_dim = new_embed_dim
        old_attn = m.attn 
        m.attn = new_multihead_attention
        del old_attn

        m.ln_2.normalized_shape = (new_embed_dim,)
        m.ln_2.weight = nn.Parameter(m.ln_2.weight[:new_embed_dim])
        m.ln_2.bias = nn.Parameter(m.ln_2.bias[:new_embed_dim])


        m.mlp.c_fc.in_features = new_embed_dim
        m.mlp.c_fc.weight = nn.Parameter(m.mlp.c_fc.weight[:, :new_embed_dim])

        m.mlp.c_proj.out_features = new_embed_dim
        m.mlp.c_proj.weight = nn.Parameter(m.mlp.c_proj.weight[:new_embed_dim, :])
        m.mlp.c_proj.bias = nn.Parameter(m.mlp.c_proj.bias[:new_embed_dim])

    return model

def trim_embed_text(model, percentage_embed, training=False):
    embed_dim = model.token_embedding.embedding_dim 
        
    new_embed_dim = int(model.token_embedding.embedding_dim * percentage_embed)

    model.positional_embedding.embedding_dim = new_embed_dim
    model.positional_embedding.data = nn.Parameter(model.positional_embedding.data[:, :new_embed_dim])

    model.token_embedding.embedding_dim = new_embed_dim
    model.token_embedding.weight = nn.Parameter(model.token_embedding.weight[:, :new_embed_dim])

    model.ln_final.normalized_shape = (new_embed_dim,)
    model.ln_final.weight = nn.Parameter(model.ln_final.weight[:new_embed_dim])
    model.ln_final.bias = nn.Parameter(model.ln_final.bias[:new_embed_dim])

    model.text_projection.data = nn.Parameter(model.text_projection.data[:new_embed_dim, :])
    # model.visual.proj.data = nn.Parameter(model.visual.proj.data[:, :new_embed_dim])

    for m in model.transformer.resblocks:
        m.ln_1.normalized_shape = (new_embed_dim,)
        m.ln_1.weight = nn.Parameter(m.ln_1.weight[:new_embed_dim])
        m.ln_1.bias = nn.Parameter(m.ln_1.bias[:new_embed_dim])
        qkv_dim = m.attn.out_proj.in_features

        # create a new MultiheadAttention module with the desired embedding dimension
        # new_multihead_attention = nn.MultiheadAttention(embed_dim=512, num_heads=m.attn.num_heads)
        new_multihead_attention = MultiheadAttentionSuper(super_embed_dim=new_embed_dim, is_encoder=True, num_heads=m.attn.num_heads, qkv_dim=qkv_dim, batchFirst=training)
        head_dim = int(embed_dim / m.attn.num_heads)
        new_head_dim = int(new_embed_dim / m.attn.num_heads)

        # new_multihead_attention.load_state_dict(m.attn.state_dict())
        new_multihead_attention.in_proj_weight = nn.Parameter(m.attn.in_proj_weight.data[:, :new_embed_dim])
        new_multihead_attention.in_proj_bias = nn.Parameter(m.attn.in_proj_bias.data[:])

        new_multihead_attention.out_proj.out_features = new_embed_dim
        new_multihead_attention.out_proj.weight = nn.Parameter(m.attn.out_proj.weight[:new_embed_dim, :])
        new_multihead_attention.out_proj.bias = nn.Parameter(m.attn.out_proj.bias[:new_embed_dim])

        m.attn.embed_dim = new_embed_dim
        old_attn = m.attn 
        m.attn = new_multihead_attention
        del old_attn

        m.ln_2.normalized_shape = (new_embed_dim,)
        m.ln_2.weight = nn.Parameter(m.ln_2.weight[:new_embed_dim])
        m.ln_2.bias = nn.Parameter(m.ln_2.bias[:new_embed_dim])

        m.mlp.c_fc.in_features = new_embed_dim
        m.mlp.c_fc.weight = nn.Parameter(m.mlp.c_fc.weight[:, :new_embed_dim])

        m.mlp.c_proj.out_features = new_embed_dim
        m.mlp.c_proj.weight = nn.Parameter(m.mlp.c_proj.weight[:new_embed_dim, :])
        m.mlp.c_proj.bias = nn.Parameter(m.mlp.c_proj.bias[:new_embed_dim])

    return model

def trim_ffn_back(model, percentage_ffn):

    for m in model:

        prune_amount = 1.0 - percentage_ffn
        m.mlp.c_fc = prune.ln_structured(m.mlp.c_fc, 'weight', amount=prune_amount, dim=0, n=float('-inf'))
        mask = m.mlp.c_fc.weight_mask
        # # Apply the same mask to the bias term
        # m.mlp.c_fc.bias.data[mask[:, 0] == 0] = 0
        
        prune.remove(m.mlp.c_fc, 'weight')

        # Create a boolean mask to select non-zero rows
        non_zero_rows = torch.sum(mask, dim=1) > 0

        # Apply the masks to the weight tensor
        out_features = int(m.mlp.c_fc.out_features * percentage_ffn)
        m.mlp.c_fc.out_features = out_features
        m.mlp.c_fc.weight = nn.Parameter(m.mlp.c_fc.weight[non_zero_rows, :])
        m.mlp.c_fc.bias = nn.Parameter(m.mlp.c_fc.bias[non_zero_rows])

        m.mlp.c_proj.in_features = out_features
        m.mlp.c_proj.weight = nn.Parameter(m.mlp.c_proj.weight[:, non_zero_rows])


        
        # out_features = int(m.mlp.c_fc.out_features * percentage_ffn)
        # m.mlp.c_fc.out_features = out_features
        # m.mlp.c_fc.weight = nn.Parameter(m.mlp.c_fc.weight[:out_features, :])
        # m.mlp.c_fc.bias = nn.Parameter(m.mlp.c_fc.bias[:out_features])

        # m.mlp.c_proj.in_features = out_features
        # m.mlp.c_proj.weight = nn.Parameter(m.mlp.c_proj.weight[:, :out_features])
    return model


def trim_num_heads_text(model, percentage_head, training=False):
    
    for m in model.transformer.resblocks:
        embed_dim = m.attn.out_proj.in_features 
        num_attn_heads = m.attn.num_heads
        new_attn_heads = int(num_attn_heads * percentage_head)
        head_dim = int(embed_dim / num_attn_heads)
        new_qkv_dim = int(new_attn_heads * head_dim)
        

        # create a new MultiheadAttention module with the desired embedding dimension
        # new_multihead_attention = nn.MultiheadAttention(embed_dim=512, num_heads=m.attn.num_heads)
        new_multihead_attention = MultiheadAttentionSuper(super_embed_dim=embed_dim, is_encoder=True, num_heads=new_attn_heads, qkv_dim=new_qkv_dim,  batchFirst=training)


        # new_multihead_attention.load_state_dict(m.attn.state_dict())

        q_in_weight = m.attn.in_proj_weight.data[:new_qkv_dim, :].clone()
        k_in_weight = m.attn.in_proj_weight.data[embed_dim:embed_dim + new_qkv_dim, :].clone()
        v_in_weight = m.attn.in_proj_weight.data[embed_dim*2:embed_dim*2 + new_qkv_dim, :].clone()
        new_multihead_attention.in_proj_weight.data = torch.cat((q_in_weight, k_in_weight, v_in_weight), axis=0)


        new_multihead_attention.in_proj_bias = nn.Parameter(m.attn.in_proj_bias.data[:])

        q_in_weight = m.attn.in_proj_bias.data[:new_qkv_dim].clone()
        k_in_weight = m.attn.in_proj_bias.data[embed_dim:embed_dim + new_qkv_dim].clone()
        v_in_weight = m.attn.in_proj_bias.data[embed_dim*2:embed_dim*2 + new_qkv_dim].clone()
        new_multihead_attention.in_proj_bias.data = torch.cat((q_in_weight, k_in_weight, v_in_weight), axis=0)

        new_multihead_attention.out_proj.in_features = new_qkv_dim
        new_multihead_attention.out_proj.weight = nn.Parameter(m.attn.out_proj.weight[:, :new_qkv_dim])
        new_multihead_attention.out_proj.bias = nn.Parameter(m.attn.out_proj.bias[:])

        m.attn.num_heads = new_attn_heads
        m.attn = new_multihead_attention


    return model

def trim_num_heads_vision(model, percentage_head, training=False):
    

    for m in model.transformer.resblocks:
        embed_dim = m.attn.out_proj.in_features 
        num_attn_heads = m.attn.num_heads
        new_attn_heads = int(num_attn_heads * percentage_head)
        head_dim = int(embed_dim / num_attn_heads)
        new_qkv_dim = int(new_attn_heads * head_dim)
        

        # create a new MultiheadAttention module with the desired embedding dimension
        # new_multihead_attention = nn.MultiheadAttention(embed_dim=512, num_heads=m.attn.num_heads)
        new_multihead_attention = MultiheadAttentionSuper(super_embed_dim=embed_dim, is_encoder=True, num_heads=new_attn_heads, qkv_dim=new_qkv_dim,  batchFirst=training)


        # new_multihead_attention.load_state_dict(m.attn.state_dict())

        q_in_weight = m.attn.in_proj_weight.data[:new_qkv_dim, :].clone()
        k_in_weight = m.attn.in_proj_weight.data[embed_dim:embed_dim + new_qkv_dim, :].clone()
        v_in_weight = m.attn.in_proj_weight.data[embed_dim*2:embed_dim*2 + new_qkv_dim, :].clone()
        new_multihead_attention.in_proj_weight.data = torch.cat((q_in_weight, k_in_weight, v_in_weight), axis=0)


        new_multihead_attention.in_proj_bias = nn.Parameter(m.attn.in_proj_bias.data[:])

        q_in_weight = m.attn.in_proj_bias.data[:new_qkv_dim].clone()
        k_in_weight = m.attn.in_proj_bias.data[embed_dim:embed_dim + new_qkv_dim].clone()
        v_in_weight = m.attn.in_proj_bias.data[embed_dim*2:embed_dim*2 + new_qkv_dim].clone()
        new_multihead_attention.in_proj_bias.data = torch.cat((q_in_weight, k_in_weight, v_in_weight), axis=0)

        new_multihead_attention.out_proj.in_features = new_qkv_dim
        new_multihead_attention.out_proj.weight = nn.Parameter(m.attn.out_proj.weight[:, :new_qkv_dim])
        new_multihead_attention.out_proj.bias = nn.Parameter(m.attn.out_proj.bias[:])

        m.attn.num_heads = new_attn_heads
        m.attn = new_multihead_attention


    return model



def trim_ffn_idx(mlp, ffn_idx, block_num=8):
    block_size = int(mlp.c_fc.out_features/block_num)
    mlp.c_fc.out_features
    start_idx = ffn_idx * block_size
    end_index = start_idx + block_size

    # mlp.c_fc = prune.ln_structured(m.mlp.c_fc, 'weight', amount=prune_amount, dim=0, n=float('-inf'))
    # mask = mlp.c_fc.weight_mask
    # # Apply the same mask to the bias term
    # m.mlp.c_fc.bias.data[mask[:, 0] == 0] = 0
    
    # prune.remove(mlp.c_fc, 'weight')

    # Create a boolean mask to select non-zero rows
    # non_zero_rows = torch.sum(mask, dim=1) > 0

    mask = torch.ones(mlp.c_fc.out_features, dtype=torch.bool)
    mask[start_idx:end_index] = False

    # Apply the masks to the weight tensor
    out_features = int(mlp.c_fc.out_features  -  block_size)
    mlp.c_fc.out_features = out_features

    mlp.c_fc.weight = nn.Parameter(mlp.c_fc.weight[mask, :])
    mlp.c_fc.bias = nn.Parameter(mlp.c_fc.bias[mask])

    mlp.c_proj.in_features = out_features
    mlp.c_proj.weight = nn.Parameter(mlp.c_proj.weight[:, mask])

    return mlp

def trim_ffn_mope(model, percentage_ffn, block_num=8, v_or_t='text'):
    num_blocks_to_prune = int (block_num * (1-percentage_ffn))

    ranking_list = text_ffn_ranking if v_or_t == "text" else vision_ffn_ranking

    for layer_num, m in enumerate(model):
        _, importance_scores = ranking_list[layer_num]
        remove_idx_list = []

        for i in range(num_blocks_to_prune):
            index, _ = importance_scores[i]
            remove_idx_list.append(index)

        block_size = int(m.mlp.c_fc.out_features/block_num)
        mask = torch.ones(m.mlp.c_fc.out_features, dtype=torch.bool)
        for ffn_idx in remove_idx_list:
            start_idx = ffn_idx * block_size
            end_index = start_idx + block_size
            mask[start_idx:end_index] = False

        # Apply the masks to the weight tensor
        out_features = int(m.mlp.c_fc.out_features  -  num_blocks_to_prune * block_size)
        m.mlp.c_fc.out_features = out_features

        m.mlp.c_fc.weight = nn.Parameter(m.mlp.c_fc.weight[mask, :])
        m.mlp.c_fc.bias = nn.Parameter(m.mlp.c_fc.bias[mask])

        m.mlp.c_proj.in_features = out_features
        m.mlp.c_proj.weight = nn.Parameter(m.mlp.c_proj.weight[:, mask])
    return model


def trim_attn_head_idx(attn, head_idx):

    embed_dim = attn.out_proj.in_features 
    num_attn_heads = attn.num_heads
    new_attn_heads = num_attn_heads - 1
    head_dim = int(embed_dim / num_attn_heads)
    new_qkv_dim = int(new_attn_heads * head_dim)


    start_idx = head_idx * head_dim
    end_index = start_idx + head_dim

    mask = torch.ones(embed_dim, dtype=torch.bool)
    mask[start_idx:end_index] = False
    

    # create a new MultiheadAttention module with the desired embedding dimension
    # new_multihead_attention = nn.MultiheadAttention(embed_dim=512, num_heads=m.attn.num_heads)
    new_multihead_attention = MultiheadAttentionSuper(super_embed_dim=embed_dim, is_encoder=True, num_heads=new_attn_heads, qkv_dim=new_qkv_dim, batchFirst=False)


    # new_multihead_attention.load_state_dict(m.attn.state_dict())

    q_in_weight = attn.in_proj_weight.data[:embed_dim, :]
    q_in_weight = q_in_weight[mask, :]
    k_in_weight = attn.in_proj_weight.data[embed_dim:embed_dim*2, :]
    k_in_weight = k_in_weight[mask, :]
    v_in_weight = attn.in_proj_weight.data[embed_dim*2:embed_dim*3, :]
    v_in_weight = v_in_weight[mask, :]

    new_multihead_attention.in_proj_weight.data = torch.cat((q_in_weight, k_in_weight, v_in_weight), axis=0)


    new_multihead_attention.in_proj_bias = nn.Parameter(attn.in_proj_bias.data[:])

    q_in_weight = attn.in_proj_bias.data[:embed_dim]
    q_in_weight = q_in_weight[mask]
    k_in_weight = attn.in_proj_bias.data[embed_dim:embed_dim*2 ]
    k_in_weight = k_in_weight[mask]
    v_in_weight = attn.in_proj_bias.data[embed_dim*2:embed_dim*3]
    v_in_weight = v_in_weight[mask]
    new_multihead_attention.in_proj_bias.data = torch.cat((q_in_weight, k_in_weight, v_in_weight), axis=0)

    new_multihead_attention.out_proj.in_features = new_qkv_dim
    new_multihead_attention.out_proj.weight = nn.Parameter(attn.out_proj.weight[:, mask])
    new_multihead_attention.out_proj.bias = nn.Parameter(attn.out_proj.bias[:])

    attn.num_heads = new_attn_heads
    attn = new_multihead_attention
    return attn


def trim_attn_head_mope(model, percentage_head, v_or_t='text', training=False):

    ranking_list = text_head_ranking if v_or_t == "text" else vision_head_ranking

    for layer_num, m in enumerate(model.transformer.resblocks):
        embed_dim = m.attn.out_proj.in_features 
        num_attn_heads = m.attn.num_heads
        new_attn_heads = int(num_attn_heads * percentage_head)
        head_dim = int(embed_dim / num_attn_heads)
        new_qkv_dim = int(new_attn_heads * head_dim)


        num_heads_to_prune = num_attn_heads - new_attn_heads 
        _, importance_scores = ranking_list[layer_num]
        remove_idx_list = []

        for i in range(num_heads_to_prune):
            index, _ = importance_scores[i]
            remove_idx_list.append(index)

        mask = torch.ones(embed_dim, dtype=torch.bool)
        for head_idx in remove_idx_list:
            start_idx = head_idx * head_dim
            end_index = start_idx + head_dim
            mask[start_idx:end_index] = False
        


        new_multihead_attention = MultiheadAttentionSuper(super_embed_dim=embed_dim, is_encoder=True, num_heads=new_attn_heads, qkv_dim=new_qkv_dim, batchFirst=training)


        # new_multihead_attention.load_state_dict(m.attn.state_dict())

        q_in_weight = m.attn.in_proj_weight.data[:embed_dim, :]
        q_in_weight = q_in_weight[mask, :]
        k_in_weight = m.attn.in_proj_weight.data[embed_dim:embed_dim*2, :]
        k_in_weight = k_in_weight[mask, :]
        v_in_weight = m.attn.in_proj_weight.data[embed_dim*2:embed_dim*3, :]
        v_in_weight = v_in_weight[mask, :]

        new_multihead_attention.in_proj_weight.data = torch.cat((q_in_weight, k_in_weight, v_in_weight), axis=0)


        new_multihead_attention.in_proj_bias = nn.Parameter(m.attn.in_proj_bias.data[:])

        q_in_weight = m.attn.in_proj_bias.data[:embed_dim]
        q_in_weight = q_in_weight[mask]
        k_in_weight = m.attn.in_proj_bias.data[embed_dim:embed_dim*2 ]
        k_in_weight = k_in_weight[mask]
        v_in_weight = m.attn.in_proj_bias.data[embed_dim*2:embed_dim*3]
        v_in_weight = v_in_weight[mask]
        new_multihead_attention.in_proj_bias.data = torch.cat((q_in_weight, k_in_weight, v_in_weight), axis=0)

        new_multihead_attention.out_proj.in_features = new_qkv_dim
        new_multihead_attention.out_proj.weight = nn.Parameter(m.attn.out_proj.weight[:, mask])
        new_multihead_attention.out_proj.bias = nn.Parameter(m.attn.out_proj.bias[:])

        m.attn.num_heads = new_attn_heads
        m.attn = new_multihead_attention
    return model