ATTN_HEAD_DIM = 64
MAX_EPOCH = 10
text_context_length = 77
vision_context_length = 197 #(B16)

vit_b_16 = {"model":"ViT-B-16", "text_layer": 12, "text_embedding_dim": 512, "text_ffn_dim":2048, "text_head_num":8, "vision_layer":12, "vision_embedding_dim":768, "vision_ffn_dim":3072, "vision_head_num":12 }
vit_b_16_gelu = {"model":"ViT-B-16-quickgelu", "text_layer": 12, "text_embedding_dim": 512, "text_ffn_dim":2048, "text_head_num":8, "vision_layer":12, "vision_embedding_dim":768, "vision_ffn_dim":3072, "vision_head_num":12 }
vit_b_32 = {"model":"ViT-B-32", "text_layer": 12, "text_embedding_dim": 512, "text_ffn_dim":2048, "text_head_num":8, "vision_layer":12, "vision_embedding_dim":768, "vision_ffn_dim":3072, "vision_head_num":12 }
orig_models ={"ViT-B-16": vit_b_16, "ViT-B-16-quickgelu": vit_b_16_gelu, "ViT-B-32": vit_b_32}

# Calcuate the number of FLOPs for a model
def calculate_flop(text_layer, text_embedding_dim, text_ffn_dim, text_head_num, vision_layer, vision_embedding_dim, vision_ffn_dim, vision_head_num):
    # Model flops ~ 6LMA+ 2LTA + 2LAM + 4LMF = 4LM(2A + F) + 2LTA
    # N = 2LM(2A + F)
    # Hence flops = 2N + 2LTA
    # A = Num_heads * 64
    text_model_FLOPS = 4* text_layer* text_embedding_dim * ((2*text_head_num*ATTN_HEAD_DIM)+ text_ffn_dim) + (2* text_context_length*text_head_num*ATTN_HEAD_DIM)
    vision_model_FLOPS = 4* vision_layer* vision_embedding_dim * ((2*vision_head_num*ATTN_HEAD_DIM)+ vision_ffn_dim) + (2* vision_context_length*vision_head_num*ATTN_HEAD_DIM)
    flops = text_model_FLOPS + vision_model_FLOPS
    return flops