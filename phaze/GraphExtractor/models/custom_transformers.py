import torch
import torch.nn as nn
from ..utils import ShapeProp, PhazeGraph
from .model import BaseModelIR
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# class Transformer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward, dropout, num_layers):
#         super(Transformer, self).__init__()
#         self.encoder = TransformerEncoder(d_model, nhead, dim_feedforward, dropout, num_layers)
#         self.decoder = TransformerDecoder(d_model, nhead, dim_feedforward, dropout, num_layers)

#     def forward(self, src, tgt):
#         memory = self.encoder(src)
#         output = self.decoder(tgt, memory)
#         return output

# class TransformerEncoder(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward, dropout, num_layers):
#         super(TransformerEncoder, self).__init__()
#         self.layers = nn.ModuleList()
#         for _ in range(num_layers):
#             self.layers.append(TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout))

#     def forward(self, src):
#         output = src
#         for layer in self.layers:
#             output = layer(output)
#         return output

# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward, dropout):
#         super(TransformerEncoderLayer, self).__init__()
#         self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
#         self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)

#     def forward(self, x):
#         x = self.self_attn(x, x, x)
#         x = self.feed_forward(x)
#         return x

# class TransformerDecoder(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward, dropout, num_layers):
#         super(TransformerDecoder, self).__init__()
#         self.layers = nn.ModuleList()
#         for _ in range(num_layers):
#             self.layers.append(TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout))

#     def forward(self, tgt, memory):
#         output = tgt
#         for layer in self.layers:
#             output = layer(output, memory)
#         return output

# class TransformerDecoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward, dropout):
#         super(TransformerDecoderLayer, self).__init__()
#         self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
#         self.encoder_attn = MultiHeadAttention(d_model, nhead, dropout)
#         self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)

#     def forward(self, tgt, memory):
#         tgt = self.self_attn(tgt, tgt, tgt)
#         tgt = self.encoder_attn(tgt, memory, memory)
#         tgt = self.feed_forward(tgt)
#         return tgt

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, nhead, dropout):
#         super(MultiHeadAttention, self).__init__()
#         self.d_model = d_model
#         self.nhead = nhead
#         self.dropout = dropout

#         self.linear_q = nn.Linear(d_model, nhead * d_model)
#         self.linear_k = nn.Linear(d_model, nhead * d_model)
#         self.linear_v = nn.Linear(d_model, nhead * d_model)
#         self.linear_out = nn.Linear(nhead * d_model, d_model)

#     def forward(self, q, k, v):
#         q = self.linear_q(q)
#         k = self.linear_k(k)
#         v = self.linear_v(v)

#         q, k, v = q.split(self.nhead, dim=1), k.split(self.nhead, dim=1), v.split(self.nhead, dim=1)

#         attention = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_model)
#         attention = attention.softmax(dim=-1)
#         attention = attention * (1.0 - self.dropout)
#         attention = attention.unsqueeze(1)

#         output = attention * v
#         output = output.transpose(1, 2).contiguous().view(-1, self.nhead * self.d_model)
#         output = self.linear_out(output)
#         return output

# class PositionwiseFeedForward(nn.Module):
#     def __init__(self, d_model, dim_feedforward, dropout):
#         super(PositionwiseFeedForward, self).__init__()
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#     def forward(self, x):
#         x = self.linear1(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#         x = self.linear2(x)
#         return x


import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.d_model = d_model
        
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, 512, 512), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
        
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src) * math.sqrt(self.d_model)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt) * math.sqrt(self.d_model)))
        
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        
        output = self.fc_out(dec_output)
        return output
        
class CustomedTracer(torch.fx.Tracer):
    """
    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
    to ``Tracer().trace(m)``.
    This Tracer override the ``is_leaf_module`` function to make symbolic trace
    right in some cases.
    """
    def __init__(self, *args, customed_leaf_module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.customed_leaf_module = customed_leaf_module

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.
        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.
        Args:
            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        if self.customed_leaf_module and isinstance(m, self.customed_leaf_module):
            return True
        
        if hasattr(m, '_is_leaf_module') and m._is_leaf_module:
            return True

        return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)

class TransformerIR(BaseModelIR):
    def __init__(self, model_name="transformer", tmp_width=1):
        super().__init__(model_name, tmp_width)

        self.out_dir = None
        self.graphmodule = None
        self.out_dir = self.create_out_dir()
             
    def set_model(self):
        self.trace_only_model = True
        device = torch.device("cuda")

        # Example usage
        src_vocab_size = 50
        tgt_vocab_size = 50
        d_model = 512
        num_heads = 2
        num_layers = 1
        d_ff = 2048
        max_seq_length = 100
        dropout = 0.1

        self.model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

        # Generate some dummy data
        src = torch.randint(1, src_vocab_size, (64, 20))  # (batch_size, seq_len)
        tgt = torch.randint(1, tgt_vocab_size, (64, 22))  # (batch_size, seq_len)
        # self.model = Transformer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, num_layers=1).to(device)

    def get_model_type(self):
        return "transformer"

    def create_out_dir(self):
        curr_dir = Path(__file__).parent.absolute()
        curr_dir = os.path.join(curr_dir, "../out/Transformer/")
        isExist = os.path.exists(curr_dir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(curr_dir)
            print("The new directory is created!")

        return curr_dir

    def get_out_dir(self):
        if not self.out_dir:
            raise ValueError("Out directory not setup for", self.model_name)

        return self.out_dir

    def print_graphmodule(self):
        self.graphmodule.print_readable()

    def obtain_symbolic_trace_model(self, micro_batch_size=1, sequence_length=1):
        tracer = CustomedTracer()
        device = torch.device("cuda")

        # src = torch.randint(1, 50, (1, 20))  # (batch_size, seq_len)
        # tgt = torch.randint(1, 50, (1, 22))  # (batch_size, seq_len)
        input_ids = torch.ones(
                    micro_batch_size, sequence_length, dtype=torch.float,).to(device)
        input_ids_2 = torch.ones(
                    micro_batch_size, sequence_length, dtype=torch.float,).to(device)

        graph = tracer.trace(self.model)
        name = self.model.__class__.__name__ if isinstance(
            self.model, torch.nn.Module) else self.model.__name__
        graphmodule = torch.fx.GraphModule(tracer.root, graph, name)

        model_shapeprop = ShapeProp(graphmodule)

        self.graphmodule = model_shapeprop.propagate(input_ids, input_ids_2)

    def get_layer_id(self, n, curr_layer_id):
        layer_annotations = ["layer", "layers"]

        node_name = n.name
        layer_details = node_name.split("_")
        for l in range(0, len(layer_details)):
            if layer_details[l] in layer_annotations:
                if layer_details[l + 1] and layer_details[l + 1].isdigit():
                    return (True, int(layer_details[l + 1]))
        return (False, 0)

    def create_graph_from_symbolic_trace(self):
        super().create_graph_from_symbolic_trace()

    def extract_model_graph(self, micro_batch_size=1, sequence_length=64, force_reextract_model=False,):
        self.load_language_model(
            self.out_dir, micro_batch_size, sequence_length, force_reextract_model)