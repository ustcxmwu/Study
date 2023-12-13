#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   model.py
@Time    :   2023-10-12 11:56
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""

import torch
from torch import nn
import torch.nn.functional as F
import copy
import math
import numpy as np
import pandas as pd
import plotly.express as px


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ScaledDotProductAttention(nn.Module):
    "Compute 'Scaled Dot Product Attention'"

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e20)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return p_attn @ value, p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # 记录 attention矩阵结果
        self.dropout = nn.Dropout(p=dropout)
        self.attention = ScaledDotProductAttention()

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# 为了让训练过程与解码过程信息流一致，遮挡tgt序列后面元素，设置其注意力为0
def tril_mask(data):
    "Mask out future positions."
    size = data.size(-1)  # size为序列长度
    full = torch.full((1, size, size), 1, dtype=torch.int, device=data.device)
    mask = torch.tril(full).bool()
    return mask


# 设置对<PAD>的注意力为0
def pad_mask(data, pad=0):
    "Mask out pad positions."
    mask = (data != pad).unsqueeze(-2)
    return mask


# 计算一个batch数据的src_mask和tgt_mask
class MaskedBatch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = pad_mask(src, pad)
        if tgt is not None:
            self.tgt = tgt[:, :-1]  # 训练时,拿tgt的每一个词输入,去预测下一个词,所以最后一个词无需输入
            self.tgt_y = tgt[:, 1:]  # 第一个总是<SOS>无需预测，预测从第二个词开始
            self.tgt_mask = self.make_tgt_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).sum()

    @staticmethod
    def make_tgt_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_pad_mask = pad_mask(tgt, pad)
        tgt_tril_mask = tril_mask(tgt)
        tgt_mask = tgt_pad_mask & (tgt_tril_mask)
        return tgt_mask


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 线性层默认作用在最后一维度
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class LayerNorm(nn.Module):
    "Construct a layernorm module (similar to torch.nn.LayerNorm)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class ResConnection(nn.Module):
    """
    A residual connection with a layer norm.
    Note the norm is at last according to the paper, but it may be better at first.
    """

    def __init__(self, size, dropout, norm_first=True):
        super(ResConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if self.norm_first:
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            return self.norm(x + self.dropout(sublayer(x)))


# 单词嵌入
class WordEmbedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(
            self.d_model
        )  # note here, multiply sqrt(d_model)


# 位置编码
class PositionEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    "TransformerEncoderLayer is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.res_layers = clones(ResConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.res_layers[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.res_layers[1](x, self.feed_forward)


class TransformerEncoder(nn.Module):
    "TransformerEncoder is a stack of N TransformerEncoderLayer"

    def __init__(self, layer, N):
        super(TransformerEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    @classmethod
    def from_config(cls, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        attn = MultiHeadAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        layer = TransformerEncoderLayer(d_model, attn, ff, dropout)
        return cls(layer, N)


class TransformerDecoderLayer(nn.Module):
    "TransformerDecoderLayer is made of self-attn, cross-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, cross_attn, feed_forward, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.res_layers = clones(ResConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.res_layers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.res_layers[1](x, lambda x: self.cross_attn(x, m, m, src_mask))
        return self.res_layers[2](x, self.feed_forward)


class TransformerDecoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(TransformerDecoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

    @classmethod
    def from_config(cls, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        self_attn = MultiHeadAttention(h, d_model)
        cross_attn = MultiHeadAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        layer = TransformerDecoderLayer(d_model, self_attn, cross_attn, ff, dropout)
        return cls(layer, N)

    class Generator(nn.Module):
        "Define standard linear + softmax generation step."

        def __init__(self, d_model, vocab):
            super(Generator, self).__init__()
            self.proj = nn.Linear(d_model, vocab)

        def forward(self, x):
            return F.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.reset_parameters()

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.generator(
            self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
        )

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    @classmethod
    def from_config(
        cls, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
    ):
        encoder = TransformerEncoder.from_config(
            N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout
        )
        decoder = TransformerDecoder.from_config(
            N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout
        )
        src_embed = nn.Sequential(
            WordEmbedding(d_model, src_vocab), PositionEncoding(d_model, dropout)
        )
        tgt_embed = nn.Sequential(
            WordEmbedding(d_model, tgt_vocab), PositionEncoding(d_model, dropout)
        )

        generator = Generator(d_model, tgt_vocab)
        return cls(encoder, decoder, src_embed, tgt_embed, generator)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class LabelSmoothingLoss(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):  # size为词典大小
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))  # 预测结果不会是<SOS> #和<PAD>
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero((target.data == self.padding_idx).int())
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


if __name__ == "__main__":
    # 测试tril_mask
    mask = tril_mask(torch.zeros(1, 10))  # 序列长度为10
    # sns.heatmap(mask[0],cmap=sns.cm.rocket);
    px.imshow(mask[0], color_continuous_scale="blues", height=600, width=600)

    # 测试 ScaledDotProductAttention

    query = torch.tensor(
        [[[0.0, 1.414], [1.414, 0.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]]]
    )
    key = query.clone()
    value = query.clone()

    attention = ScaledDotProductAttention()

    # 没有mask
    out, p_att = attention(query, key, value)
    fig = px.imshow(
        p_att[0],
        color_continuous_scale="blues",
        title="without mask",
        height=600,
        width=600,
    )
    fig.show()

    # 考虑mask
    out, p_att = attention(query, key, value, mask=tril_mask(torch.zeros(3, 5)))
    fig = px.imshow(
        p_att[0],
        color_continuous_scale="blues",
        height=600,
        width=600,
        title="with mask",
    )
    fig.show()

    # 测试MultiHeadAttention
    cross_attn = MultiHeadAttention(h=2, d_model=4)
    cross_attn.eval()
    q1 = torch.tensor([[[0.1, 0.1, 0.1, 0.1], [0.1, 0.3, 0.1, 0.3]]])
    k1 = q1.clone()
    v1 = q1.clone()
    tgt_mask = tril_mask(torch.zeros(2, 2))

    out1 = cross_attn.forward(q1, k1, v1, mask=tgt_mask)
    print("out1:\n", out1)

    # 改变序列的第2个元素取值，由于有mask的遮挡，不会影响第1个输出
    q2 = torch.tensor([[[0.1, 0.1, 0.1, 0.1], [0.4, 0.5, 0.5, 0.8]]])
    k2 = q2.clone()
    v2 = q2.clone()
    tgt_mask = tril_mask(torch.zeros(2, 2))
    out2 = cross_attn.forward(q2, k2, v2, mask=tgt_mask)
    print("out2:\n", out2)

    # 测试MaskedBatch
    mbatch = MaskedBatch(src=src, tgt=tgt, pad=0)
    print(mbatch.src.shape)
    print(mbatch.tgt.shape)
    print(mbatch.tgt_y.shape)

    print(mbatch.src_mask.shape)
    print(mbatch.tgt_mask.shape)
    px.imshow(mbatch.tgt_mask[0], color_continuous_scale="blues", width=600, height=600)
