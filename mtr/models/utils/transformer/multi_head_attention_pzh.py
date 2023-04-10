# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


"""
Mostly copy-paste from https://github.com/dvlab-research/DeepVision3D/blob/master/EQNet/eqnet/transformer/multi_head_attention.py
"""

import warnings
import torch
from torch.nn import Linear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch.nn as nn


# from mtr.ops import attention


class MultiheadAttentionLocal(nn.MultiheadAttention):
    def forward(
        self,
        query,  # total_q_num, c
        key,  # total_k_num, c
        value,  # total_k_num, c
        index_pair,  # total_q_num, max_memory_num
        query_batch_cnt,  # bs: query_amount of each batch
        key_batch_cnt,  # bs: key_amount of each batch.
        index_pair_batch,  # total_q_num, batch_index of each query.
        attn_mask=None,  # total_q_num, max_memory_num
        vdim=None,

        # positional encoding setting.
        relative_atten_weights=None,  # total_q_num, max_memory_num, nhead

        # crpe module.
        ctx_rpe_query=None,
        ctx_rpe_key=None,
        ctx_rpe_value=None,
        rpe_distance=None,
        **kwargs
    ):
        total_query_len, embed_dim = query.size()
        max_memory_len = index_pair.shape[1]
        
        if vdim is None:
            assert key.size() == value.size()
            vdim = embed_dim
            v_head_dim = self.head_dim
        else:
            v_head_dim = vdim // self.num_heads
            assert v_head_dim * self.num_heads == vdim 

        scaling = float(self.head_dim) ** -0.5

        # generate qkv features.
        if not self.without_weight:
            q = self._proj_qkv(query, 0, embed_dim)
            q = q * scaling
            k = self._proj_qkv(key, embed_dim, embed_dim * 2)
            v = self._proj_qkv(value, embed_dim * 2, embed_dim * 3)
        else:
            q = query * scaling
            k, v = key, value 

        # -1 in index_pair means this key not joining attention computation.
        used_attn_mask = (index_pair == -1)  # Ignore the -1 pair.
        if attn_mask is not None:
            # attn_mask should have a shape as [total_query_size, max_memory_size]
            attn_mask = attn_mask.to(torch.bool)
            used_attn_mask = torch.logical_or(used_attn_mask, attn_mask)

        q = q.contiguous().view(total_query_len, self.num_heads, self.head_dim)
        k = k.contiguous().view(-1, self.num_heads, self.head_dim)
        v = v.contiguous().view(-1, self.num_heads, v_head_dim)

        # compute attention weight.
        attn_output_weights = attention.__all__[self.attention_version].attention_weight_computation(
            query_batch_cnt, key_batch_cnt, index_pair_batch, index_pair,
            q, k)  # total_query_len, max_memory_len, num_heads
        assert list(attn_output_weights.size()) == [total_query_len, max_memory_len, self.num_heads]

        if ctx_rpe_key is not None:
            rpe_attn_weight = ctx_rpe_key(rpe_distance, k, scaling,
                                          query_batch_cnt, key_batch_cnt,
                                          index_pair_batch, index_pair)
            attn_output_weights = attn_output_weights + rpe_attn_weight
        if ctx_rpe_query is not None:
            rpe_attn_weight = ctx_rpe_query(rpe_distance, q, 1.0, query_batch_cnt)
            attn_output_weights = attn_output_weights + rpe_attn_weight

        if relative_atten_weights is not None:
            # relative_atten_weights: A float tensor with shape [total_query_num, max_memory_num, nhead]
            attn_output_weights = attn_output_weights + relative_atten_weights

        # attn_output_weights: [total_query_num, max_memory_num, nhead]
        used_attn_mask = used_attn_mask.unsqueeze(-1).repeat(1, 1, self.num_heads).contiguous()
        attn_output_weights.masked_fill_(used_attn_mask, float("-inf"))
        attn_output_weights = F.softmax(attn_output_weights, dim=1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        if ctx_rpe_value is not None:
            attn_output = ctx_rpe_value(rpe_distance, attn_output_weights, v,
                                        query_batch_cnt, key_batch_cnt,
                                        index_pair_batch, index_pair)
        else:
            attn_output = attention.__all__[self.attention_version].attention_value_computation(
                query_batch_cnt, key_batch_cnt, index_pair_batch, index_pair,
                attn_output_weights, v)
        assert list(attn_output.size()) == [total_query_len, self.num_heads, v_head_dim]

        attn_output = attn_output.view(total_query_len, vdim)
        
        if self.out_proj is not None:
            attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        return attn_output, attn_output_weights.sum(dim=-1) / self.num_heads