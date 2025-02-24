# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import numpy as np
import torch
import torch.nn as nn


from mtr.models.utils.transformer import transformer_encoder_layer, position_encoding_utils
from mtr.models.utils import polyline_encoder
from mtr.utils import common_utils
# from mtr.ops.knn import knn_utils


def find_k_nearest_neighbors(pos, K, valid_mask):
    """
    Find the K-nearest neighbors of all vehicles, excluding invalid vehicles.

    Args:
        pos (torch.Tensor): Position tensor of shape [B, N, 3].
        K (int): The number of nearest neighbors to find.
        valid_mask (torch.Tensor): Boolean mask tensor of shape [B, N], where True indicates valid vehicles.

    Returns:
        (torch.Tensor): Indices matrix of shape [B, N, K].
    """
    B, N, _ = pos.shape

    INF = float("+inf")

    # Compute pairwise distances using cdist
    pairwise_distances = torch.cdist(pos, pos, p=2)

    # NOTE: We will include "itself" into "K-neighbors".
    # Set diagonal elements and invalid vehicle distances to a large value
    # diag_indices = torch.arange(N, device=pos.device)
    # pairwise_distances[:, diag_indices, diag_indices] = INF

    # Adjust the valid_mask tensor to match the shape of pairwise_distances
    invalid_mask_expanded = (~valid_mask).unsqueeze(-1).expand(B, N, N)

    # Set distances for invalid vehicles to infinity
    pairwise_distances[invalid_mask_expanded] = INF

    # Find indices of the K-nearest neighbors using topk
    _, indices = torch.topk(pairwise_distances, K, dim=2, largest=False)

    return indices[valid_mask].int()


class MTREncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config

        # build polyline encoders
        self.agent_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_AGENT + 1,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_AGENT,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_AGENT,
            out_channels=self.model_cfg.D_MODEL
        )
        self.map_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_MAP,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_MAP,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_MAP,  # PZH NOTE: totally 5 layers, 3 in pre layers, 2 in out.
            num_pre_layers=self.model_cfg.NUM_LAYER_IN_PRE_MLP_MAP,  # = 3
            out_channels=self.model_cfg.D_MODEL  # 2
        )

        # build transformer encoder layers
        self.use_local_attn = self.model_cfg.get('USE_LOCAL_ATTN', False)
        self_attn_layers = []
        for _ in range(self.model_cfg.NUM_ATTN_LAYERS):
            self_attn_layers.append(self.build_transformer_encoder_layer(
                d_model=self.model_cfg.D_MODEL,
                nhead=self.model_cfg.NUM_ATTN_HEAD,
                dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
                normalize_before=False,
                use_local_attn=self.use_local_attn
            ))

        self.self_attn_layers = nn.ModuleList(self_attn_layers)
        self.num_out_channels = self.model_cfg.D_MODEL

    def build_polyline_encoder(self, in_channels, hidden_dim, num_layers, num_pre_layers=1, out_channels=None):
        ret_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_pre_layers=num_pre_layers,
            out_channels=out_channels
        )
        return ret_polyline_encoder

    def build_transformer_encoder_layer(self, d_model, nhead, dropout=0.1, normalize_before=False, use_local_attn=False):
        single_encoder_layer = transformer_encoder_layer.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,
            normalize_before=normalize_before, use_local_attn=use_local_attn
        )
        return single_encoder_layer

    def apply_global_attn(self, x, x_mask, x_pos):
        """

        Args:
            x (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)

        batch_size, N, d_model = x.shape
        x_t = x.permute(1, 0, 2)
        x_mask_t = x_mask.permute(1, 0, 2)
        x_pos_t = x_pos.permute(1, 0, 2)
 
        pos_embedding = position_encoding_utils.gen_sineembed_for_position(x_pos_t, hidden_dim=d_model)

        for k in range(len(self.self_attn_layers)):
            x_t = self.self_attn_layers[k](
                src=x_t,
                src_key_padding_mask=~x_mask_t,
                pos=pos_embedding
            )
        x_out = x_t.permute(1, 0, 2)  # (batch_size, N, d_model)
        return x_out

    def apply_local_attn(self, x, x_mask, x_pos, num_of_neighbors):
        """

        Args:
            x (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)
        batch_size, N, d_model = x.shape

        x_stack_full = x.view(-1, d_model)  # (batch_size * N, d_model)
        x_mask_stack = x_mask.view(-1)
        x_pos_stack_full = x_pos.view(-1, 3)
        batch_idxs_full = torch.arange(batch_size).type_as(x)[:, None].repeat(1, N).view(-1).int()  # (batch_size * N)

        # filter invalid elements
        x_stack = x_stack_full[x_mask_stack]
        x_pos_stack = x_pos_stack_full[x_mask_stack]

        # It is in shape (BS * N,). It record the batch index of each selected "map feat".
        batch_idxs = batch_idxs_full[x_mask_stack]

        # knn
        # batch_offsets is in shape (bs + 1,)
        # It record:
        # batch_offsets[0] = 0
        # batch_offsets[1] = how many map feat's index == 0?
        # batch_offsets[2] = how many map feat's index == 0 or 1?
        # batch_offsets[i+1] = batch_offsets[i] + (batch_idxs==i).sum()
        batch_offsets = common_utils.get_batch_offsets(batch_idxs=batch_idxs, bs=batch_size).int()  # (batch_size + 1)

        # in shape (bs,)
        # how many map_feat's index==i
        batch_cnt = batch_offsets[1:] - batch_offsets[:-1]


        # PZH NOTE: We abadon the old implementation which requires additional CUDA code compliation.
        # After testing, in a 3090, the new implementation takes 10s to finish 10,000 operation with (9000, 16) output.
        # While the old implementation takes 5s. Their results are compatible.
        # I think it is accepetable to use native pytorch impl at the cost of a little overhead.

        # index_pair = knn_utils.knn_batch_mlogk(
        #     x_pos_stack, x_pos_stack,  batch_idxs, batch_offsets, num_of_neighbors
        # )  # (num_valid_elems, K)

        # [Number of valid elements, K]
        index_pair = find_k_nearest_neighbors(pos=x_pos, K=num_of_neighbors, valid_mask=x_mask)

        # positional encoding
        pos_embedding = position_encoding_utils.gen_sineembed_for_position(x_pos_stack[None, :, 0:2], hidden_dim=d_model)[0]

        # local attn
        output = x_stack
        for k in range(len(self.self_attn_layers)):
            output = self.self_attn_layers[k](
                src=output,
                pos=pos_embedding,
                index_pair=index_pair,
                query_batch_cnt=batch_cnt,
                key_batch_cnt=batch_cnt,
                index_pair_batch=batch_idxs
            )

        ret_full_feature = torch.zeros_like(x_stack_full)  # (batch_size * N, d_model)
        ret_full_feature[x_mask_stack] = output

        ret_full_feature = ret_full_feature.view(batch_size, N, d_model)
        return ret_full_feature

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
              input_dict:
        """
        input_dict = batch_dict['input_dict']
        obj_trajs, obj_trajs_mask = input_dict['obj_trajs'].cuda(), input_dict['obj_trajs_mask'].cuda() 
        map_polylines, map_polylines_mask = input_dict['map_polylines'].cuda(), input_dict['map_polylines_mask'].cuda() 

        obj_trajs_last_pos = input_dict['obj_trajs_last_pos'].cuda()  # [14, 55, 3]
        map_polylines_center = input_dict['map_polylines_center'].cuda()  # [14, 768, 3]
        track_index_to_predict = input_dict['track_index_to_predict']  # [14, ] in int.

        assert obj_trajs_mask.dtype == torch.bool and map_polylines_mask.dtype == torch.bool

        num_center_objects, num_objects, num_timestamps, _ = obj_trajs.shape
        num_polylines = map_polylines.shape[1]

        # apply polyline encoder
        obj_trajs_in = torch.cat((obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1)
        obj_polylines_feature = self.agent_polyline_encoder(obj_trajs_in, obj_trajs_mask)  # (num_center_objects, num_objects, C)
        map_polylines_feature = self.map_polyline_encoder(map_polylines, map_polylines_mask)  # (num_center_objects, num_polylines, C)

        # apply self-attn
        obj_valid_mask = (obj_trajs_mask.sum(dim=-1) > 0)  # (num_center_objects, num_objects)
        map_valid_mask = (map_polylines_mask.sum(dim=-1) > 0)  # (num_center_objects, num_polylines)

        global_token_feature = torch.cat((obj_polylines_feature, map_polylines_feature), dim=1) 
        global_token_mask = torch.cat((obj_valid_mask, map_valid_mask), dim=1) 
        global_token_pos = torch.cat((obj_trajs_last_pos, map_polylines_center), dim=1) 

        if self.use_local_attn:
            global_token_feature = self.apply_local_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos,
                num_of_neighbors=self.model_cfg.NUM_OF_ATTN_NEIGHBORS
            )
        else:
            global_token_feature = self.apply_global_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos
            )

        obj_polylines_feature = global_token_feature[:, :num_objects]  # [14, 55, 256]
        map_polylines_feature = global_token_feature[:, num_objects:]  # [14, 768, 256]
        assert map_polylines_feature.shape[1] == num_polylines

        # organize return features
        # in shape [14, 256]
        center_objects_feature = obj_polylines_feature[torch.arange(num_center_objects), track_index_to_predict]

        batch_dict['center_objects_feature'] = center_objects_feature
        batch_dict['obj_feature'] = obj_polylines_feature  #
        batch_dict['map_feature'] = map_polylines_feature
        batch_dict['obj_mask'] = obj_valid_mask
        batch_dict['map_mask'] = map_valid_mask
        batch_dict['obj_pos'] = obj_trajs_last_pos
        batch_dict['map_pos'] = map_polylines_center

        return batch_dict
