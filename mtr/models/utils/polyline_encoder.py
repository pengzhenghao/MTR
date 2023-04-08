# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import torch
import torch.nn as nn
from ..utils import common_layers


class PointNetPolylineEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None):
        super().__init__()
        self.pre_mlps = common_layers.build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False
        )
        self.mlps = common_layers.build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False
        )
        
        if out_channels is not None:
            self.out_mlps = common_layers.build_mlps(
                c_in=hidden_dim, mlp_channels=[hidden_dim, out_channels], 
                ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None 

    def forward(self, polylines, polylines_mask):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):

        Returns:
        """
        batch_size, num_polylines,  num_points_each_polylines, C = polylines.shape

        # pre-mlp
        polylines_feature_valid = self.pre_mlps(polylines[polylines_mask])  # (N, C)
        polylines_feature = polylines.new_zeros(batch_size, num_polylines,  num_points_each_polylines, polylines_feature_valid.shape[-1])
        polylines_feature[polylines_mask] = polylines_feature_valid

        # get global feature
        # PZH NOTE: polylines_feature is in shape [T, num poly, num points, D]
        pooled_feature = polylines_feature.max(dim=2)[0]
        # PZH NOTE: pooled_feature is in shape [T, num poly, D]

        # PZH NOTE: Concat [T, num poly, num points, D] with [T, num poly, 1 * numpoints, D]
        # PZH NOTE: = [T, num poly, num points, 2D], D = hidden dim = 256
        polylines_feature = torch.cat((polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1)), dim=-1)

        # PZH NOTE: [T, num poly, num points, 2D] -> [T, num poly, num points, D]
        polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        # PZH NOTE: feature_buffers in shape [T, num poly, num points, D]
        feature_buffers = polylines_feature.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1])
        feature_buffers[polylines_mask] = polylines_feature_valid

        # max-pooling
        # PZH NOTE: [T, num poly, num points, D] -> [T, num poly, D]
        feature_buffers = feature_buffers.max(dim=2)[0]  # (batch_size, num_polylines, C)
        
        # out-mlp 
        if self.out_mlps is not None:
            valid_mask = (polylines_mask.sum(dim=-1) > 0)
            feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])  # (N, C)
            feature_buffers = feature_buffers.new_zeros(batch_size, num_polylines, feature_buffers_valid.shape[-1])
            feature_buffers[valid_mask] = feature_buffers_valid

        # PZH NOTE: in shape [T, num poly, D_MODEL (256 for encoder 512 for decoder)]
        return feature_buffers
