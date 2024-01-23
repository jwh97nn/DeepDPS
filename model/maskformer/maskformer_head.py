# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .pixel_decoder import build_pixel_decoder
from .transformer_decoder import build_transformer_decoder


@SEM_SEG_HEADS_REGISTRY.register()
class MaskFormerHead(nn.Module):

    _version = 2

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
        depth_decoder: bool,
        depth_decoder_name: str,
        depth_hidden_dim: int,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape1 = input_shape
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes

        # ----------------------- Depth BEGIN-----------------------
        self.depth_decoder = depth_decoder
        if self.depth_decoder:
            input_shape = {
                k: v
                for k, v in input_shape1.items()
                if k in ["res2", "res3", "res4", "res5"]
            }
            if depth_decoder_name == "Base":
                from .pixel_decoder.fpn import BasePixelDecoder

                self.depth_pixel_decoder = BasePixelDecoder(
                    input_shape=input_shape,
                    conv_dim=256,
                    mask_dim=256,
                    norm="GN",
                )
            elif depth_decoder_name == "Deformable":
                from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder

                self.depth_pixel_decoder = MSDeformAttnPixelDecoder(
                    input_shape=input_shape,
                    transformer_dropout=0.0,
                    transformer_nheads=8,
                    transformer_dim_feedforward=1024,
                    # transformer_dim_feedforward=1024 // 4,
                    transformer_enc_layers=6,
                    conv_dim=depth_hidden_dim,
                    mask_dim=depth_hidden_dim,
                    norm="GN",
                    # deformable transformer encoder args
                    transformer_in_features=["res3", "res4", "res5"],
                    common_stride=4,
                )
        # ----------------------- Depth END -----------------------

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif (
            cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder"
        ):  # for maskformer2
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            transformer_predictor_in_channels = input_shape[
                cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE
            ].channels

        return {
            "input_shape": {
                k: v
                for k, v in input_shape.items()
                if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
            "depth_decoder": cfg.MODEL.MASK_FORMER.DEPTH_DECODER,
            "depth_decoder_name": cfg.MODEL.MASK_FORMER.DEPTH_DECODER_NAME,
            "depth_hidden_dim": cfg.MODEL.MASK_FORMER.DEPTH_HIDDEN_DIM,
        }

    def forward(self, features, mask=None):
        """
        Returns:
            dict:
                * 'pred_logits': [B, Q, class + 1]
                * 'pred_masks': [B, Q, H/4, W/4]
                * 'aux_outputs: list[dict] (len = MASK_FORMER.DEC_LAYERS - 1)
                    * 'pred_logits': [B, Q, class + 1]
                    * 'pred_masks': [B, Q, H/4, W/4]
        """

        return self.layers(features, mask)

    def layers(self, features, mask=None):
        (
            mask_features,  # [256, H/4, W/4]
            transformer_encoder_features,
            multi_scale_features,  # [256, H/32, W/32], [256, H/16, W/16], [256, H/8, W/8]
        ) = self.pixel_decoder.forward_features(features)

        # ----------------------- Depth BEGIN-----------------------
        if self.depth_decoder:
            (
                depth_features1,
                _,
                depth_features2,
            ) = self.depth_pixel_decoder.forward_features(features)
            depth_features = depth_features2 + [depth_features1]
        else:
            depth_features = None
        # ----------------------- Depth END -----------------------

        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            # predictions = self.predictor(multi_scale_features, mask_features, mask)
            # predictions = self.predictor(
            #     multi_scale_features, mask_features, mask, depth_features
            # )
            predictions = self.predictor(
                multi_scale_features, mask_features, mask, features, depth_features
            )
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(
                    transformer_encoder_features, mask_features, mask
                )
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(
                    features[self.transformer_in_feature], mask_features, mask
                )

        return predictions
