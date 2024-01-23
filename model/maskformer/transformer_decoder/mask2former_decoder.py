# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .maskformer_decoder import TRANSFORMER_DECODER_REGISTRY

from collections import OrderedDict


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        latent_layers: int,
        cnn_depth: bool,
        train_layers: list,
        depth_feature: str,
        depth_multiscale: bool,
        with_mean: bool,
        learn_mean: bool,
        depth_max: float,
        depth_to_semantic: bool,
        depth_decoder: bool,
        depth_hidden_dim: int,
        semantic_guided: bool,
        depth_guided: bool,
        depth_guided_scales: list,
        depth_guided_source: str,
        freeze_semantic: bool,
        backup_token: bool,
        train_depth: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # ----------------------- Depth BEGIN-----------------------
        self.train_depth = train_depth

        if self.train_depth:
            self.semantic_guided = semantic_guided
            self.depth_guided = depth_guided
            self.depth_guided_cov = None
            if self.depth_guided:
                self.depth_guided_cov = nn.Parameter(
                    torch.tensor([1.0, 1.0] * len(depth_guided_scales))
                )
            self.depth_guided_source = depth_guided_source
            self.backup_token = backup_token

            if self.backup_token:
                self.backup_feat = nn.Embedding(1, depth_hidden_dim)
                self.backup_embed = nn.Embedding(1, depth_hidden_dim)
                self.backup_cross_attention_layers = nn.ModuleList()
                self.backup_ffn_layers = nn.ModuleList()
                for _ in range(self.num_feature_levels):
                    self.backup_cross_attention_layers.append(
                        CrossAttentionLayer(
                            d_model=depth_hidden_dim,
                            nhead=nheads,
                            dropout=0.0,
                            normalize_before=pre_norm,
                        )
                    )
                    self.backup_ffn_layers.append(
                        FFNLayer(
                            d_model=depth_hidden_dim,
                            dim_feedforward=dim_feedforward,
                            # dim_feedforward=dim_feedforward // 4,
                            dropout=0.0,
                            normalize_before=pre_norm,
                        )
                    )

            self.depth_decoder = depth_decoder
            self.cnn_depth = cnn_depth
            self.with_mean = with_mean
            self.learn_mean = learn_mean
            self.depth_max = depth_max

            if self.cnn_depth:
                num_ch_enc = [256, 512, 1024, 2048]
                num_ch_dec = [16, 32, 64, 128, 256]
                self.convs = OrderedDict()
                for i in range(3, -1, -1):
                    # upconv_0
                    num_ch_in = num_ch_enc[-1] if i == 3 else num_ch_dec[i + 1]
                    num_ch_out = num_ch_dec[i]
                    self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

                    # upconv_1
                    num_ch_in = num_ch_dec[i]
                    if i > 0:
                        num_ch_in += num_ch_enc[i - 1]
                    num_ch_out = num_ch_dec[i]
                    self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

                for s in [0, 1, 2]:
                    self.convs[("dispconv", s)] = Conv3x3(num_ch_dec[s], 1)
                self.decoder = nn.ModuleList(list(self.convs.values()))
                self.sigmoid = nn.Sigmoid()

            else:
                self.depth_pe_layer = PositionEmbeddingSine(
                    depth_hidden_dim // 2, normalize=True
                )
                if self.depth_decoder:  # Seperate Depth pixel_decoder
                    self.depth_level_embed = nn.Embedding(
                        self.num_feature_levels, depth_hidden_dim
                    )

                self.depth_norm = nn.LayerNorm(depth_hidden_dim)
                if depth_multiscale:
                    self.depth_pred_embed = nn.ModuleList(
                        [
                            MLP(depth_hidden_dim, depth_hidden_dim, 256, 3),
                            MLP(depth_hidden_dim, depth_hidden_dim, 256, 3),
                            MLP(depth_hidden_dim, depth_hidden_dim, 256, 3),
                            MLP(depth_hidden_dim, depth_hidden_dim, 256, 3),
                        ]
                    )
                else:
                    self.depth_pred_embed = MLP(
                        depth_hidden_dim, depth_hidden_dim, depth_hidden_dim, 3
                    )
                    if self.with_mean:
                        if self.learn_mean:
                            self.mean_embed = nn.Embedding(num_queries, 2)
                        else:
                            self.depth_mean_pred_embed = nn.Linear(depth_hidden_dim, 2)

                self.depth_to_semantic = depth_to_semantic
                if self.depth_to_semantic:
                    self.depth_to_semantic_embed = FFNLayer(
                        depth_hidden_dim, depth_hidden_dim
                    )

                self.latent_layers = latent_layers
                self.latent = nn.Embedding(num_queries, depth_hidden_dim)
                self.latent_pos = nn.Embedding(num_queries, depth_hidden_dim)
                self.latent_encode_layers = nn.ModuleList()
                self.latent_process_layers = nn.ModuleList()
                self.latent_decode_layers = nn.ModuleList()
                self.latent_ffn_layers = nn.ModuleList()
                for _ in range(latent_layers):
                    self.latent_encode_layers.append(
                        CrossAttentionLayer(
                            d_model=depth_hidden_dim,
                            nhead=nheads,
                            dropout=0.0,
                            normalize_before=False,
                        )
                    )
                    self.latent_process_layers.append(
                        SelfAttentionLayer(
                            d_model=depth_hidden_dim,
                            nhead=nheads,
                            dropout=0.0,
                            normalize_before=False,
                        )
                    )
                    self.latent_decode_layers.append(
                        CrossAttentionLayer(
                            d_model=depth_hidden_dim,
                            nhead=nheads,
                            dropout=0.0,
                            normalize_before=False,
                        )
                    )
                    self.latent_ffn_layers.append(
                        FFNLayer(
                            d_model=depth_hidden_dim,
                            dim_feedforward=dim_feedforward,
                            # dim_feedforward=dim_feedforward // 4,
                            dropout=0.0,
                            normalize_before=False,
                        )
                    )

                if depth_hidden_dim != hidden_dim:
                    self.hidden_adaptor = nn.Linear(hidden_dim, depth_hidden_dim)
                else:
                    self.hidden_adaptor = None

                self.train_layers = train_layers
                for i in range(self.latent_layers):
                    if i not in train_layers:
                        for k, v in self.named_parameters():
                            if f"latent_encode_layers.{i}" in k:
                                v.requires_grad = False
                            if f"latent_process_layers.{i}" in k:
                                v.requires_grad = False
                            if f"latent_decode_layers.{i}" in k:
                                v.requires_grad = False
                            if f"latent_ffn_layers.{i}" in k:
                                v.requires_grad = False

                self.depth_feature = depth_feature
                self.depth_multiscale = depth_multiscale
            # ----------------------- Depth END -----------------------
            if freeze_semantic:
                for k, v in self.named_parameters():
                    if "depth" in k:
                        continue
                    if "latent" in k:
                        continue
                    v.requires_grad = False

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        ret["latent_layers"] = cfg.MODEL.MASK_FORMER.LATENT_LAYERS
        ret["cnn_depth"] = cfg.MODEL.MASK_FORMER.CNN_DEPTH
        ret["train_layers"] = cfg.MODEL.MASK_FORMER.TRAIN_LAYERS
        ret["depth_feature"] = cfg.MODEL.MASK_FORMER.DEPTH_FEATURE
        ret["depth_multiscale"] = cfg.MODEL.MASK_FORMER.DEPTH_MULTISCALE
        ret["with_mean"] = cfg.MODEL.MASK_FORMER.WITH_MEAN
        ret["learn_mean"] = cfg.MODEL.MASK_FORMER.LEARN_MEAN
        ret["depth_max"] = cfg.MODEL.MASK_FORMER.DEPTH_MAX
        ret["depth_to_semantic"] = cfg.MODEL.MASK_FORMER.DEPTH_TO_SEMANTIC
        ret["depth_decoder"] = cfg.MODEL.MASK_FORMER.DEPTH_DECODER
        ret["depth_hidden_dim"] = cfg.MODEL.MASK_FORMER.DEPTH_HIDDEN_DIM
        ret["semantic_guided"] = cfg.MODEL.MASK_FORMER.SEMANTIC_GUIDED
        ret["depth_guided"] = cfg.MODEL.MASK_FORMER.DEPTH_GUIDED
        ret["depth_guided_scales"] = cfg.MODEL.MASK_FORMER.DEPTH_GUIDED_SCALES
        ret["depth_guided_source"] = cfg.MODEL.MASK_FORMER.DEPTH_GUIDED_SOURCE
        ret["freeze_semantic"] = cfg.MODEL.MASK_FORMER.FREEZE_SEMANTIC
        ret["backup_token"] = cfg.MODEL.MASK_FORMER.BACKUP_TOKEN
        ret["train_depth"] = cfg.MODEL.MASK_FORMER.TRAIN_DEPTH

        return ret

    # def forward(self, x, mask_features, mask=None):
    # def forward(self, x, mask_features, mask, depth_features):
    def forward(self, x, mask_features, mask, features, depth_features=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])  # 1/32, 1/16, 1/8
            pos.append(self.pe_layer(x[i], None).flatten(2))  # [N, C, H*W/32*32]
            src.append(
                self.input_proj[i](x[i]).flatten(2)  # [N, C, H*W/32*32]
                + self.level_embed.weight[i][None, :, None]  # [1, C, 1]
            )

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        # ----------------------- Depth BEGIN-----------------------
        if self.train_depth:
            if not self.cnn_depth:
                depth_src, depth_pos, depth_src_size = [], [], []
                for i in range(self.num_feature_levels):
                    depth_src_size.append(x[i].shape[-2:])

                    if self.depth_decoder:  # Separate Depth pixel_decoder
                        depth_pos.append(
                            self.depth_pe_layer(depth_features[i], None).flatten(2)
                        )
                        depth_src.append(
                            depth_features[i].flatten(2)
                            + self.depth_level_embed.weight[i][None, :, None]
                        )
                    else:
                        depth_pos.append(self.depth_pe_layer(x[i], None).flatten(2))
                        depth_src.append(
                            x[i].flatten(2) + self.level_embed.weight[i][None, :, None]
                        )

                    depth_src[-1] = depth_src[-1].permute(2, 0, 1)
                    depth_pos[-1] = depth_pos[-1].permute(2, 0, 1)

                latent = self.latent.weight.unsqueeze(1).repeat(1, bs, 1)  # [C, N, C]
                latent_pos = self.latent_pos.weight.unsqueeze(1).repeat(
                    1, bs, 1
                )  # [C, N, C]
                if self.learn_mean:
                    learned_mean = self.mean_embed.weight.unsqueeze(0).repeat(
                        bs, 1, 1
                    )  # [B, Q, 2]

                if self.backup_token:
                    backup = self.backup_feat.weight.unsqueeze(1).repeat(1, bs, 1)
                    backup_pos = self.backup_embed.weight.unsqueeze(1).repeat(1, bs, 1)

            predictions_depth = []
            predictions_depth.append([None, None])

            if self.depth_feature == "single":
                if self.depth_decoder:
                    d_feature = depth_features[-1]
                    d_feature_multi = depth_features[:3]
                else:
                    d_feature = mask_features
                    d_feature_multi = x
            else:
                d_feature = mask_features + features["res2"]
        # ----------------------- Depth END -----------------------

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[
                torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])
            ] = False  # [B*h, Q, HW]
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index],
                query_pos=query_embed,
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

            # ----------------------- Depth BEGIN-----------------------
            if self.train_depth:
                if not self.cnn_depth:
                    if i not in self.train_layers:
                        predictions_depth.append([None, None])
                        continue

                    _attn_mask = F.interpolate(
                        outputs_mask,
                        size=depth_src_size[level_index],
                        mode="bilinear",
                        align_corners=False,
                    )
                    _attn_mask = (
                        _attn_mask.sigmoid()
                        .flatten(2)
                        .unsqueeze(1)
                        .repeat(1, 8, 1, 1)
                        .flatten(0, 1)
                        < 0.5
                    ).bool()
                    _attn_mask = _attn_mask.detach()
                    _attn_mask[
                        torch.where(_attn_mask.sum(-1) == _attn_mask.shape[-1])
                    ] = False  # [B*h, Q, HW]

                    if self.latent_layers == 9:
                        latent_layer_idx = i
                    elif self.latent_layers == 3:
                        if level_index != self.num_feature_levels - 1:
                            predictions_depth.append([None, None])
                            continue
                        latent_layer_idx = i // 3

                    latent = self.latent_encode_layers[latent_layer_idx](
                        latent,
                        depth_src[level_index],
                        memory_mask=_attn_mask,
                        memory_key_padding_mask=None,
                        pos=depth_pos[level_index],
                        query_pos=latent_pos,
                    )
                    latent = self.latent_process_layers[latent_layer_idx](
                        latent,
                        tgt_mask=None,
                        tgt_key_padding_mask=None,
                        query_pos=latent_pos,
                    )

                    if self.hidden_adaptor is None:
                        depth_output = self.latent_decode_layers[latent_layer_idx](
                            output,
                            latent,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=latent_pos,
                            query_pos=query_embed,
                        )
                    else:
                        output_adapt = self.hidden_adaptor(output)
                        depth_output = self.latent_decode_layers[latent_layer_idx](
                            output_adapt,
                            latent,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=latent_pos,
                            query_pos=None,
                        )

                    depth_output = self.latent_ffn_layers[latent_layer_idx](
                        depth_output
                    )

                    if self.depth_to_semantic:
                        output = output + self.depth_to_semantic_embed(depth_output)

                    d_output = self.depth_norm(depth_output)
                    d_output = d_output.transpose(0, 1)

                    if self.depth_multiscale:
                        # Low Resolution (1/32, 1/16, 1/8)
                        f_low = depth_src[level_index]  # [HW, B, C]
                        f_low = f_low.unflatten(
                            0, depth_src_size[level_index]
                        )  # [H, W, B, C]
                        f_low = f_low.permute(2, 3, 0, 1)  # [B, C, H, W]

                        d_output_low = self.depth_pred_embed[level_index](d_output)
                        outputs_depth_low = torch.einsum(
                            "bqc, bchw -> bqhw", d_output_low, f_low
                        )
                        outputs_depth_low = torch.sigmoid(outputs_depth_low)

                        # High Resolution (1/4)
                        # d_output = self.depth_pred_embed[-1](d_output)
                        d_output = self.depth_pred_embed[-1](d_output_low)
                        outputs_depth_high = torch.einsum(
                            "bqc, bchw -> bqhw", d_output, d_feature
                        )
                        # outputs_depth_high = torch.sigmoid(outputs_depth_high)

                        outputs_depth = (
                            F.interpolate(
                                outputs_depth_low,
                                size=outputs_depth_high.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                            * self.depth_max
                            + outputs_depth_high
                        )
                        # outputs_depth = outputs_depth * self.depth_max
                    else:
                        depth_embed = self.depth_pred_embed(d_output)
                        outputs_depth = torch.einsum(
                            "bqc, bchw -> bqhw", depth_embed, d_feature
                        )
                        outputs_depth = torch.sigmoid(outputs_depth)

                        if self.with_mean:
                            if self.learn_mean:
                                outputs_mean = learned_mean
                            else:
                                outputs_mean = self.depth_mean_pred_embed(
                                    d_output
                                )  # [B, Q, 2]
                            outputs_mean = torch.sigmoid(outputs_mean)
                            d_mean = outputs_mean[:, :, 0] * self.depth_max / 2
                            d_range = outputs_mean[:, :, 1] * self.depth_max

                            # outputs_depth = outputs_depth * 2 - 1 # (0, 1) -> (-1, 1)
                            outputs_depth = outputs_depth - 0.5  # (0, 1) -> (-0.5, 0.5)
                            outputs_depth = torch.einsum(
                                "bq, bqhw -> bqhw", d_range, outputs_depth
                            ) + d_mean.reshape(*d_mean.shape, 1, 1)
                        else:
                            outputs_depth = outputs_depth * self.depth_max
                            outputs_mean = None

                    if self.backup_token:
                        backup = self.backup_cross_attention_layers[level_index](
                            backup,
                            depth_src[level_index],
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=depth_pos[level_index],
                            query_pos=backup_pos,
                        )
                        backup = self.backup_ffn_layers[level_index](backup)
                        backup_output = self.depth_norm(backup)
                        backup_output = backup_output.transpose(0, 1)
                        depth_embed = self.depth_pred_embed(backup_output)
                        backup_depth = torch.einsum(
                            "bqc, bchw -> bqhw", depth_embed, d_feature
                        )
                        backup_depth = torch.sigmoid(backup_depth)
                        if not self.with_mean:
                            backup_depth = backup_depth * self.depth_max
                    else:
                        backup_depth = None

                    predictions_depth.append(
                        [outputs_depth, outputs_mean, backup_depth]
                    )

                if self.cnn_depth:
                    input_features = [
                        features[i] for i in ["res2", "res3", "res4", "res5"]
                    ]
                    for _ in range(9 - 3):
                        predictions_depth.append([None, None])
                    x = input_features[-1]
                    for i in range(3, -1, -1):
                        x = self.convs[("upconv", i, 0)](x)
                        x = [F.interpolate(x, scale_factor=2, mode="nearest")]
                        if i > 0:
                            x += [input_features[i - 1]]
                        x = torch.cat(x, 1)
                        x = self.convs[("upconv", i, 1)](x)
                        if i in [0, 1, 2]:
                            predictions_depth.append(
                                [
                                    self.sigmoid(self.convs[("dispconv", i)](x))
                                    * self.depth_max,
                                    None,
                                ]
                            )
            else:
                # Not training Depth
                predictions_depth = [[None, None]] * 9
            # ----------------------- Depth END ------------------------

        assert len(predictions_class) == self.num_layers + 1

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class if self.mask_classification else None,
                predictions_mask,
                predictions_depth,
            ),
            "pred_depths": predictions_depth[-1],
        }

        if self.train_depth:
            if self.semantic_guided:
                out.update({"depth_feat": d_feature_multi + [d_feature]})
            if self.depth_guided:
                out.update({"semantic_feat": x + [mask_features]})
                if self.depth_guided_source == "feature":
                    if not out.__contains__("depth_feat"):
                        out.update({"depth_feat": d_feature_multi + [d_feature]})
                if self.depth_guided_cov is not None:
                    out.update({"depth_guided_cov": self.depth_guided_cov})

        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # [N, Q, C]
        outputs_class = self.class_embed(decoder_output)  # [N, Q, num_classes + 1]
        mask_embed = self.mask_embed(decoder_output)  # [N, Q, C]
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(
            outputs_mask,
            size=attn_mask_target_size,
            mode="bilinear",
            align_corners=False,
        )
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_seg_masks):
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_depth):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            # return [
            #     {"pred_logits": a, "pred_masks": b}
            #     for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            # ]

            ret = [
                {"pred_logits": a, "pred_masks": b, "pred_depths": c}
                for a, b, c in zip(
                    outputs_class[:-1], outputs_seg_masks[:-1], outputs_depth[:-1]
                )
            ]
            return ret
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
            )
        return self.forward_post(
            tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
        )


class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


# Monodepth2
class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU"""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input"""

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out
