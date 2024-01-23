# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.layers import cat

from .misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

# from detectron2.projects.point_rend.point_features import (
#     get_uncertain_point_coords_with_randomness,
#     point_sample,
# )

# Tag: Depth
def si_loss(
    depth_gt: torch.Tensor,
    depth_pred: torch.Tensor,
    valid_mask: torch.Tensor,
    # depth_loss_weights: torch.Tensor,
):
    """
    Compute Scale-invariant Loss    [N, H, W]
    """
    eps = 1e-6
    # if valid_mask is None:
    #     valid_mask = torch.ones_like(depth_gt).bool()

    # depth_gt = (depth_gt * valid_mask).clamp(min=eps)
    # depth_pred = (depth_pred * valid_mask).clamp(min=eps)
    depth_gt = depth_gt.clamp(min=eps)
    depth_pred = depth_pred.clamp(min=eps)
    valid_nums = valid_mask.sum(dim=(-1, -2)).clamp(min=1)

    depth_log_diff = torch.log(depth_gt) - torch.log(depth_pred)
    # depth_log_diff = depth_log_diff * depth_loss_weights

    scale_invar_log_error_1 = ((depth_log_diff**2) * valid_mask).sum(
        dim=(-1, -2)
    ) / valid_nums
    scale_invar_log_error_2 = ((depth_log_diff * valid_mask).sum(dim=(-1, -2)) ** 2) / (
        valid_nums**2
    )
    # relat_sqrt_error = torch.sqrt((((1. - depth_pred / depth_gt) ** 2) * valid_mask).sum(dim=(-1,-2)) / valid_nums)
    relat_sqrt_error = torch.sqrt(
        (
            (((1.0 - depth_pred / depth_gt) ** 2) * valid_mask).sum(dim=(-1, -2))
            / valid_nums
        )
        + eps
    )
    loss = (scale_invar_log_error_1 - scale_invar_log_error_2) * 5.0 + relat_sqrt_error

    # loss = (scale_invar_log_error_1 - scale_invar_log_error_2) * 5.

    # depth_gt_mean = depth_gt.sum(dim=(-1, -2)) / valid_nums
    # depth_pred_mean = depth_pred.sum(dim=(-1, -2)) / valid_nums
    # loss += (depth_gt_mean - depth_pred_mean) ** 2

    return loss.mean()


si_loss_jit = torch.jit.script(si_loss)  # type: torch.jit.ScriptModule


def si_loss2(
    depth_gt: torch.Tensor,
    depth_pred: torch.Tensor,
    valid_mask: torch.Tensor,
):
    """
    Compute Scale-invariant Loss    [#Points]
    """
    eps = 1e-6

    depth_gt = depth_gt.clamp(min=eps)
    depth_pred = depth_pred.clamp(min=eps)
    valid_nums = valid_mask.sum(dim=(-1, -2)).clamp(min=1)

    depth_log_diff = torch.log(depth_gt) - torch.log(depth_pred)

    scale_invar_log_error_1 = ((depth_log_diff**2) * valid_mask).sum(
        dim=(-1, -2)
    ) / valid_nums
    scale_invar_log_error_2 = ((depth_log_diff * valid_mask).sum(dim=(-1, -2)) ** 2) / (
        valid_nums**2
    )

    loss = scale_invar_log_error_1 - 0.85 * scale_invar_log_error_2
    loss = 10 * torch.sqrt(loss + eps)

    return loss.mean()


si_loss2_jit = torch.jit.script(si_loss2)  # type: torch.jit.ScriptModule


def si_loss_single(depth_gt, depth_pred):
    # eps = 1e-6
    depth_log_diff = torch.log(depth_gt) - torch.log(depth_pred)
    scale_invar_log_error_1 = (depth_log_diff**2).sum()
    scale_invar_log_error_2 = depth_log_diff.sum() ** 2
    relat_sqrt_error = torch.sqrt(((1.0 - depth_pred / depth_gt) ** 2).sum())
    loss = (scale_invar_log_error_1 - scale_invar_log_error_2) * 5.0 + relat_sqrt_error
    return loss.mean()


si_loss_single_jit = torch.jit.script(si_loss_single)  # type: torch.jit.ScriptModule
# -----------------------------------------------------------------------------


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    # num_points: torch.Tensor,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks
    # return (loss.sum(dim=1) / num_points.squeeze()).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        depth_resolution_full,
        # uncertainty_weight,
        cnn_depth,
        semantic_guided,
        semantic_guided_scales,
        depth_guided,
        depth_guided_scales,
        depth_guided_source,
        backup_token,
        semkitti,
        train_depth,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        self.depth_resolution_full = depth_resolution_full
        # self.uncertainty_weight = uncertainty_weight
        # if uncertainty_weight:
        #     # [Label, Mask_CE, Mask_DICE, Depth]
        #     self.loss_scale = nn.Parameter(torch.tensor([-0.5] * 4))
        self.cnn_depth = cnn_depth
        self.semantic_guided = semantic_guided
        self.semantic_guided_scales = semantic_guided_scales
        self.depth_guided = depth_guided
        self.depth_guided_scales = depth_guided_scales
        self.depth_guided_source = depth_guided_source
        self.backup_token = backup_token
        self.semkitti = semkitti
        self.train_depth = train_depth

        if self.semantic_guided:
            kernel_size = 5
            self.kernel = nn.Parameter(torch.ones(kernel_size, kernel_size))

    # def loss_labels(self, outputs, targets, indices, num_masks):
    def loss_labels(self, outputs, targets, indices, num_masks, depths):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )

        # if self.uncertainty_weight:
        #     loss_ce = loss_ce / (2 * self.loss_scale[0].exp())

        losses = {"loss_ce": loss_ce}
        return losses

    # def loss_masks(self, outputs, targets, indices, num_masks):
    def loss_masks(self, outputs, targets, indices, num_masks, depths):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        if self.semkitti:  # SemKITTI-DVPS sparse GTs
            with torch.no_grad():
                # target_masks: [N, 1, H/4, W/4]
                valid_masks = [t["valid_mask"] for t in targets]
                valid_masks = torch.cat(valid_masks, dim=0)  # [B, 1, H/4, W/4]

                max_points = valid_masks.sum(dim=(-1, -2)).max()

                valid_masks = valid_masks[tgt_idx[0]]  # [N, 1, H/4, W/4] bool
                point_labels_num = valid_masks.sum(dim=(-1, -2))  # [N, 1]

                point_labels = [
                    target_mask[valid_mask]
                    for target_mask, valid_mask in zip(target_masks, valid_masks)
                ]  # [P] * N

                point_labels_ = []
                for labels in point_labels:
                    if labels.shape[0] < max_points:
                        point_labels_.append(
                            torch.cat(
                                [
                                    labels,
                                    torch.zeros(
                                        max_points - labels.shape[0],
                                        device=labels.device,
                                    ),
                                ]
                            )
                        )
                    else:
                        point_labels_.append(labels)
                point_labels = torch.stack(point_labels_)  # [N, P]

            point_logits = [
                src_mask[valid_mask]
                for src_mask, valid_mask in zip(src_masks, valid_masks)
            ]  # [P] * N

            point_logits_ = []
            for logits in point_logits:
                if logits.shape[0] < max_points:
                    point_logits_.append(
                        torch.cat(
                            [
                                logits,
                                torch.zeros(
                                    max_points - logits.shape[0], device=logits.device
                                ),
                            ]
                        )
                    )
                else:
                    point_logits_.append(logits)
            point_logits = torch.stack(point_logits_)  # [N, P]

            losses = {
                "loss_mask": sigmoid_ce_loss_jit(
                    point_logits, point_labels, num_masks, point_labels_num
                ),
                "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
            }

            del valid_masks

        else:
            with torch.no_grad():
                # sample point_coords
                point_coords = get_uncertain_point_coords_with_randomness(
                    src_masks,
                    lambda logits: calculate_uncertainty(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
                # get gt labels
                point_labels = point_sample(
                    target_masks,
                    point_coords,
                    align_corners=False,
                ).squeeze(1)

            point_logits = point_sample(
                src_masks,
                point_coords,
                align_corners=False,
            ).squeeze(
                1
            )  # [N, num_points]

            losses = {
                "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
                "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
            }

        # if self.uncertainty_weight:
        #     losses["loss_mask"] = losses["loss_mask"] / (2 * self.loss_scale[1].exp())
        #     losses["loss_dice"] = losses["loss_dice"] / (2 * self.loss_scale[2].exp())

        if self.train_depth:
            # Tag: Depth
            if outputs["pred_depths"][0] is not None:
                with torch.no_grad():
                    if self.cnn_depth:
                        gt_depths = torch.stack(depths)  # [B, H, W]
                        gt_masks = (gt_depths > 0).float()  # [B, H, W]
                    else:
                        gt_depths = torch.stack(depths)[tgt_idx[0]]  # [#Q, H, W]

                        if not self.depth_resolution_full:  # 1/4 resolution
                            depth_pred_size = outputs["pred_depths"][0].shape[-2:]

                            # Nearest
                            gt_depths = F.interpolate(
                                gt_depths.unsqueeze(0),
                                size=depth_pred_size,
                                mode="nearest",
                            ).squeeze(
                                0
                            )  # [#Q, H', W']
                            gt_depths = (
                                gt_depths
                                * F.interpolate(
                                    target_masks,
                                    size=depth_pred_size,
                                    mode="nearest",
                                )
                                .squeeze(1)
                                .float()
                            )  # [#Q, H', W']
                        gt_masks = (gt_depths > 0).float()  # [#Q, H, W]

                if self.cnn_depth:
                    depth_pred = outputs["pred_depths"][0]
                    mask_shape = target_masks.shape[-2:]
                    depth_pred = F.interpolate(
                        depth_pred,
                        size=mask_shape,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(
                        1
                    )  # [B, H, W]
                else:
                    depth_pred = outputs["pred_depths"][0][src_idx]  # [#Q, H', W']
                    if self.backup_token:
                        backup_depth_pred = outputs["pred_depths"][-1].squeeze(
                            1
                        )  # [B, H', W']

                        if self.depth_resolution_full:
                            mask_shape = target_masks.shape[-2:]  # [#Q, 1, H, W]
                            backup_depth_pred = F.interpolate(
                                backup_depth_pred.unsqueeze(0),
                                size=mask_shape,
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(0)

                        backup_gt_depths = torch.stack(depths)  # [B, H, W]
                        if not self.depth_resolution_full:  # 1/4 resolution
                            backup_gt_depths = F.interpolate(
                                backup_gt_depths.unsqueeze(0),
                                size=depth_pred_size,
                                mode="nearest",
                            ).squeeze(
                                0
                            )  # [B, H', W']
                        backup_gt_masks = (backup_gt_depths > 0).float()  # [B, H', W']

                        backup_loss_depth = si_loss2_jit(
                            backup_gt_depths, backup_depth_pred, backup_gt_masks
                        )
                        backup_loss_depth *= 5 / 2
                        losses.update({"backup_loss_depth": backup_loss_depth})

                        del backup_gt_masks

                    if self.depth_resolution_full:
                        mask_shape = target_masks.shape[-2:]  # [#Q, 1, H, W]
                        depth_pred = F.interpolate(
                            depth_pred.unsqueeze(0),
                            size=mask_shape,
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(
                            0
                        )  # [#Q, H, W]

                loss_depth = si_loss2_jit(gt_depths, depth_pred, gt_masks)
                loss_depth *= 5 / 2

                losses.update({"loss_depth": loss_depth})

                # if self.uncertainty_weight:
                #     losses["loss_depth"] = losses["loss_depth"] / (
                #         2 * self.loss_scale[3].exp()
                #     )

                del gt_masks
                del gt_depths
                del depth_pred
        # -----------------------------------------------------------------------------

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    # def get_loss(self, loss, outputs, targets, indices, num_masks):
    def get_loss(self, loss, outputs, targets, indices, num_masks, depths):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        # return loss_map[loss](outputs, targets, indices, num_masks)
        return loss_map[loss](outputs, targets, indices, num_masks, depths)

    def forward(self, outputs, targets, depths):
        # def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs_without_aux, targets)
        indices = self.matcher(outputs_without_aux, targets, depths)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            # losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_masks, depths)
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # indices = self.matcher(aux_outputs, targets)
                indices = self.matcher(aux_outputs, targets, depths)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_masks, depths
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        # -----------------------------------------------------------------------------

        # if self.uncertainty_weight:
        #     losses["uncertainty"] = 0.5 * (self.loss_scale.sum())

        if self.semantic_guided:
            losses["semantic_guided"] = self.semantic_guided_loss(outputs, targets)

        if self.depth_guided:
            losses["depth_guided"] = self.depth_guided_loss(
                outputs, targets, depths, offsets=[1]
            )

        return losses

    def semantic_guided_loss(self, outputs, targets):
        seg_target = [t["sem_seg"] for t in targets]
        seg_target = torch.stack(seg_target, dim=0).float()  # [B, H, W]

        total_loss = 0
        for s in self.semantic_guided_scales:  # [1/32, 1/16, 1/8, 1/4]
            depth_feat = outputs["depth_feat"][s]
            h, w = depth_feat.shape[-2:]

            seg = F.interpolate(
                seg_target.unsqueeze(1), size=(h, w), mode="nearest"
            )  # [B, 1, H/4, W/4]

            kernel_size = 5
            margin = 0.3
            pad = kernel_size // 2

            center = seg
            padded = F.pad(center, [pad] * 4, value=255)  # [B, 1, H/4 + 2, W/4 + 2]
            aggregated_label = torch.zeros(
                *(center.shape + (kernel_size, kernel_size))
            ).to(
                center.device
            )  # [B, 1, H/4, W/4, 5, 5]

            for i in range(kernel_size):
                for j in range(kernel_size):
                    shifted = padded[:, :, i : i + h, j : j + w]
                    label = shifted == center
                    aggregated_label[:, :, :, :, i, j] = label

            aggregated_label = aggregated_label.float()  # [B, 1, H/4, W/4, 5, 5]
            pos_idx = (aggregated_label == 1).float()  # [B, 1, H/4, W/4, 5, 5]
            neg_idx = (aggregated_label == 0).float()  # [B, 1, H/4, W/4, 5, 5]
            pos_idx_num = pos_idx.sum(dim=(-1, -2))  # [B, 1, H/4, W/4]
            neg_idx_num = neg_idx.sum(dim=(-1, -2))  # [B, 1, H/4, W/4]

            boundary_region = (pos_idx_num >= kernel_size - 1) & (
                neg_idx_num >= kernel_size - 1
            )  # [B, 1, H/4, W/4]
            boundary_region &= center != 255
            # non_boundary_region = (pos_idx_num != 0) & (neg_idx_num == 0)

            affinity = self.compute_affinity(
                depth_feat, kernel_size
            )  # [B, 1, H/4, W/4, 5, 5]

            max_pos_dist = (pos_idx * affinity).amax(dim=(-1, -2))[boundary_region]
            min_neg_dist = (neg_idx * affinity).amin(dim=(-1, -2))[boundary_region]

            zeros = torch.zeros_like(max_pos_dist)
            loss = torch.max(zeros, margin + max_pos_dist - min_neg_dist)

            # pos_dist = (pos_idx * affinity).sum(dim=(-1, -2))[
            #     boundary_region
            # ] / pos_idx_num[boundary_region]
            # neg_dist = (neg_idx * affinity).sum(dim=(-1, -2))[
            #     boundary_region
            # ] / neg_idx_num[boundary_region]
            # zeros = torch.zeros_like(pos_dist)
            # loss = torch.max(zeros, pos_dist - neg_dist + margin)

            total_loss += loss.mean() / (2 ** (3 - s))

        return total_loss

    def compute_affinity(self, feature, kernel_size):
        pad = kernel_size // 2
        feature = F.normalize(feature, dim=1)
        unfolded = (
            F.pad(feature, [pad] * 4)
            .unfold(2, kernel_size, 1)
            .unfold(3, kernel_size, 1)
        )
        feature = feature.unsqueeze(-1).unsqueeze(-1)
        similarity = (feature * unfolded).sum(dim=1, keepdim=True)  # [B, 1, H, W, 5, 5]

        # similarity = similarity * self.gaussian_kernel
        similarity = similarity * torch.sigmoid(self.kernel)

        eps = torch.zeros(similarity.shape).to(similarity.device) + 1e-9
        affinity = torch.max(eps, 2 - 2 * similarity).sqrt()
        return affinity

    def depth_guided_loss(self, outputs, targets, depths, offsets=[1, 2]):
        semantic_feat = outputs["semantic_feat"]
        if self.depth_guided_source == "feature":
            depth_feat = outputs["depth_feat"]
        elif self.depth_guided_source == "gt":
            depth_feat = torch.stack(depths).unsqueeze(1)  # [B, 1, H, W]
        depth_guided_cov = outputs["depth_guided_cov"]

        total_loss = 0
        for s in self.depth_guided_scales:
            s_feat = semantic_feat[s]
            b, c, h, w = s_feat.shape

            if self.depth_guided_source == "feature":
                d_feat = depth_feat[s]
            elif self.depth_guided_source == "gt":
                d_feat = F.interpolate(
                    depth_feat,
                    size=(h, w),
                    mode="nearest",
                )

            cov = depth_guided_cov[(3 - s) * 2 : (3 - s) * 2 + 2]

            s_cov = cov[0]
            d_cov = cov[1]

            for offset in offsets:
                for direction in ["x", "y"]:
                    s_distance = self.mahalanobis_distance(
                        s_feat, s_cov, offset, direction
                    )

                    if self.depth_guided_source == "feature":
                        d_distance = self.mahalanobis_distance(
                            d_feat, d_cov, offset, direction
                        )
                        total_loss += torch.abs(s_distance - d_distance).mean()
                    elif self.depth_guided_source == "gt":
                        d_distance, d_mask = self.mahalanobis_distance(
                            d_feat, d_cov, offset, direction, valid_mask=True
                        )
                        total_loss += (
                            torch.abs(s_distance - d_distance) * d_mask
                        ).mean()

                # total_loss += (torch.abs(s_distance - d_distance) * mask).mean()

                # d_distance, mask = self.mahalanobis_distance(
                #     d_feat, d_cov, offset, "y", True
                # )
                # s_distance = self.mahalanobis_distance(s_feat, s_cov, offset, "y")
                # d_distance = self.mahalanobis_distance(d_feat, d_cov, offset, "y")
                # total_loss += torch.abs(s_distance - d_distance).mean()

                # total_loss += (torch.abs(s_distance - d_distance) * mask).mean()

        return total_loss / len(self.depth_guided_scales) / len(offsets)

    def mahalanobis_distance(self, x, cov, offset, offset_dir="x", valid_mask=False):
        if offset_dir == "x":
            shift_1 = x[:, :, :, :-offset]
            shift_2 = x[:, :, :, offset:]
        elif offset_dir == "y":
            shift_1 = x[:, :, :-offset, :]
            shift_2 = x[:, :, offset:, :]

        diff = shift_1 - shift_2
        b, c, _, _ = diff.shape

        diff = diff.view(b, c, -1)

        if valid_mask:
            with torch.no_grad():
                mask1 = (shift_1 != 0).view(b, 1, -1).float()
                mask2 = (shift_2 != 0).view(b, 1, -1).float()
                mask = (mask1 * mask2) != 0  # [B, 1, HW]
                mask = mask.float()

            diff = diff * mask

        # distance = diff.transpose(1, 2) @ cov @ diff
        # distance = torch.exp(-distance / 2)

        distance = (diff**2) * cov  # [B, C, HW]
        distance = torch.exp(-distance / 2)
        # distance = distance.sum(dim=1)  # [B, HW]
        distance = distance.mean(dim=1)  # [B, HW]

        if valid_mask:
            return distance, mask
        return distance

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(
    coarse_logits,
    uncertainty_func,
    num_points,
    oversample_ratio,
    importance_sample_ratio,
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        num_boxes, dtype=torch.long, device=coarse_logits.device
    )
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(
                    num_boxes, num_random_points, 2, device=coarse_logits.device
                ),
            ],
            dim=1,
        )
    return point_coords
