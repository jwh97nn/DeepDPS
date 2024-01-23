# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

# from detectron2.projects.point_rend.point_features import point_sample


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


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(batch_dice_loss)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def batch_depth_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    inputs: [Q, H, W]
    targets: [#Q, H, W]
    """
    n = targets.shape[0]
    eps = 1e-6

    if n == 0:
        return torch.zeros((inputs.shape[0], 0), device=inputs.device)

    losses = []
    for i in range(n):
        _target = targets[i : i + 1]  # [1, H, W]
        _target_mask = (_target > 0).float()  # [1, H, W]
        num = torch.sum(_target_mask) + 1

        _input = inputs * _target_mask  # [Q, H, W]

        # Scale-invariant error
        _target = _target.clamp(min=eps)
        _input = _input.clamp(min=eps)
        depth_log_diff = torch.log(_input) - torch.log(_target)  # [Q, H, W]
        scale_invar_log_error_1 = ((depth_log_diff**2) * _target_mask).sum(
            dim=(-1, -2)
        ) / num
        scale_invar_log_error_2 = (
            (depth_log_diff * _target_mask).sum(dim=(-1, -2)) ** 2
        ) / (num**2)
        # losses.append(scale_invar_log_error_1 - scale_invar_log_error_2)

        loss = scale_invar_log_error_1 - 0.85 * scale_invar_log_error_2
        # loss = 10 * torch.sqrt(loss + eps) / 2
        losses.append(loss)

    losses = torch.stack(losses, dim=0)  # [#Q, Q]

    return losses.transpose(1, 0)


batch_depth_loss_jit = torch.jit.script(
    batch_depth_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
        num_points: int = 0,
        cost_depth: float = 1,
        semkitti: bool = False,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_depth = cost_depth
        self.semkitti = semkitti

        assert (
            cost_class != 0 or cost_mask != 0 or cost_dice != 0
        ), "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    # def memory_efficient_forward(self, outputs, targets):
    def memory_efficient_forward(self, outputs, targets, depths):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        if self.cost_depth == 0:
            include_depth = False
        else:
            output_depth = outputs["pred_depths"][0]
            include_depth = output_depth is not None

        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(
                -1
            )  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)  # [#Q, H_gt, W_gt]

            out_mask = out_mask[:, None]  # [num_queries, 1, H_pred, W_pred]
            tgt_mask = tgt_mask[:, None]  # [#Q, 1, H_gt, W_gt]

            # ----------------------- Depth BEGIN-----------------------
            if include_depth:
                with autocast(enabled=False):
                    out_depth = output_depth[b]  # [Q, H_pred, W_pred]
                    tgt_depth = depths[b]  # [H_gt, W_gt]
                    tgt_depth = F.interpolate(
                        tgt_depth.unsqueeze(0).unsqueeze(0),
                        size=out_depth.shape[-2:],
                        mode="nearest",
                    ).squeeze(
                        0
                    )  # [1, H_pred, W_pred]

                    _tgt_mask = F.interpolate(
                        tgt_mask,
                        size=out_mask.shape[-2:],
                        mode="nearest",
                    ).squeeze(
                        1
                    )  # [#Q, H_pred, W_pred]

                    tgt_depth = tgt_depth * _tgt_mask  # [#Q, H_pred, W_pred]

                    cost_depth = batch_depth_loss_jit(out_depth, tgt_depth)  # [Q, #Q]
            # ----------------------- Depth END ------------------------

            if self.semkitti:
                valid_mask = targets[b]["valid_mask"].squeeze(0)  # [1, H/4, W/4]

                point_labels = []
                for i in range(tgt_mask.shape[0]):
                    point_labels.append(tgt_mask[i][valid_mask])
                tgt_mask = torch.stack(point_labels, dim=0)  # [#Q, #P]

                point_logits = []
                for i in range(out_mask.shape[0]):
                    point_logits.append(out_mask[i][valid_mask])
                out_mask = torch.stack(point_logits, dim=0)  # [Q, #P]
            else:
                # all masks share the same set of points for efficient matching!
                point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
                # get gt labels
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()  # [Q, #P]
                tgt_mask = tgt_mask.float()  # [#Q, #P]

                if tgt_mask.shape[0] == 0:
                    cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                    cost_dice = batch_dice_loss(out_mask, tgt_mask)
                else:
                    # Compute the focal loss between masks
                    cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                    # Compute the dice loss betwen masks
                    cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )

            if include_depth:
                C = C + self.cost_depth * cost_depth

            C = C.reshape(num_queries, -1).cpu()  # [Q, #Q]

            indices.append(linear_sum_assignment(C))

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]

    @torch.no_grad()
    # def forward(self, outputs, targets):
    def forward(self, outputs, targets, depths):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # return self.memory_efficient_forward(outputs, targets)
        return self.memory_efficient_forward(outputs, targets, depths)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
