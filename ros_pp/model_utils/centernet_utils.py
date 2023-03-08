# This file is modified from https://github.com/tianweiy/CenterPoint

import torch
import torch.nn.functional as F
from typing import Tuple
from torchvision.ops import nms


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    batch, num_class, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.flatten(2, 3), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_classes = (topk_ind // K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_classes, topk_ys, topk_xs


def oriented_boxes_to_axis_aligned(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert oriented boxes to axis-aligned (in BEV) to be used in axis-aligned NMS

    Args:
        boxes: (N, 7) - x, y, z, dx, dy, dz, yaw
    """
    # construct corners in boxes' local coord
    corners_x = torch.tensor([1, 1, -1, -1], dtype=boxes.dtype, device=boxes.device)
    corners_y = torch.tensor([-1, 1, 1, -1], dtype=boxes.dtype, device=boxes.device)
    corners = torch.stack([corners_x, corners_y], dim=1)  # (4, 2)

    corners = corners.unsqueeze(0) * boxes[:, [3, 4]].unsqueeze(1) / 2.0  # (N, 4, 2)

    # map corners to world coord (frame where boxes are expressed w.r.t)
    cos, sin = torch.cos(boxes[:, -1]), torch.sin(boxes[:, -1])  # (N,), (N,)
    rot = torch.stack([cos, -sin,
                       sin, cos], dim=1).contiguous().view(boxes.shape[0], 2, 2)
    
    corners = torch.matmul(corners, rot.permute(0, 2, 1)) + boxes[:, :2].unsqueeze(1)  # (N, 4, 2)
    
    xy_min = torch.min(corners, dim=1)[0]  # (N, 2)
    xy_max = torch.max(corners, dim=1)[0]  # (N, 2)
    out = torch.cat([xy_min, xy_max], dim=1)  # (N, 4)

    return out


# =========================================================================================
# =========================================================================================


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Src: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/43fd8be9e82b351619a467373d211ee5bf73cef8/utils.py#L356
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


# =========================================================================================
# =========================================================================================


def nms_axis_aligned(boxes: torch.Tensor, boxes_score: torch.Tensor, boxes_label: torch.Tensor, max_overlap: float,
                     nms_pre_max_size: int, nms_post_max_size: int) \
    -> Tuple[torch.Tensor]:
    """
    Src: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py#L480
    Args:
        boxes: (N, 7) - x, y, z, dx, dy, dz, yaw
        boxes_score: (N,)
        boxes_label: (N,)
    
    Returns:
        boxes: (M, 7) - NOTE: M < N
        boxes_score: (M,)
        boxes_label: (M,)
    """
    device = boxes.device
    
    # Sort predicted boxes and scores by scores
    boxes_score, sort_ind = boxes_score.sort(dim=0, descending=True)  # (N), (N)
    boxes = boxes[sort_ind]  # (N, 7)
    boxes_label = boxes_label[sort_ind]  # (N,)

    # boxes = boxes[:nms_pre_max_size]
    # boxes_score = boxes_score[:nms_pre_max_size]
    # boxes_label = boxes_label[:nms_pre_max_size]

    # # convert boxes to axis-align format
    # boxes_axis_aligned = oriented_boxes_to_axis_aligned(boxes)  # (N, 4) - x_min, y_min, x_max, y_max

    # keep_idx = nms(boxes_axis_aligned, boxes_score, max_overlap)
    # boxes, boxes_score, boxes_label = boxes[keep_idx], boxes_score[keep_idx], boxes_label[keep_idx]

    boxes = boxes[:nms_post_max_size]
    boxes_score = boxes_score[:nms_post_max_size]
    boxes_label = boxes_label[:nms_post_max_size]

    return boxes, boxes_score, boxes_label
