"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn

from ..utils import concat_box_prediction_layers
from maskrcnn_benchmark.layers import IOULoss
from maskrcnn_benchmark.layers import SigmoidFocalLoss
from maskrcnn_benchmark.layers import AELoss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist


INF = 100000000


class LOFLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.LOF.LOSS_GAMMA,
            cfg.MODEL.LOF.LOSS_ALPHA
        )
        self.lof_loss_func = AELoss(
            cfg.MODEL.LOF.PULL_FACTOR,
            cfg.MODEL.LOF.PUSH_FACTOR,
            cfg.MODEL.LOF.DISTANCE,
            cfg.MODEL.LOF.MARGIN_PUSH
        )
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.avg_weight = cfg.MODEL.LOF.AVG_WEIGHT
        self.avg_lof_loss = cfg.MODEL.LOF.AVG_LOF_LOSS

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets, locations_to_gtis = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        labels_level_batch = torch.cat([label[None] for label in labels], dim=0)
        reg_targets_level_batch = torch.cat([reg_target[None] for reg_target in reg_targets], dim=0)
        locations_to_gtis_batch = torch.cat([locations_to_gti[None] for locations_to_gti in locations_to_gtis], dim=0)
        return labels_level_batch, torch.clamp(reg_targets_level_batch, 0.), locations_to_gtis_batch

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        locations_to_gt_indss = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_aera, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_aera == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            locations_to_gt_indss.append(locations_to_gt_inds)

        return labels, reg_targets, locations_to_gt_indss

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[..., [0, 2]]
        top_bottom = reg_targets[..., [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def compute_lof_tag_targets(self, lof_tag, labels, reg_targets, locations_to_gtis):
        max_batch = torch.max(locations_to_gtis, dim=1)[0]
        max_num_bbox = torch.max(locations_to_gtis).item() + 1

        lof_tag = torch.zeros_like(lof_tag)[..., None].repeat(
            1, 1, max_num_bbox).scatter_(-1, locations_to_gtis[..., None], lof_tag[..., None])
        indicator = (torch.zeros_like(lof_tag).scatter_(
            -1, locations_to_gtis[..., None], labels[..., None].float()) > 0).float()
        lof_tag = lof_tag * indicator
        centerness = self.compute_centerness_targets(reg_targets)
        if self.avg_weight:
            centerness_box = centerness[..., None].repeat(1, 1, max_num_bbox) * indicator
            sum_weight = centerness_box.sum(dim=1, keepdim=True) + 1e-8
            weight = centerness_box / sum_weight
            avg_lof_tag = (lof_tag * weight).sum(1, keepdim=True)
        else:
            num_bbox = indicator.sum(1, keepdim=True) + 1e-8
            avg_lof_tag = lof_tag.sum(1, keepdim=True) / num_bbox
        len_pyramid_feature = lof_tag.size(1)
        avg_lof_tag_gather = torch.round(torch.gather(avg_lof_tag.repeat((1, len_pyramid_feature, 1)),
                                         -1, locations_to_gtis[..., None]).squeeze(-1))
        return avg_lof_tag.squeeze(1), avg_lof_tag_gather, max_batch, centerness

    def __call__(self, locations, box_cls, box_regression, centerness, lof_tag, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            lof_tag (list[Tensor)
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
            lof_tag_loss (Tensor)
        """
        N, num_classes = box_cls[0].size(0), box_cls[0].size(1)
        labels, reg_targets, locations_to_gtis = self.prepare_targets(locations, targets)
        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        lof_tag_list = []
        for l in range(len(box_cls)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(N, -1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(N, -1, 4))
            centerness_flatten.append(centerness[l].reshape(N, -1))
            lof_tag_list.append(lof_tag[l].reshape(N, -1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=1).reshape((-1, num_classes))
        box_regression_flatten = torch.cat(box_regression_flatten, dim=1).reshape((-1, 4))
        centerness_flatten = torch.cat(centerness_flatten, dim=1).reshape((-1, ))
        lof_tag = torch.cat(lof_tag_list, dim=1)

        pos_inds = torch.nonzero(labels.reshape((-1, )) > 0).squeeze(1)
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels.reshape((-1, )).int()
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets.reshape((-1, 4))[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
            lof_tag_avg, lof_tag_avg_gather, max_batch, centerness_targets_all = \
                self.compute_lof_tag_targets(lof_tag, labels, reg_targets, locations_to_gtis)
            pull_loss, push_loss = torch.zeros_like(centerness_loss), torch.zeros_like(centerness_loss)
            for batch_index in range(N):
                select_mask = torch.nonzero(labels[batch_index] > 0).squeeze(-1)
                if select_mask.numel() > 0:

                    # prepare some data for lof loss computation
                    lof_tag_img = lof_tag[batch_index][select_mask]
                    num_box = max_batch[batch_index] + 1
                    lof_tag_avg_img = lof_tag_avg[batch_index, :num_box]
                    lof_tag_avg_gather_img = lof_tag_avg_gather[batch_index][select_mask]
                    centerness_img = centerness_targets_all[batch_index][select_mask]
                    mask = torch.arange(num_box).unsqueeze(0) != torch.arange(num_box).unsqueeze(1)

                    pull, push = self.lof_loss_func(lof_tag_img, lof_tag_avg_img, lof_tag_avg_gather_img,
                                                    mask, torch.sqrt(centerness_img))
                    pull_loss += pull / N
                    push_loss += push / N

        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()
            pull_loss = torch.zeros_like(centerness_loss)
            push_loss = torch.zeros_like(centerness_loss)

        return cls_loss, reg_loss, centerness_loss, pull_loss, push_loss


def make_lof_loss_evaluator(cfg):
    loss_evaluator = LOFLossComputation(cfg)
    return loss_evaluator
