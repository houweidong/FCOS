import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_lof_postprocessor
from .loss import make_lof_loss_evaluator
from .display import make_lof_display

from maskrcnn_benchmark.layers import Scale


class LOFHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(LOFHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.LOF.NUM_CLASSES - 1

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.LOF.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.LOF.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        pass


class LOFNewBranchHead(LOFHead):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(LOFNewBranchHead, self).__init__(cfg, in_channels)
        num_lof = cfg.MODEL.LOF.NUM_LOF
        lof_tower = []
        for i in range(cfg.MODEL.LOF.NUM_CONVS):
            lof_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            lof_tower.append(nn.GroupNorm(32, in_channels))
            lof_tower.append(nn.ReLU())

        self.add_module('lof_tower', nn.Sequential(*lof_tower))
        self.lof_tag = nn.Conv2d(
            in_channels, num_lof, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.lof_tower, self.lof_tag]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.1)
                    torch.nn.init.constant_(l.bias, 0)
                    # torch.nn.init.normal_(l.bias, std=1)

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        lof_tag = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            centerness.append(self.centerness(cls_tower))
            bbox_reg.append(torch.exp(self.scales[l](
                self.bbox_pred(self.bbox_tower(feature))
            )))
            lof_tag.append(self.lof_tag(self.lof_tower(feature)))

        return logits, bbox_reg, centerness, lof_tag


class LOFClsBranchHead(LOFHead):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(LOFClsBranchHead, self).__init__(cfg, in_channels)
        num_lof = cfg.MODEL.LOF.NUM_LOF
        self.lof_tag = nn.Conv2d(
            in_channels, num_lof, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for l in self.lof_tag.modules():
            if isinstance(l, nn.Conv2d):
                torch.nn.init.normal_(l.weight, std=0.01)
                torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        lof_tag = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            centerness.append(self.centerness(cls_tower))
            lof_tag.append(self.lof_tag(cls_tower))
            bbox_reg.append(torch.exp(self.scales[l](
                self.bbox_pred(self.bbox_tower(feature))
            )))

        return logits, bbox_reg, centerness, lof_tag


class LOFBboxBranchHead(LOFHead):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(LOFBboxBranchHead, self).__init__(cfg, in_channels)
        num_lof = cfg.MODEL.LOF.NUM_LOF
        self.lof_tag = nn.Conv2d(
            in_channels, num_lof, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for l in self.lof_tag.modules():
            if isinstance(l, nn.Conv2d):
                torch.nn.init.normal_(l.weight, std=0.01)
                torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        lof_tag = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            centerness.append(self.centerness(cls_tower))

            bbox_tower = self.bbox_tower(feature)
            bbox_reg.append(torch.exp(self.scales[l](
                self.bbox_pred(bbox_tower)
            )))
            lof_tag.append(self.lof_tag(bbox_tower))

        return logits, bbox_reg, centerness, lof_tag


def lof_head(cfg, in_channels):
    assert cfg.MODEL.LOF.LOF_TOWER_BRANCH in ['new', 'cls', 'bbox']
    if cfg.MODEL.LOF.LOF_TOWER_BRANCH == 'new':
        return LOFNewBranchHead(cfg, in_channels)
    elif cfg.MODEL.LOF.LOF_TOWER_BRANCH == 'cls':
        return LOFClsBranchHead(cfg, in_channels)
    else:
        return LOFBboxBranchHead(cfg, in_channels)


class LOFModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(LOFModule, self).__init__()

        head = lof_head(cfg, in_channels)

        result_display = make_lof_display(cfg)

        box_selector_test = make_lof_postprocessor(cfg)

        loss_evaluator = make_lof_loss_evaluator(cfg)
        self.head = head
        self.result_display = result_display
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.LOF.FPN_STRIDES
        self.display = cfg.DISPLAY.SWITCH

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness, lof_tag = self.head(features)
        locations = self.compute_locations(features)
 
        if self.training:
            return self._forward_train(
                locations, box_cls, 
                box_regression, 
                centerness,
                lof_tag, targets
            )
        else:
            if self.display:
                self._forward_display(
                    box_cls,
                    box_regression,
                    centerness,
                    lof_tag)
            return self._forward_test(
                locations, box_cls, box_regression,
                centerness, lof_tag, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, lof_tag, targets):
        loss_box_cls, loss_box_reg, loss_centerness, pull_loss, push_loss = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, lof_tag, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness,
            "pull_loss": pull_loss,
            "push_loss": push_loss
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, lof_tag, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, 
            centerness, lof_tag, image_sizes
        )
        return boxes, {}

    def _forward_display(self, box_cls, box_regression, centerness, lof_tag):
        self.result_display(box_cls, box_regression, centerness, lof_tag)

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


def build_lof(cfg, in_channels):
    return LOFModule(cfg, in_channels)
