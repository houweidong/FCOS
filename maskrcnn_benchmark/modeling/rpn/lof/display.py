import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_lof_nms, boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes
from matplotlib import pyplot as plt


class LOFDispaly(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        # pre_nms_thresh,
        # pre_nms_top_n,
        # nms_thresh,
        # fpn_post_nms_top_n,
        display_cls,
        display_cls_id,
        display_ctn,
        display_lof,
        min_size,
        num_classes,
        num_lof,
        distance,
        version
        # use_lof
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(LOFDispaly, self).__init__()
        # self.pre_nms_thresh = pre_nms_thresh
        # self.pre_nms_top_n = pre_nms_top_n
        # self.nms_thresh = nms_thresh
        # self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.display_cls = display_cls
        self.display_cls_id = display_cls_id
        self.display_ctn = display_ctn
        self.display_lof = display_lof
        self.min_size = min_size
        self.num_classes = num_classes
        self.num_lof = num_lof
        self.distance = distance
        self.version = version
        # self.use_lof = use_lof

    def forward_for_single_feature_map(self, box_cls, box_regression, centerness, lof_tag, axes):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape

        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1).sigmoid()

        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1).sigmoid().squeeze()

        if self.version == 1:
            lof_tag = lof_tag.view(N, self.num_lof, H, W).permute(0, 2, 3, 1).squeeze()
        elif self.version == 2:
            lof_tag = torch.round(lof_tag.view(N, self.num_lof, H, W).permute(0, 2, 3, 1).sigmoid())
            exp = torch.arange(self.num_lof, device=lof_tag.device)[None, None, None, :].repeat((N, H, W, 1)).float()
            factor = torch.pow(2, exp)
            lof_tag = (lof_tag * factor).sum(-1).squeeze(0)

        # lof_tag = torch.round(lof_tag.view(N, self.num_lof, H, W).permute(0, 2, 3, 1)).int().squeeze()
        box_cls_id = (torch.argmax(box_cls, dim=-1) + 1).squeeze()
        box_cls = torch.sqrt(torch.max(box_cls, dim=-1)[0].squeeze() * centerness)
        start = 0
        if self.display_cls:
            axes[start].matshow(box_cls.detach().cpu())
            start += 1
        if self.display_cls_id:
            axes[start].matshow(box_cls_id.detach().cpu())
            start += 1
        if self.display_ctn:
            axes[start].matshow(centerness.detach().cpu())
            start += 1
        if self.display_lof:
            axes[start].matshow(lof_tag.detach().cpu())
            start += 1

    def forward(self, box_cls, box_regression, centerness, lof_tag):
        rows = self.display_cls + self.display_cls_id + self.display_ctn + self.display_lof
        fig, axes = plt.subplots(nrows=rows, ncols=5)
        display_tile = []
        for name, indicator in zip(['class', 'class_id', 'centerness', 'lof'],
                                   [self.display_cls, self.display_cls_id, self.display_ctn, self.display_lof]):
            if indicator:
                display_tile.append(name)
        for i in range(rows):
            for j in range(5):
                axes[i, j].set(title=display_tile[i] + " : %d level" % (j+1))
        for j, (o, b, c, lt) in enumerate(zip(box_cls, box_regression, centerness, lof_tag)):
            self.forward_for_single_feature_map(o, b, c, lt, axes[:, j])


def make_lof_display(config):
    # pre_nms_thresh = config.MODEL.LOF.INFERENCE_TH
    # pre_nms_top_n = config.MODEL.LOF.PRE_NMS_TOP_N
    # nms_thresh = config.MODEL.LOF.NMS_TH
    # fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    display_cls = config.DISPLAY.CLASS
    display_cls_id = config.DISPLAY.CLASS_ID
    display_ctn = config.DISPLAY.CENTERNESS
    display_lof = config.DISPLAY.LOF

    result_dispaly = LOFDispaly(
        # pre_nms_thresh=pre_nms_thresh,
        # pre_nms_top_n=pre_nms_top_n,
        # nms_thresh=nms_thresh,
        # fpn_post_nms_top_n=fpn_post_nms_top_n,
        display_cls=display_cls,
        display_cls_id=display_cls_id,
        display_ctn=display_ctn,
        display_lof=display_lof,
        min_size=0,
        num_classes=config.MODEL.FCOS.NUM_CLASSES,
        num_lof=config.MODEL.LOF.NUM_LOF,
        distance=config.MODEL.LOF.DISTANCE,
        version=config.MODEL.LOF.VERSION
        # use_lof=config.MODEL.LOF.USE_LOF
    )

    return result_dispaly
