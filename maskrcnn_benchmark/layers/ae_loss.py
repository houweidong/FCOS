import torch
from torch import nn


class AELoss(nn.Module):
    def __init__(self, pull_factor, push_factor, distance, margin_push):
        super(AELoss, self).__init__()
        self.pull_factor = pull_factor
        self.push_factor = push_factor
        self.distance = distance
        self.margin_push = margin_push

    def forward(self, lof_tag_img, lof_tag_avg_img, lof_tag_avg_gather_img, mask, centerness_img=None):
        # lof_tag_img               shape (selected 5level)
        # lof_tag_avg_img           shape (num_boxes)
        # lof_tag_avg_gather_img    shape (selected 5level)
        lof_tag_avg_gather_img = torch.round(lof_tag_avg_gather_img / self.distance) * self.distance
        tag = torch.pow(lof_tag_img - torch.round(lof_tag_avg_gather_img), 2)

        dist = lof_tag_avg_img.unsqueeze(0) - lof_tag_avg_img.unsqueeze(1)
        dist = self.distance + self.margin_push - torch.abs(dist)
        dist = nn.functional.relu(dist, inplace=True)
        dist = dist[mask]

        if centerness_img is not None:
            pull = (tag * centerness_img).sum() / centerness_img.sum()
            push = torch.zeros_like(pull)
            if mask.any():
                # centerness = (centerness_img.unsqueeze(0) * centerness_img.unsqueeze(1))[mask]
                # push = (dist * centerness).sum() / centerness.sum()
                push = dist.sum() / mask.sum().float()

        else:
            pull = tag.mean()
            push = dist.mean()

        return self.pull_factor*pull, self.push_factor*push


class AELossV2(nn.Module):
    def __init__(self, pull_factor, push_factor, margin_push, num_lof):
        super(AELossV2, self).__init__()
        self.pull_factor = pull_factor
        self.push_factor = push_factor
        self.distance = 0.5
        self.margin_push = margin_push
        self.tag_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.num_lof = num_lof

    def forward(self, lof_tag_img, lof_tag_avg_img, lof_tag_avg_gather_img, mask, centerness_img=None):
        # lof_tag_img               shape (selected 5level, num_lof)
        # lof_tag_avg_img           shape (num_lof, num_boxes)
        # lof_tag_avg_gather_img    shape (selected 5level, num_lof)
        # centerness_img            shape (selected 5level, num_lof)
        lof_tag_avg_gather_img = torch.round(torch.sigmoid(lof_tag_avg_gather_img))
        # tag = torch.pow(lof_tag_img - torch.round(lof_tag_avg_gather_img), 2)
        tag = self.tag_loss(lof_tag_img, lof_tag_avg_gather_img)

        dist = (0.5 + self.margin_push) - torch.abs(torch.sigmoid(lof_tag_avg_img.unsqueeze(1))
                                                    - torch.sigmoid(lof_tag_avg_img.unsqueeze(2)))
        dist_mask = ((dist < (0.5 + self.margin_push)).sum(0, keepdim=True) > 0)
        mask = dist_mask & mask
        # dist = dist_mask * dist
        dist = dist[mask]

        if centerness_img is not None:
            pull = (tag * centerness_img).sum() / centerness_img.sum()
            push = torch.zeros_like(pull)
            if mask.any():
                # centerness = (centerness_img.unsqueeze(0) * centerness_img.unsqueeze(1))[mask]
                # push = (dist * centerness).sum() / centerness.sum()
                push = dist.sum() / mask.sum().float()

        else:
            pull = tag.mean()
            push = dist.mean()

        return self.pull_factor*pull, self.push_factor*push
