import torch
from torch import nn


class AELoss(nn.Module):
    def __init__(self, pull_factor, push_factor):
        super(AELoss, self).__init__()
        self.pull_factor = pull_factor
        self.push_factor = push_factor

    def forward(self, lof_tag_img, lof_tag_avg_img, lof_tag_avg_gather_img, mask, centerness_img=None):
        tag = torch.pow(lof_tag_img - torch.round(lof_tag_avg_gather_img), 2)

        dist = lof_tag_avg_img.unsqueeze(0) - lof_tag_avg_img.unsqueeze(1)
        dist = 1 - torch.abs(dist)
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
