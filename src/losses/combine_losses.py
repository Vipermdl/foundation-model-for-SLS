import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from .lovasz_losses import lovasz_hinge


class CombineLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super(CombineLoss, self).__init__()
        self.ignore_index = ignore_index
        
    def forward(self, preds, target):
        lovasz_loss = self.symmetric_lovasz(preds, target)
        focal_loss = self.focal_loss(preds, target)
        return lovasz_loss + focal_loss

    def symmetric_lovasz(self, outputs, targets):
        return (lovasz_hinge(outputs, targets, ignore=self.ignore_index) + lovasz_hinge(-outputs, 1 - targets, ignore=self.ignore_index)) / 2
    
    def focal_loss(self, output, target, alpha=1.0, gamma=0.5, OHEM_percent=0.25):        
        output = output.contiguous().view(-1)
        target = target.contiguous().view(-1)

        valid_mask = (target != self.ignore_index).long()
        output, target = output * valid_mask, target * valid_mask

        max_val = (-output).clamp(min=0)
        loss = output - output * target + max_val + ((-max_val).exp() + (-output - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-output * (target * 2 - 1))
        focal_loss = alpha * (invprobs * gamma).exp() * loss 

        # Online Hard Example Mining: top x% losses (pixel-wise). Refer to http://www.robots.ox.ac.uk/~tvg/publications/2017/0026.pdf
        OHEM, _ = focal_loss.topk(k=int(OHEM_percent * [*focal_loss.shape][0]))
        return OHEM.mean() 