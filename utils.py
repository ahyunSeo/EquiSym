import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

def draw_axis(lines, size):
    axis = Image.new('L', size)
    # w, h = img.size
    draw = ImageDraw.Draw(axis)
    length = np.array([size[0], size[1], size[0], size[1]])
    
    # x1, y1, x2, y2
    line_coords = []

    for idx, coords in enumerate(lines):
        if coords[0] > coords[2]:
            coords = np.roll(coords, -2)
        draw.line(list(coords), fill=(idx + 1))
        coords = np.array(coords).astype(np.float32)
        _line_coords = coords / length
        line_coords.append(_line_coords)
    axis = np.asarray(axis).astype(np.float32)
    return axis, line_coords

def match_input_type(img):
    img = np.asarray(img)
    if img.shape[-1] != 3:
        img = np.stack((img, img, img), axis=-1)
    return img 
    
def norm(img):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img = (img - mean) / std
    return img

def unnorm(img):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img = img * std + mean
    return img

##########################
### Train ################
##########################

def sigmoid_focal_loss(
    source: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
    is_logits=True
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    if is_logits:
        p = nn.Sigmoid()(source)
        ce_loss = F.binary_cross_entropy_with_logits(
            source, targets, reduction="none"
        )
    else:
        p = source
        ce_loss = F.binary_cross_entropy(source, targets, reduction="none")
    
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // (args.num_epochs * 0.5))) * (0.1 ** (epoch // (args.num_epochs * 0.75)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


##########################
### Evaluation ###########
##########################

class PointEvaluation(object):
    def __init__(self, n_thresh=100, max_dist=5, blur_pred=False):
        self.n_thresh = n_thresh
        self.max_dist = max_dist
        self.thresholds = torch.linspace(1.0 / (n_thresh + 1),
                            1.0 - 1.0 / (n_thresh + 1), n_thresh)
        self.filters = self.make_gt_filter(max_dist)
        self.tp = torch.zeros((n_thresh,))
        self.pos_label = torch.zeros((n_thresh,))
        self.pos_pred = torch.zeros((n_thresh,))
        self.num_samples = 0
        self.blur_pred = blur_pred
    
    def make_gt_filter(self, max_dist):
        # expand 1-pixel gt to a circle w/ radius of max_dist
        ks = max_dist * 2 + 1
        filters = torch.zeros(1, 1, ks, ks)
        for i in range(ks):
            for j in range(ks):
                dist = (i-max_dist) ** 2 + (j-max_dist) ** 2
                if dist <= max_dist**2:
                    filters[0, 0, i, j] = 1
        return filters
    
    def f1_score(self,):
        # If there are no positive pixel detections on an image,
        # the precision is set to 0. (funk and liu)
        precision = torch.where(self.pos_pred > 0, self.tp / self.pos_pred, torch.zeros(1))
        recall = self.tp / self.pos_label
        numer = precision + recall
        f1 = torch.where(numer > 0, 2 * precision * recall / numer, torch.zeros(1))
        return precision, recall, f1
        
    def __call__ (self, pred, gt):
        pred = pred.cpu()
        gt = F.conv2d(gt, self.filters, padding=self.max_dist)
        gt = (gt > 0).float()
        pos_label = gt.float().sum(dim=(2, 3))
        self.num_samples = self.num_samples + pred.shape[0]
        # Evaluate predictions (B, 1, H, W)
        for idx, th in enumerate(self.thresholds):
            _pred = (pred > th).float() # binary

            if self.blur_pred:
                _pred = F.conv2d(_pred, self.filters, padding=self.max_dist)
                _pred = (_pred > 0).float()

            tp = ((gt * _pred) > 0).float().sum(dim=(2, 3))
            pos_pred = _pred.sum(dim=(2, 3))
            
            self.tp[idx] += tp.sum()
            self.pos_pred[idx] += pos_pred.sum()
            self.pos_label[idx] += pos_label.sum()

