# The edge code refers to 'Non-Local Deep Features for Salient Object Detection', CVPR 2017.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)
fx = np.reshape(fx, (1, 1, 3, 3))
fy = np.reshape(fy, (1, 1, 3, 3))
fx = Variable(torch.from_numpy(fx)).cuda()
fy = Variable(torch.from_numpy(fy)).cuda()
contour_th = 1.5


def label_edge_prediction(label):
    # convert label to edge
    label = label.gt(0.5).float()
    label = F.pad(label, (1, 1, 1, 1), mode='replicate')
    label_fx = F.conv2d(label, fx)
    label_fy = F.conv2d(label, fy)
    label_grad = torch.sqrt(torch.mul(label_fx, label_fx) + torch.mul(label_fy, label_fy))
    label_grad = torch.gt(label_grad, contour_th).float()

    return label_grad


def pred_edge_prediction(pred):
    # infer edge from prediction
    pred = F.pad(pred, (1, 1, 1, 1), mode='replicate')
    pred_fx = F.conv2d(pred, fx)
    pred_fy = F.conv2d(pred, fy)
    pred_grad = (pred_fx*pred_fx + pred_fy*pred_fy).sqrt().tanh()

    return pred_fx, pred_fy, pred_grad


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return np.mean(self.losses[np.maximum(len(self.losses)-self.num, 0):])