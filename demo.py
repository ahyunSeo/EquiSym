import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import random
from tqdm import tqdm
from torchvision import transforms
import torch.utils.data as data
from model import *
from utils import *
from config import *
import cv2
import matplotlib.pyplot as plt

def set_seed(global_seed):
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(999)
    np.random.seed(global_seed)
    random.seed(global_seed)

def vis(net, args, test_loaders, device, mode='test', sym_type='reflection'):
    net.eval()
    with torch.no_grad():
        for i, test_loader in enumerate(test_loaders):
            for idx, data in enumerate(tqdm(test_loader)):
                img, im_path = data
                axis_out = net(img=img.to(device), lbl=img.to(device), mask=None, is_syn=False, sym_type=sym_type, vis_only=True)

                img_vis = (unnorm(img)[0])

                axis_max = F.adaptive_max_pool2d(axis_out, (1, 1))
                axis_min = - F.adaptive_max_pool2d(-axis_out, (1, 1))
                axis_out = (axis_out - axis_min) / (axis_max - axis_min)

                axis = axis_out[0][0].cpu()
                axis = axis / axis.max()
                heatmap = cv2.applyColorMap(np.uint8(255 * axis), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                heatmap = np.float32(heatmap) / 255
                img_ = np.float32(img_vis.cpu().permute(1, 2, 0)) #/ 255
                _attended_img = heatmap + np.float32(img_)
                attended_img = _attended_img / np.max(_attended_img)
                im_path = './demo/pred/%s_%s.png' % (im_path[0].split('/')[-1][:-4], sym_type, )
                plt.imsave(im_path, np.clip(attended_img, 0, 1))


if __name__ == '__main__':
    set_seed(1)
    args = get_parser()
    args.sync_bn = True
    comment = str(args.ver)
    print(comment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Let's use", torch.cuda.device_count(), "GPUs!")
   
    if args.rot_data:
        sym_type = 'rotation'
    else:
        sym_type = 'reflection'
    
    net = SymmetryDetectionNetwork(args)
    net = nn.DataParallel(net)
    net.to(device)

    from dendi_loader import CustomSymmetryDatasets

    testset = CustomSymmetryDatasets()
    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

    print('load pretrained model')
    ckpt_path = './weights/v_' + args.ver + '_best_checkpoint.pt'
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint['state_dict'], strict=True)
    if args.eq_cnn:
        net.module.export()
        print('export done')
        net.to(device)
    vis(net, args, (test_loader, ), device, mode='test', sym_type=sym_type)
