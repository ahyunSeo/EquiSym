import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.module import build_decoder, build_aspp, build_backbone

import pdb
from e2cnn import gspaces
from e2cnn import nn as enn
    
class SymmetryDetectionNetwork(nn.Module):
    def __init__(self, args=None):
        super(SymmetryDetectionNetwork, self).__init__()
        self.args = args
        self.sync_bn = args.sync_bn
        self.backbone_model = args.backbone
        self.output_stride = 8
        self.freeze_bn = False
        self.num_classes = 1
        self.n_angle = args.n_angle
        self.angle_interval = 360 / self.n_angle
        self.eq_cnn = args.eq_cnn
        self.get_theta = args.get_theta

        if self.sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        res_featdim = {18: 512, 34: 512, 50:2048, 101: 2048}
        featdim = 256
        out_indices = (3,)
        last_convout = 256
        last_conv_only = True
        score_dim = 0
        last_convin = featdim + score_dim
        
        if self.eq_cnn:
            global gspace_orientation, gspace_flip, cutoff_dilation
            gspace_orientation = 8 #self.n_angle
            gspace_flip = 1
            cutoff_dilation = 0
            from modeling.eq_resnet import ReResNet

            self.backbone = ReResNet(depth=self.args.depth, strides=[1, 2, 1, 1], dilations=[1, 1, 2, 4], out_indices=out_indices)

            if args.load_eq_pretrained:
                print('loading pretrained e2cnn backbone')
                # hardcode for now
                eq_ckpt = torch.load(args.eq_model_dir)['state_dict']
                eq_ckpt = {k:v for k, v in eq_ckpt.items() if 'head' not in k}
                temp = self.backbone.state_dict()
                for k, v in temp.items():
                    # init -> eval mode
                    if 'filter' in k:
                        eq_ckpt[k] = v
                self.backbone.load_state_dict(eq_ckpt)
                print('done')
                
        else:
            self.backbone_model = 'resnet%d' % args.depth
            self.backbone = build_backbone(self.backbone_model, self.output_stride, BatchNorm, False)

        if self.get_theta in [10]:
            theta_convin = featdim + score_dim

            if args.rot_data:
                n_angle = args.n_rot
            else:
                n_angle = self.n_angle

            if self.eq_cnn:
                from modeling.eq_module import Decoder
                self.decoder_theta = Decoder(n_angle + 1, self.backbone_model, BatchNorm, last_conv_only, theta_convin, last_convout, use_bn=True, restrict=False)
            else:
                self.decoder_theta = build_decoder(n_angle + 1, self.backbone_model, BatchNorm, last_conv_only, theta_convin, last_convout, use_bn=True)

        if self.eq_cnn:
            from modeling.eq_module import ASPP
            self.aspp = ASPP(self.output_stride, in_channels=res_featdim[args.depth])
        else:
            self.aspp = build_aspp(self.backbone_model, self.output_stride, BatchNorm, inchannels=res_featdim[args.depth])

        if self.eq_cnn:
            if self.get_theta in [10]:
                from modeling.eq_module import SingleDecoder
                from modeling.eq_resnet import trivial_feature_type
                in_type = self.decoder_theta.conv3.out_type
                in_type = in_type + trivial_feature_type(in_type.gspace, 1, fixparams=False)
                self.decoder_axis = SingleDecoder(in_type, self.num_classes)
            else:
                from modeling.eq_module import Decoder
                self.decoder_axis = Decoder(self.num_classes, self.backbone_model, BatchNorm, last_conv_only, last_convin, last_convout, use_bn=True)
        else:
            self.decoder_axis = build_decoder(self.num_classes, self.backbone_model, BatchNorm, last_conv_only, last_convin, last_convout, use_bn=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, lbl, mask, is_syn, a_lbl=None, sym_type='reflection', vis_only=False):
        ### [Step 1]: Base feature extraction with ENC
        feat = self.backbone(img)
        feat = self.aspp(feat)

        if self.get_theta in [10]:
            ### [Step 2a]: Apply auxiliary DEC
            theta_out = self.decoder_theta(None, feat)
            if not torch.is_tensor(theta_out):
                _theta_out = theta_out.tensor
                theta_sum = torch.softmax(_theta_out, dim=1)[:, 1:, :, :].sum(dim=1, keepdim=True)
            else:
                theta_sum = torch.softmax(theta_out, dim=1)[:, 1:, :, :].sum(dim=1, keepdim=True)
            theta_sum = torch.clamp(theta_sum, min=0+1e-5, max=1-1e-5)
            ### [Step 2b]: Apply final DEC
            out = self.decoder_axis(theta_sum, theta_out)
            if not torch.is_tensor(theta_out):
                theta_out = theta_out.tensor
                out = out.tensor
        else:
            ### [Step 2]: Apply final DEC
            if self.eq_cnn:
                out = self.decoder_axis(None, feat)
                if not torch.is_tensor(out):
                    out = out.tensor
            else:
                out = self.decoder_axis(None, feat)

        ### [Step 3a]: localization loss
        axis_out = F.interpolate(out, size=lbl.size()[2:], mode='bilinear', align_corners=True)
        if vis_only:
            return self.sigmoid(axis_out) # vis purpose
        axis_loss = utils.sigmoid_focal_loss(axis_out, lbl, alpha=0.95)
        axis_out = self.sigmoid(axis_out) # vis purpose
        loss = axis_loss
        losses = (axis_loss, axis_loss)

        if self.get_theta:
            ### [Step 3b]: classification loss
            a_lbl = F.max_pool2d(a_lbl, kernel_size=5, stride=1, padding=2)
            theta_out = F.interpolate(theta_out, size=a_lbl.size()[2:], mode='bilinear', align_corners=True)

            weight = torch.ones(theta_out.shape[1])
            weight[0] = self.args.theta_loss_weight
            weight = weight.view(1, -1, 1, 1).to(a_lbl.device)

            fg_mask = (a_lbl.sum(dim=1) > 0).long()
            a_lbl = fg_mask * (torch.argmax(a_lbl, dim=1)+1)
            theta_loss = F.cross_entropy(theta_out, a_lbl, weight=weight, reduction='none') / weight.sum(dim=1, keepdim=True)
            theta_loss = theta_loss.mean(dim=1, keepdim=True)

            loss += theta_loss
            losses = (axis_loss, theta_loss)

            return axis_out, theta_out, loss, losses

        return axis_out, feat, loss, losses

    def export(self):
        ### For fast inference (use this only at the test time)
        ### For more information about 'export' ,
        ### refer to https://quva-lab.github.io/e2cnn/api/e2cnn.nn.html?highlight=export#e2cnn.nn.EquivariantModule
        self.backbone = self.backbone.export()
        self.decoder_theta = self.decoder_theta.export()
        self.decoder_axis = self.decoder_axis.export()
        self.aspp = self.aspp.export()