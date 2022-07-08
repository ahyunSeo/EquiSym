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
import wandb
from copy import deepcopy

def set_seed(global_seed):
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(999)
    np.random.seed(global_seed)
    random.seed(global_seed)

def train(net, args, train_loader, val_loaders, test_loaders, device, mode='synth'):
    param_groups = net.parameters()
    max_f1 = 0.0
    start_epoch = 1
    optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    if args.rot_data:
        sym_type = 'rotation'
    else:
        sym_type = 'reflection'

    for epoch in range(start_epoch, args.num_epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        net.train()
        print('%s image train' % mode)
        print(epoch)
        running_sample = 0
        running_loss = 0
        running_theta_loss = 0
        train_log_images = []
        
        for idx, data in enumerate(tqdm(train_loader)):
            if args.get_theta:
                img, mask, axis, axis_gs, is_syn, a_lbl = data
                a_lbl = a_lbl.to(device)
            else:
                img, mask, axis, axis_gs, is_syn, _ = data
                a_lbl = None

            _mask = (mask > 0).float().to(device)
            axis_out, mask_out, total_loss, losses = net(img=img.to(device), lbl=axis_gs.float().to(device), mask=_mask, is_syn = is_syn.to(device), a_lbl=a_lbl)
            loss = total_loss.mean()

            if a_lbl is not None:
                loss += losses[1].mean() * args.theta_loss_scale
                running_theta_loss += losses[1].mean().item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            running_sample += 1
            running_loss += loss.item()

            if (idx == len(train_loader) - 1) and not args.wandb_off:
                log_dict = {"Train Loss": running_loss / running_sample,}
                if a_lbl is not None:
                    log_dict['Train theta Loss'] = running_theta_loss / running_sample

                running_sample = 0
                running_loss = 0
                running_theta_loss = 0

                train_log_images = [
                    wandb.Image(unnorm(img), caption='Image/train_%d_image' % idx),
                    wandb.Image(axis_gs.float(),  caption='Image/train_%d_axisGT' % idx),              
                    wandb.Image(axis_out.cpu(), caption='Image/train_%d_axisPred' % idx),
                ]
                        
                log_dict['train_log_images'] = train_log_images
                wandb.log(log_dict)
        
        if epoch % 5 == 0:
            rec, prec, f1, f1_max = test(net, args, val_loaders, device, mode='val', sym_type=sym_type)  # subset of testset

            _max_f1 = f1_max[0]
            max_f1 = max(max_f1, _max_f1)

            checkpoint = {
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': args,
                    'epoch': epoch,
                    'max_f1': max_f1,
                }
            if not os.path.exists('./weights'):
                os.makedirs('./weights')
            torch.save(checkpoint, './weights/v_' + args.ver + '_last_checkpoint.pt')
            print(max_f1)

            if max_f1 == _max_f1:
                print('best model renewed.')
                best_ckpt = deepcopy(net.state_dict())
                torch.save(checkpoint, './weights/v_' + args.ver + '_best_checkpoint.pt')

    net.load_state_dict(best_ckpt)
    rec, prec, f1, f1_max  = test(net, args, test_loaders, device, mode='test', sym_type=sym_type)
    checkpoint['stats'] = rec, prec, f1, f1_max
    print(f1_max)
    torch.save(checkpoint, './weights/v_' + args.ver + '_best_checkpoint.pt')

def test(net, args, test_loaders, device, mode='test', sym_type='reflection'):
    net.eval()
    val_sample, val_loss = 0, 0

    n_thresh = 100 if mode in ['test', 'ref_test', 'rot_test'] else 10

    recs, precs, f1s, f1_maxs= [], [], [], []

    with torch.no_grad():
        for i, test_loader in enumerate(test_loaders):
            
            axis_eval = PointEvaluation(n_thresh, blur_pred=True, device=device)
            val_log_images = []

            for idx, data in enumerate(tqdm(test_loader)):
                if args.get_theta:
                    img, mask, axis, axis_gs, is_syn, a_lbl = data
                    a_lbl.to(device)
                else:
                    img, mask, axis, axis_gs, is_syn, a_lbl = data
                    # a_lbl = None
                    a_lbl = a_lbl.to(device)
                _mask = (mask > 0).float().to(device)

                axis_out, mask_out, total_loss, losses = net(img=img.to(device), lbl=axis_gs.float().to(device), mask=_mask, is_syn = is_syn, a_lbl=a_lbl, sym_type=sym_type)
                axis_out = F.interpolate(axis_out, size=axis.size()[2:], mode='bilinear', align_corners=True)
                img = F.interpolate(img, size=axis.size()[2:], mode='bilinear', align_corners=True)

                axis_eval(axis_out, axis)
                val_loss += total_loss.mean().item()
                val_sample += 1

                if not args.wandb_off:

                    _val_log_images = [
                        wandb.Image(unnorm(img), caption='Image/%s%d_image%d' % (mode, i, idx)),
                        wandb.Image(axis_gs,  caption='Image/%s%d_axisGT%d' % (mode, i, idx)),              
                        wandb.Image(axis_out.cpu(), caption='Image/%s%d_axisPred%d' % (mode, i, idx)),
                    ]

                    val_log_images += _val_log_images
            
            rec, prec, f1 = axis_eval.f1_score()
            recs.append(rec), precs.append(prec), f1s.append(f1), f1_maxs.append(f1.max())
            print(f1.max())

            if not args.wandb_off:
                log_dict = {
                   "%s_%d Loss" % (mode, i): val_loss / val_sample,
                   "%s_%d new max f1" % (mode, i): f1.max(),
                   "%s_%d new max_f1_index" % (mode, i): f1.argmax()
                }
                log_dict['%s_%d_log_images' % (mode, i)] = val_log_images

                wandb.log(log_dict)

    return recs, precs, f1s, f1_maxs

def vis(net, args, test_loaders, device, mode='test', sym_type='reflection'):
    net.eval()
    with torch.no_grad():
        for i, test_loader in enumerate(test_loaders):
            if (i == 0) and (mode in ['test']):
                print('skip nyu')
                continue

            val_log_images = []

            for idx, data in enumerate(tqdm(test_loader)):
                if args.get_theta:
                    img, mask, axis, axis_gs, is_syn, a_lbl = data
                    a_lbl.to(device)
                else:
                    img, mask, axis, axis_gs, is_syn, _ = data
                    a_lbl = None

                axis_out, _, _, _ = net(img=img.to(device), lbl=axis_gs.float().to(device), \
                                                        mask=(mask > 0).float().to(device), is_syn = is_syn, a_lbl=a_lbl, sym_type=sym_type)
                axis_out = F.interpolate(axis_out, size=axis.size()[2:], mode='bilinear', align_corners=True)
                img = F.interpolate(img, size=axis.size()[2:], mode='bilinear', align_corners=True)

                if args.save_qual:
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
                    plt.imsave("./vis_result/%s/%03d_pred.png" % (args.model_name, idx), np.clip(attended_img, 0, 1))

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // (args.num_epochs * 0.5))) * (0.1 ** (epoch // (args.num_epochs * 0.75)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    set_seed(1)
    args = get_parser()
    args.sync_bn = True
    comment = str(args.ver)
    print(comment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    if not args.wandb_off:
        print('wandb on')
        if args.test_only:
            wandb.init(project="symdet_val", entity="symdet", notes=comment, save_code=True)
        else:
            wandb.init(project="symdet", entity="symdet", notes=comment, save_code=True)
        wandb.config.update(args)
    else:
        print('wandb off')
        
    if args.rot_data:
        sym_type = 'rotation'
    else:
        sym_type = 'reflection'
    
    net = SymmetryDetectionNetwork(args)
    net = nn.DataParallel(net)
    net.to(device)

    from dendi_loader import NewSymmetryDatasets

    trainset = NewSymmetryDatasets(sym_type=sym_type, split='train', input_size=[args.input_size, args.input_size], get_theta=args.get_theta, with_ref_circle=True, t_resize=False)
    valset = NewSymmetryDatasets(sym_type=sym_type, split='val', input_size=[args.input_size, args.input_size], get_theta=args.get_theta, with_ref_circle=True, t_resize=False)
    testset = NewSymmetryDatasets(sym_type=sym_type, split='test', input_size=[args.input_size, args.input_size], get_theta=1, with_ref_circle=True, t_resize=False)
    train_loader = data.DataLoader(trainset, batch_size=args.bs_train, shuffle=True, num_workers=4, drop_last=True)
    val_loader = data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

    if args.test_only:
        print('load pretrained model')
        ckpt_path = './weights/v_' + args.ver + '_best_checkpoint.pt'
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint['state_dict'], strict=True)

        # net.module.export()
        # net.to(device)
        # print('export done')

        # print(checkpoint['args'])
        rec, prec, f1, f1_max = test(net, args, (test_loader,), device, mode='test', sym_type=sym_type)
        checkpoint['stats'] = rec, prec, f1, f1_max
        print(f1_max)
    else:
        train(net, args, train_loader, (val_loader, ), (test_loader, ), device)
