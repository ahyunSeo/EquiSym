import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='EquiSym arguments')
    parser.add_argument('--input_size', default=417, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--bs_train', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--bs_val', '-bs', default=1, type=int,
                        help='Batch size for validation')
    parser.add_argument('--ver', default='init', type=str)
    parser.add_argument('--lr',         type=float, default=0.001,      metavar='LR',
                        help="base learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0,        metavar='DECAY',
                        help="weight decay, if 0 nothing happens")
    parser.add_argument('-t', '--test_only', action='store_true')
    parser.add_argument('-wf', '--wandb_off', action='store_true', default=False)
    parser.add_argument('--sync_bn', action='store_true', default=False)

    # e2cnn backbone variants
    parser.add_argument('-eq', '--eq_cnn', action='store_true', default=False)
    parser.add_argument('-bb', '--backbone', default='resnet', type=str)
    parser.add_argument('-res', '--depth', default=50, type=int)

    parser.add_argument('-gt', '--get_theta', default=0, type=int)
    parser.add_argument('--n_angle', default=8, type=int)
    parser.add_argument('-rot', '--rot_data', default=0, type=int)
    parser.add_argument('-load_eq', '--load_eq_pretrained', default=1, type=int)
    parser.add_argument('--eq_model_dir', default='./weights/re_resnet50_custom_d8_batch_512.pth', type=str)

    parser.add_argument('--theta_loss_type', default=1, type=int)
    parser.add_argument('--n_rot', default=21, type=int)
    parser.add_argument('-theta_ls', '--theta_loss_scale', default=1, type=float)
    parser.add_argument('-tlw', '--theta_loss_weight', default=1e-3, type=float) # 1e-2 ref, 1e-3 rot
    args = parser.parse_args()

    return args
