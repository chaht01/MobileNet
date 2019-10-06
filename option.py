import argparse

parser = argparse.ArgumentParser(description='MobileNet Pytorch')
parser.add_argument('--exp', type=str, required=True)
parser.add_argument('--arch', type=str,
                    choices=['MobileNet', 'MobileNet2', 'CNN'], default='MobileNet')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay', type=float)
parser.add_argument('--lr_patience', type=int, default=10)
parser.add_argument('--lr_step', type=int, default=None)
parser.add_argument('--lr_plat_factor', type=float, default=0.1)
parser.add_argument('--lr_threshold', type=float, default=0.05)
parser.add_argument(
    '--lr_scheduler', choices=['exp', 'step', 'plat'], default=None)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument(
    '--optimizer', choices=['SGD', 'RMSProp', 'Adam'], default='RMSProp')

parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--start_epoch', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--width_mult', type=float, default=1.0)

parser.add_argument('--resume', default=None, help='checkpoint to resume')
parser.add_argument('--log_step', type=int, default=50,
                    help='step for logging in iteration')
parser.add_argument('--save_epoch', type=int, default=10,
                    help='step for saving in epoch')
parser.add_argument('--dataset', choices=['tiny-imagenet', 'stanford-dogs'], default='tiny-imagenet')
parser.add_argument('--data_dir', default='../../data/tiny-imagenet-200')
parser.add_argument('--save_dir', default='./ckpoints',
                    help='dave directory for checkpoint')

parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


def get_option():
    opt = parser.parse_args()
    return opt
