import argparse

parser = argparse.ArgumentParser(description='MobileNet Pytorch')
parser.add_argument('--lr', type=float, default=0.2)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--wd', '--weight_decay', type=float, default=0.9)
parser.add_argument('--eps', type=float, default=1.0)

parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=256)

parser.add_argument('--resume', default=None, help='checkpoint to resume')
parser.add_argument('--log_step', type=int, default=50,
                    help='step for logging in iteration')
parser.add_argument('--save_epoch', type=int, default=10,
                    help='step for saving in epoch')
parser.add_argument('--data_dir', default='./')
parser.add_argument('--save_dir', default='./',
                    help='dave directory for checkpoint')

parser.add_argument('--workers', type=int, default=4)
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


def get_option(override):
    opt = parser.parse_args()
    return opt
