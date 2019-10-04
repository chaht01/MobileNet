import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed


class Trainer(object):
    def __init__(
            self,
            datasets,
            model,
            criterion,
            optimizer,
            summarizer,
            option):
        self.datasets = datasets
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.summarizer = summarizer
        self.option = option

        if self.option.gpu is not None:
            warnings.warn('You have chosen a specific GPU. This will completely '
                          'disable data parallelism.')

        if self.option.dist_url == "env://" and self.option.world_size == -1:
            self.option.world_size = int(os.environ["WORLD_SIZE"])

        self.option.distributed = self.option.world_size > 1 or self.option.multiprocessing_distributed

        ngpus_per_node = torch.cuda.device_count()

        if self.option.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            self.option.world_size = ngpus_per_node * self.option.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            mp.spawn(self._main_worker, nprocs=ngpus_per_node, self.option=(ngpus_per_node, self.option))
        else:
            # Simply call main_worker function
            self._main_worker(self.option.gpu, ngpus_per_node, self.option)

    def _main_worker(self, gpu, ngpus_per_node, args):
        global best_acc1
        args.gpu = gpu

        if args.gpu is not None:
            print("Use GPU: {} for training".format(args.gpu))

        if args.distributed:
            if args.dist_url == "env://" and args.rank == -1:
                args.rank = int(os.environ["RANK"])
            if args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                args.rank = args.rank * ngpus_per_node + args.gpu
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
        # create model
        model = self.model()
        if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(
                    args.batch_size / ngpus_per_node)
                args.workers = int(
                    (args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(
                    model)
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            model = torch.nn.DataParallel(model).cuda()

        # define loss function (criterion) and optimizer
        criterion = self.criterion.cuda(args.gpu)
        optimizer = self.optimizer(model.parameters(), args.lr,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                if args.gpu is None:
                    checkpoint = torch.load(args.resume)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(args.gpu)
                    checkpoint = torch.load(args.resume, map_location=loc)
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                if args.gpu is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    best_acc1 = best_acc1.to(args.gpu)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        cudnn.benchmark = True

        # Data loading code
        train_dataset, val_datasets = self.datasets
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_datasets, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    def _mode_settings(self, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

    def _train_step(self, epoch, data_loader):
        self._mode_settings(True)
        for step, (images, labels) in enumerate(data_loader):
            self.optimizer.zero_grad()
            pred = self.model(images)
            loss = self.criterion(pred, labels)
            loss.backward()
            self.optimizer.step()

    def _validate(self, data_loader):
        self._mode_settings(False)

    def train(self):
        train_loader, val_loader = self.loaders

        for epoch in range(self.option.epochs):
            self._train_step(epoch, train_loader)
            self._validate(val_loader)

    def main(self):
