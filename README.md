# MobileNet-64 on Tiny ImageNet

This model is implemented and modified based on the original MobileNet to handle images 64x64 size (The original model requires images with size at least 128). To do this, the first layer conducted without down-sampling, which indicated its stride size is 1(the original model's choice is 2). 

## Results
![Imgur](https://imgur.com/6O5iMHn.png)
![Imgur](https://imgur.com/gRPgEgh.png)

## Reproduce experiments (training mobilenet only)
```
./scripts/run.sh
```
All pth files will be placed `results/[EXPERIMENT_NAME]` and tensorboard event files will be places `results/[EXPERIMENT_NAME]/runs`.

## Run tensorboard
```
tensorboard --logdir results/[EXPERIMENT_NAME]/runs
```

## Resume
Resume experiment with a pth file does not require explicit start epoch. In other word, it is unable to specify start epoch.
Thus, if you want resume some experiment and execute from right after epoch, run like this.

For example,
```
python main.py --resume results/[EXPERIMENT_NAME]/4.pth
```
will perform training from epoch number 5(0-based index) with an assumption training of epoch number 4 completed without any error.

## Get Flops and # of params
```
python flop.py
```
