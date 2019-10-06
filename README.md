# MobileNet

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
