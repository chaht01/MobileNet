#!/bin/bash
tmux new-session -d -s cyclic

tmux send-keys -t cyclic "python find_cyclic_bound.py --gpu 0 --save_dir hyperparam --exp cyclic" ENTER
