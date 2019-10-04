#!/bin/bash
tmux new-session -d -s 1
tmux new-session -d -s 2
tmux new-session -d -s 3
tmux new-session -d -s 4
tmux new-session -d -s 5
tmux new-session -d -s 6

tmux send-keys -t 1 "python main.py --exp m_RMS_1e-2_x100 --arch MobileNet --lr 0.01 --optimizer RMSProp --width_mult 1.0  --gpu 0" ENTER
tmux send-keys -t 2 "python main.py --exp m_RMS_1e-2_x75 --arch MobileNet --lr 0.01 --optimizer RMSProp --width_mult 0.75 --gpu 1" ENTER
tmux send-keys -t 3 "python main.py --exp m_RMS_1e-2_x50 --arch MobileNet --lr 0.01 --optimizer RMSProp --width_mult 0.5 --gpu 1" ENTER
tmux send-keys -t 4 "python main.py --exp m_RMS_4e-2 --arch MobileNet --lr 0.04 --optimizer RMSProp --gpu 0" ENTER
tmux send-keys -t 5 "python main.py --exp c_RMS_1e-2 --arch CNN --lr 0.01 --optimizer RMSProp --gpu 2" ENTER
tmux send-keys -t 6 "python main.py --exp c_RMS_4e-2 --arch CNN --lr 0.04 --optimizer RMSProp --gpu 3" ENTER

