#!/bin/bash
tmux new-session -d -s 1
tmux new-session -d -s 2
tmux new-session -d -s 3
tmux new-session -d -s 4
tmux new-session -d -s 5
tmux new-session -d -s 6
tmux new-session -d -s 7
tmux new-session -d -s 8

tmux send-keys -t 1 "python ../main.py\
 --exp m2_RMS_inception_x25\
  --arch MobileNet2\
   --lr 0.045\
    --optimizer RMSProp\
     --weight_decay 4e-5\
      --width_mult 0.25\
       --epochs 100\
        --lr_scheduler step\
         --lr_step 2\
          --lr_decay 0.94\
           --save_dir results\
            --gpu 0" ENTER

tmux send-keys -t 2 "python ../main.py\
 --exp m2_RMS_inception_x50\
  --arch MobileNet2\
   --lr 0.045\
    --optimizer RMSProp\
     --weight_decay 4e-5\
      --width_mult 0.5\
       --epochs 100\
        --lr_scheduler step\
         --lr_step 2\
          --lr_decay 0.94\
           --save_dir results\
            --gpu 1" ENTER

tmux send-keys -t 3 "python ../main.py\
 --exp m2_RMS_inception_x75\
  --arch MobileNet2\
   --lr 0.045\
    --optimizer RMSProp\
     --weight_decay 4e-5\
      --width_mult 0.75\
       --epochs 100\
        --lr_scheduler step\
         --lr_step 2\
          --lr_decay 0.94\
           --save_dir results\
            --gpu 2" ENTER

tmux send-keys -t 4 "python ../main.py\
 --exp m2_RMS_inception_x100\
  --arch MobileNet2\
   --lr 0.045\
    --optimizer RMSProp\
     --weight_decay 4e-5\
      --width_mult 1.0\
       --epochs 100\
        --lr_scheduler step\
         --lr_step 2\
          --lr_decay 0.94\
           --save_dir results\
            --gpu 3" ENTER


tmux send-keys -t 5 "python ../main.py\
 --exp m2_RMS_inception_x25_shallow\
  --arch MobileNet2\
   --shallow\
   --lr 0.045\
    --optimizer RMSProp\
     --weight_decay 4e-5\
      --width_mult 0.25\
       --epochs 100\
        --lr_scheduler step\
         --lr_step 2\
          --lr_decay 0.94\
           --save_dir results\
            --gpu 3" ENTER

tmux send-keys -t 6 "python ../main.py\
 --exp m2_RMS_inception_x50_shallow\
  --arch MobileNet2\
   --shallow\
   --lr 0.045\
    --optimizer RMSProp\
     --weight_decay 4e-5\
      --width_mult 0.5\
       --epochs 100\
        --lr_scheduler step\
         --lr_step 2\
          --lr_decay 0.94\
           --save_dir results\
            --gpu 2" ENTER

tmux send-keys -t 7 "python ../main.py\
 --exp m2_RMS_inception_x75_shallow\
  --arch MobileNet2\
   --shallow\
   --lr 0.045\
    --optimizer RMSProp\
     --weight_decay 4e-5\
      --width_mult 0.75\
       --epochs 100\
        --lr_scheduler step\
         --lr_step 2\
          --lr_decay 0.94\
           --save_dir results\
            --gpu 1" ENTER

tmux send-keys -t 8 "python ../main.py\
 --exp m2_RMS_inception_x100_shallow\
  --arch MobileNet2\
   --shallow\
   --lr 0.045\
    --optimizer RMSProp\
     --weight_decay 4e-5\
      --width_mult 1.0\
       --epochs 100\
        --lr_scheduler step\
         --lr_step 2\
          --lr_decay 0.94\
           --save_dir results\
            --gpu 0" ENTER
