#!/bin/bash
tmux new-session -d -s dog1
tmux new-session -d -s dog2
tmux new-session -d -s dog3
tmux new-session -d -s dog4


tmux send-keys -t dog1 "python main.py\
 --exp m_SGD_inception_x50\
 --dataset stanford-dogs\
  --arch MobileNet\
   --lr 0.045\
    --optimizer SGD\
     --weight_decay 4e-5\
      --width_mult 0.5\
       --epochs 100\
        --lr_scheduler step\
         --lr_step 2\
          --lr_decay 0.94\
           --save_dir exp-dogs\
            --gpu 0" ENTER
