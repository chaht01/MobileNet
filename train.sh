#!/bin/bash
tmux new-session -d -s 1
tmux new-session -d -s 2
tmux new-session -d -s 3
tmux new-session -d -s 4
tmux new-session -d -s 5
tmux new-session -d -s 6
tmux new-session -d -s 7

tmux send-keys -t 1 "python main.py --exp m_RMS_1e-2_x50 --arch MobileNet --lr 0.01 --optimizer RMSProp --weight_decay 4e-5 --width_mult 0.5 --epochs 150 --lr_scheduler exp --lr_decay 0.9695 --save_dir optim_check --gpu 0" ENTER
tmux send-keys -t 2 "python main.py --exp m_Adam_1e-2_x50 --arch MobileNet --lr 0.01 --optimizer Adam --weight_decay 4e-5 --width_mult 0.5 --epochs 150 --lr_scheduler exp --lr_decay 0.9695 --save_dir optim_check --gpu 1" ENTER
tmux send-keys -t 3 "python main.py --exp m_Adam_5e-2_x50 --arch MobileNet --lr 0.05 --optimizer Adam --weight_decay 4e-5 --width_mult 0.5 --epochs 100 --lr_scheduler exp --lr_decay 0.9397 --save_dir optim_check --gpu 3" ENTER
tmux send-keys -t 4 "python main.py --exp m_SGD_5e-2_x50 --arch MobileNet --lr 0.05 --optimizer SGD --weight_decay 4e-5 --width_mult 0.5 --epochs 100 --lr_scheduler exp --lr_decay 0.9397 --save_dir optim_check --gpu 3" ENTER
tmux send-keys -t 5 "python main.py --exp m_Adam_1e-3_x50_nodecay --arch MobileNet --lr 0.001 --optimizer Adam --weight_decay 4e-5 --width_mult 0.5 --epochs 100 --save_dir optim_check --gpu 2" ENTER
tmux send-keys -t 6 "python main.py --exp m_RMS_1e-3_x50_nodecay --arch MobileNet --lr 0.001 --optimizer RMSProp --weight_decay 4e-5 --width_mult 0.5 --epochs 100 --save_dir optim_check --gpu 2" ENTER
tmux send-keys -t 7 "python main.py --exp m_RMS_1e-3_x50 --arch MobileNet --lr 0.001 --optimizer RMSProp --weight_decay 4e-5 --width_mult 0.5 --epochs 100 --lr_scheduler exp --lr_decay 0.9695 --save_dir optim_check --gpu 2" ENTER
tmux send-keys -t 8 "python main.py --exp m_SGD_1e-1_x50_100epoch --arch MobileNet --lr 0.1 --optimizer SGD --weight_decay 4e-5 --width_mult 0.5 --epochs 100 --lr_scheduler step --lr_step 30 --lr_decay 0.1 --save_dir optim_check --gpu 0" ENTER
tmux send-keys -t 9 "python main.py --exp m_SGD_inception_x50 --arch MobileNet --lr 0.045 --optimizer SGD --weight_decay 4e-5 --width_mult 0.5 --epochs 100 --lr_scheduler step --lr_step 2 --lr_decay 0.94 --save_dir optim_check --gpu 0" ENTER
tmux send-keys -t 10 "python main.py --exp m_RMS_inception_x50 --arch MobileNet --lr 0.045 --optimizer RMSProp --weight_decay 4e-5 --width_mult 0.5 --epochs 100 --lr_scheduler step --lr_step 2 --lr_decay 0.94 --save_dir optim_check --gpu 1" ENTER
tmux send-keys -t 11 "python main.py\
 --exp m_RMS_inception_x50_plateau\
  --arch MobileNet\
   --lr 0.045\
    --optimizer RMSProp\
     --weight_decay 4e-5\
      --width_mult 0.5\
       --epochs 100\
       --lr_threshold 0.05\
        --lr_scheduler plat\
         --lr_plat_factor 0.1\
          --save_dir optim_check\
           --gpu 1" ENTER
tmux send-keys -t 12 "python main.py\
 --exp m_SGD_inception_x50_plateau\
  --arch MobileNet\
   --lr 0.045\
    --optimizer SGD\
     --weight_decay 4e-5\
      --width_mult 0.5\
       --epochs 100\
       --lr_threshold 0.05\
        --lr_scheduler plat\
         --lr_plat_factor 0.1\
          --save_dir optim_check\
           --gpu 2" ENTER

tmux send-keys -t 13 "python main.py\
 --exp m_SGD_inception_x50_plateau_th-1e-1_pf-2e-1\
  --arch MobileNet\
   --lr 0.045\
    --optimizer SGD\
     --weight_decay 4e-5\
      --width_mult 0.5\
       --epochs 100\
       --lr_threshold 0.1\
        --lr_scheduler plat\
         --lr_plat_factor 0.2\
          --save_dir optim_check\
           --gpu 1" ENTER

tmux send-keys -t 14 "python main.py\
 --exp m_RMS_inception_x50_plateau_th-1e-1_pf-2e-1\
  --arch MobileNet\
   --lr 0.045\
    --optimizer RMSProp\
     --weight_decay 4e-5\
      --width_mult 0.5\
       --epochs 300\
       --lr_threshold 0.1\
        --lr_scheduler plat\
         --lr_plat_factor 0.2\
          --save_dir optim_check\
           --gpu 2" ENTER

tmux send-keys -t 15 "python main.py\
 --exp m_SGD_inception_x50_plateau_th-1e-1_pf-2e-1_pc-5\
  --arch MobileNet\
   --lr 0.045\
    --optimizer SGD\
     --weight_decay 4e-5\
      --width_mult 0.5\
       --epochs 100\
       --lr_patience 5\
       --lr_threshold 0.1\
        --lr_scheduler plat\
         --lr_plat_factor 0.2\
          --save_dir optim_check\
           --gpu 0" ENTER

tmux send-keys -t 16 "python main.py\
 --exp m_RMS_inception_x50_plateau_th-1e-1_pf-2e-1_pc-5\
  --arch MobileNet\
   --lr 0.045\
    --optimizer RMSProp\
     --weight_decay 4e-5\
      --width_mult 0.5\
       --epochs 100\
       --lr_patience 5\
       --lr_threshold 0.1\
        --lr_scheduler plat\
         --lr_plat_factor 0.2\
          --save_dir optim_check\
           --gpu 0" ENTER

tmux send-keys -t 17 "python main.py\
 --exp c_RMS_inception_x50_plateau_th-1e-1_pf-2e-1\
  --arch CNN\
   --lr 0.045\
    --optimizer RMSProp\
     --weight_decay 4e-5\
      --width_mult 0.5\
       --epochs 300\
       --lr_threshold 0.1\
        --lr_scheduler plat\
         --lr_plat_factor 0.2\
          --save_dir optim_check\
           --gpu 3" ENTER


## minimal mobile

tmux send-keys -t 18 "python main.py\
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
           --save_dir mobile_cmp\
            --gpu 0" ENTER

tmux send-keys -t 19 "python main.py\
 --exp m_RMS_inception_x50\
  --arch MobileNet\
   --lr 0.045\
    --optimizer RMSProp\
     --weight_decay 4e-5\
      --width_mult 0.5\
       --epochs 100\
        --lr_scheduler step\
         --lr_step 2\
          --lr_decay 0.94\
           --save_dir mobile_cmp\
            --gpu 0" ENTER

tmux send-keys -t 20 "python main.py\
 --exp m2_RMS_inception_x50_plateau_th-1e-1_pf-2e-1_pc-5\
  --arch MobileNet2\
   --lr 0.045\
    --optimizer RMSProp\
     --weight_decay 4e-5\
      --width_mult 0.5\
       --epochs 100\
       --lr_patience 5\
       --lr_threshold 0.1\
        --lr_scheduler plat\
         --lr_plat_factor 0.2\
          --save_dir mobile_cmp\
           --gpu 1" ENTER

tmux send-keys -t 21 "python main.py\
 --exp m2_RMS_inception_x50_step_30_1e-1\
  --arch MobileNet2\
   --lr 0.045\
    --optimizer RMSProp\
     --weight_decay 4e-5\
      --width_mult 0.5\
       --epochs 100\
        --lr_scheduler step\
        --lr_step 30\
         --lr_decay 0.1\
          --save_dir mobile_cmp\
           --gpu 0" ENTER

tmux send-keys -t 22 "python main.py\
 --exp m2_RMS_inception_x50_plateau_th-1e-1_pf-2e-1\
  --arch MobileNet2\
   --lr 0.045\
    --optimizer RMSProp\
     --weight_decay 4e-5\
      --width_mult 0.5\
       --epochs 100\
       --lr_threshold 0.1\
        --lr_scheduler plat\
         --lr_plat_factor 0.2\
          --save_dir mobile_cmp\
           --gpu 1" ENTER


## Stanford

tmux send-keys -t 23 "python main.py\
 --exp m2_RMS_inception_x50\
  --arch MobileNet2\
   --dataset stanford-dogs\
    --lr 0.045\
    --optimizer RMSProp\
     --weight_decay 4e-5\
      --width_mult 0.5\
       --epochs 100\
        --lr_scheduler step\
         --lr_step 2\
          --lr_decay 0.94\
           --save_dir stanford\
            --gpu 2" ENTER
