#! /bin/bash
python train.py \
        --data_path /mnt/petrelfs/zhaobin/gyp/DeRec-main/realDataset/ \
        --dataset SC \
        -N realDataset \
        -E 800 \
        --BS 1 \
        --use_cuda \
        --device 0 \
        --recon_all \
        --visible_image_num 6 \
        --image_num 6 \
        --save_checkpoint \
        --save_best \
        --save_last \
        --sm_loss_beta 2.5 \
        --verbose \
        --aif_recon_loss_lambda 100 \
        --aif_blur_loss_lambda 1e1 \
        --blur_loss_lambda 1e1 \
        --sm_loss_lambda 10 \
        --manual_seed 0 \
        --log \
        --vis 

