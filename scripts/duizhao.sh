#! /bin/bash
# python finetune.py \
#         --data_path /mnt/cfs/sihaozhe/data/dfd_datasets/dfd_indoor/dfd_dataset_indoor_N8 \
#         --dataset DSLR \
#         --recon_all \
#         -N DSLR_finetune_one_shot_0 \
#         --use_cuda \
#         --device 5 \
#         -E 500 \
#         --BS 32 \
#         --save_best \
#         --save_last \
#         --sm_loss_beta 2.5 \
#         --camera_far 10. \
#         --verbose \
#         --n_shot 1 \
#         --aif_recon_loss_lambda 1e2 \
#         --aif_blur_loss_lambda 1e1 \
#         --blur_loss_lambda 1e1 \
#         --sm_loss_lambda 10 \
#         --n_shot_indices 0 

python finetune.py \
        --data_path /mnt/petrelfs/zhaobin/gyp/data/ddff-dataset-trainval.h5 \
        --dataset DDFF \
        -N duizhao \
        --use_cuda \
        --scale 4 \
        --device 0 \
        -E 10 \
        --BS 32 \
        --recon_all \
        --save_checkpoint \
        --save_best \
        --save_last \
        --sm_loss_beta 2.5 \
        --camera_far 10. \
        --verbose \
        --aif_recon_loss_lambda 1e3 \
        --aif_blur_loss_lambda 1e1 \
        --blur_loss_lambda 1e1 \
        --sm_loss_lambda 10 \
        --manual_seed 0 \
        --log \
        --vis
