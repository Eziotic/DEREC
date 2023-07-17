#! /bin/bash
# python train.py \
#         --data_path /mnt/cfs/sihaozhe/data/NYUv2 \
#         --dataset NYUv2 \
#         -N NYU_orig_unsup \
#         --use_cuda \
#         --device 7 \
#         --BS 32 \
#         --save_best \
#         --continue_from best-model.pth \
#         --camera_far 10. \
#         --verbose 

# python train.py \
#         --data_path /mnt/cfs/sihaozhe/data/NYUv2 \
#         --dataset NYUv2 \
#         -N NYU_norm_unsup \
#         --normalize_dpt \
#         --use_cuda \
#         --device 5 \
#         --BS 32 \
#         --save_best \
#         --continue_from best-model.pth \
#         --verbose 

 python train.py \
         --data_path /mnt/petrelfs/zhaobin/gyp/data/ddff-dataset-trainval.h5 \
         --dataset DDFF \
         -N new \
         --use_cuda \
         --scale 1 \
         --recon_all \
         --device 0 \
         --continue_from best-model.pth \
         --BS 32 \
         --save_best \
         --camera_far 8. \
         --verbose \
         --eval
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
#python train.py \
#        --data_path /mnt/petrelfs/zhaobin/gyp/data/ddff-dataset-trainval.h5 \
#        --dataset DDFF \
#        -N NYU10 \
#        --use_cuda \
#        --device 0 \
#        --BS 64 \
#        --camera_far 10. \
#        --verbose \
#        --eval

# task(){ 
# python train.py \
#         --data_path /mnt/cfs/sihaozhe/data/dfd_datasets/dfd_indoor/dfd_dataset_indoor_N8 \
#         --dataset DSLR \
#         -N DSLR_finetune_one_shot_$1 \
#         --recon_all \
#         --use_cuda \
#         --device 3 \
#         --BS 1 \
#         --camera_far 10. \
#         --verbose \
#         --continue_from best-model.pth \
#         --eval
# }

# for i in 68
# do
#     task $i
# done


