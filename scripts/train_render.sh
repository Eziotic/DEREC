python train.py \
        --data_path /mnt/cfs/sihaozhe/data/fs_7_transfer \
        --dataset render \
        -N Render_transfer_unsup \
        --use_cuda \
        --device 7 \
        -E 2000 \
        --BS 32 \
        --save_checkpoint \
        --save_best \
        --sm_loss_beta 2.5 \
        --camera_far 10. \
        --continue_from best-model.pth \
        --verbose \
        --aif_recon_loss_lambda 1e1 \
        --aif_blur_loss_lambda 1e1

# python train.py \
#         --data_path /mnt/cfs/sihaozhe/data/fs_7_transfer \
#         --dataset render \
#         -N Render_transfer_unsup \
#         --use_cuda \
#         --device 6 \
#         --BS 32 \
#         --save_best \
#         --camera_far 10. \
#         --verbose \
#         --continue_from best-model.pth \
#         --eval \
#         --scale 1