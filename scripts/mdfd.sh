read -p "TarID:" TID
read -p "Device:" ID

python finetune.py \
    --data_path /mnt/cfs/sihaozhe/data/mobileDFD/ \
    --dataset mobileDFD \
    -N mobileDFD_FS5_$TID \
    --use_cuda \
    --device $ID \
    -E 200 \
    --BS 1 \
    --save_best \
    --save_last \
    --sm_loss_beta 2.5 \
    --camera_far 5 \
    --verbose \
    --n_shot 1 \
    --aif_recon_loss_lambda 1e2 \
    --aif_blur_loss_lambda 1e1 \
    --blur_loss_lambda 1e1 \
    --sm_loss_lambda 10 \
    --n_shot_indices $TID \
    --manual_seed 0