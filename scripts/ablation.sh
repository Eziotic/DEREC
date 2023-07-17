read -p "Exp ID:" ID
read -p "Device:" DID
#######################################################
# Standarad Hyperparameter                            #
#######################################################
if [[ $ID -eq 0 ]]
then
python train.py \
        --data_path /mnt/cfs/sihaozhe/data/NYUv2 \
        --dataset NYU100 \
        -N NYU_100_FS5 \
        --use_cuda \
        --device $DID \
        -E 2000 \
        --BS 32 \
        --save_checkpoint \
        --save_best \
        --save_last \
        --sm_loss_beta 2.5 \
        --camera_far 10. \
        --verbose \
        --aif_recon_loss_lambda 1e2 \
        --aif_blur_loss_lambda 1e1 \
        --blur_loss_lambda 1e1 \
        --sm_loss_lambda 10 \
        --log \
        --vis \
        --manual_seed 0
elif [[ $ID -eq 1 ]]
then
python train.py \
        --data_path /mnt/cfs/sihaozhe/data/NYUv2 \
        --dataset NYU100 \
        -N NYU_100_FS5_no_caif_blur \
        --use_cuda \
        --device $DID \
        -E 2000 \
        --BS 32 \
        --save_checkpoint \
        --save_best \
        --save_last \
        --sm_loss_beta 2.5 \
        --camera_far 10. \
        --verbose \
        --aif_recon_loss_lambda 1e2 \
        --aif_blur_loss_lambda 1e1 \
        --blur_loss_lambda 0 \
        --sm_loss_lambda 10 \
        --log \
        --vis \
        --manual_seed 0
elif [[ $ID -eq 2 ]]
then
python train.py \
        --data_path /mnt/cfs/sihaozhe/data/NYUv2 \
        --dataset NYU100 \
        -N NYU_100_FS5_no_caif \
        --use_cuda \
        --device $DID \
        -E 2000 \
        --BS 32 \
        --save_checkpoint \
        --save_best \
        --save_last \
        --sm_loss_beta 2.5 \
        --camera_far 10. \
        --verbose \
        --aif_recon_loss_lambda 0 \
        --aif_blur_loss_lambda 1e1 \
        --blur_loss_lambda 0 \
        --sm_loss_lambda 10 \
        --log \
        --vis \
        --manual_seed 0
elif [[ $ID -eq 3 ]]
then
python train.py \
        --data_path /mnt/cfs/sihaozhe/data/NYUv2 \
        --dataset NYU100 \
        -N NYU_100_FS5_no_sm \
        --use_cuda \
        --device $DID \
        -E 2000 \
        --BS 32 \
        --save_checkpoint \
        --save_best \
        --save_last \
        --sm_loss_beta 2.5 \
        --camera_far 10. \
        --verbose \
        --aif_recon_loss_lambda 1e2 \
        --aif_blur_loss_lambda 1e1 \
        --blur_loss_lambda 10 \
        --sm_loss_lambda 0 \
        --log \
        --vis \
        --manual_seed 0
elif [[ $ID -eq 4 ]]
then
python train.py \
        --data_path /mnt/cfs/sihaozhe/data/NYUv2 \
        --dataset NYU100 \
        -N NYU_100_FS5_no_sharp \
        --use_cuda \
        --device $DID \
        -E 2000 \
        --BS 32 \
        --save_checkpoint \
        --save_best \
        --save_last \
        --sm_loss_beta 2.5 \
        --camera_far 10. \
        --verbose \
        --aif_recon_loss_lambda 1e2 \
        --aif_blur_loss_lambda 1e1 \
        --blur_loss_lambda 10 \
        --sm_loss_lambda 10 \
        --sharp_loss_lambda 0 \
        --log \
        --vis \
        --manual_seed 0
elif [[ $ID -eq 5 ]]
then
python train.py \
        --data_path /mnt/cfs/sihaozhe/data/NYUv2 \
        --dataset NYU100 \
        -N NYU_100_FS5_no_recon \
        --use_cuda \
        --device $DID \
        -E 2000 \
        --BS 32 \
        --save_checkpoint \
        --save_best \
        --save_last \
        --sm_loss_beta 2.5 \
        --camera_far 10. \
        --verbose \
        --aif_recon_loss_lambda 1e2 \
        --aif_blur_loss_lambda 1e1 \
        --blur_loss_lambda 10 \
        --sm_loss_lambda 10 \
        --recon_loss_lambda 0 \
        --log \
        --vis \
        --manual_seed 0
elif [[ $ID -eq 6 ]]
then
python train.py \
        --data_path /mnt/cfs/sihaozhe/data/NYUv2 \
        --dataset NYU100 \
        -N NYU_100_FS5_only_recon \
        --use_cuda \
        --device $DID \
        -E 2000 \
        --BS 32 \
        --save_checkpoint \
        --save_best \
        --save_last \
        --sm_loss_beta 2.5 \
        --camera_far 10. \
        --verbose \
        --aif_recon_loss_lambda 0 \
        --aif_blur_loss_lambda 0 \
        --blur_loss_lambda 0 \
        --sm_loss_lambda 0 \
        --sharp_loss_lambda 0 \
        --log \
        --vis \
        --manual_seed 0
elif [[ $ID -eq 7 ]]
then
python train.py \
        --data_path /mnt/cfs/sihaozhe/data/NYUv2 \
        --dataset NYU100 \
        -N NYU_100_FS5_no_aif_recon \
        --use_cuda \
        --device $DID \
        -E 2000 \
        --BS 32 \
        --save_checkpoint \
        --save_best \
        --save_last \
        --sm_loss_beta 2.5 \
        --camera_far 10. \
        --verbose \
        --aif_recon_loss_lambda 0 \
        --aif_blur_loss_lambda 1e1 \
        --blur_loss_lambda 1e1 \
        --sm_loss_lambda 10 \
        --log \
        --vis \
        --manual_seed 0
elif [[ $ID -eq 8 ]]
then
python train.py \
        --data_path /mnt/cfs/sihaozhe/data/NYUv2 \
        --dataset NYU100 \
        -N NYU_100_FS5_CAM \
        --use_cuda \
        --device $DID \
        -E 2000 \
        --BS 32 \
        --save_checkpoint \
        --save_best \
        --save_last \
        --sm_loss_beta 2.5 \
        --camera_far 10. \
        --verbose \
        --aif_recon_loss_lambda 1e2 \
        --aif_blur_loss_lambda 1e1 \
        --blur_loss_lambda 1e1 \
        --sm_loss_lambda 10 \
        --log \
        --vis \
        --manual_seed 0
elif [[ $ID -eq 9 ]]
then
python train.py \
        --data_path /mnt/cfs/sihaozhe/data/NYUv2 \
        --dataset NYU100 \
        -N NYU_100_FS5_D5 \
        --use_cuda \
        --device $DID \
        -E 2000 \
        --BS 32 \
        --save_checkpoint \
        --save_best \
        --save_last \
        --sm_loss_beta 2.5 \
        --camera_far 10. \
        --verbose \
        --aif_recon_loss_lambda 1e2 \
        --aif_blur_loss_lambda 1e1 \
        --blur_loss_lambda 1e1 \
        --sm_loss_lambda 10 \
        --log \
        --vis \
        --manual_seed 0 \
        -D 5
fi