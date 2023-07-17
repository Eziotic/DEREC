# python finetune.py \
#       --data_path /mnt/cfs/sihaozhe/data/dfd_datasets/dfd_indoor/dfd_dataset_indoor_N8 \
#       --dataset DSLR \
#       --recon_all \
#       -N DSLR_finetune_ten_shot_top \
#       --use_cuda \
#       --device 5 \
#       -E 500 \
#       --BS 32 \
#       --save_best \
#       --save_last \
#       --sm_loss_beta 2.5 \
#       --camera_far 10. \
#       --verbose \
#       --n_shot 5 \
#       --aif_recon_loss_lambda 1e2 \
#       --aif_blur_loss_lambda 1e1 \
#       --blur_loss_lambda 1e1 \
#       --sm_loss_lambda 10 \
#       --n_shot_indices 42 35 55 0 31 52 78 15 57 37 \
#       --manual_seed 0

# python finetune.py \
#       --data_path /mnt/cfs/sihaozhe/data/dfd_datasets/dfd_indoor/dfd_dataset_indoor_N8 \
#       --dataset DSLR \
#       --recon_all \
#       -N DSLR_finetune_full\
#       --use_cuda \
#       --device 6 \
#       -E 500 \
#       --BS 32 \
#       --save_best \
#       --save_last \
#       --sm_loss_beta 2.5 \
#       --camera_far 10. \
#       --verbose \
#       --aif_recon_loss_lambda 1e2 \
#       --aif_blur_loss_lambda 1e1 \
#       --blur_loss_lambda 1e1 \
#       --sm_loss_lambda 10 \
#       --manual_seed 0

task(){
      python finetune.py \
            --data_path /mnt/cfs/sihaozhe/data/dfd_datasets/dfd_indoor/dfd_dataset_indoor_N8 \
            --dataset DSLR \
            --recon_all \
            -N DSLR_finetune_five_shot_rand$1 \
            --use_cuda \
            --device $2 \
            -E 500 \
            --BS 32 \
            --save_best \
            --save_last \
            --sm_loss_beta 2.5 \
            --camera_far 10. \
            --verbose \
            --n_shot 5 \
            --aif_recon_loss_lambda 1e2 \
            --aif_blur_loss_lambda 1e1 \
            --blur_loss_lambda 1e1 \
            --sm_loss_lambda 10 \
            --manual_seed 0
}

j=6
(
for i in 0 2 4 6 8
do
   task $i $j
done
) &

j=7
(
for i in 1 3 5 7 9
do
   task $i $j
done
)
