task(){
   python finetune.py \
        --data_path /mnt/cfs/sihaozhe/data/dfd_datasets/dfd_indoor/dfd_dataset_indoor_N8 \
        --dataset DSLR \
        --recon_all \
        -N DSLR_finetune_one_shot_$1 \
        --use_cuda \
        --device $2 \
        -E 500 \
        --BS 32 \
        --save_best \
        --save_last \
        --sm_loss_beta 2.5 \
        --camera_far 10. \
        --verbose \
        --n_shot 1 \
        --aif_recon_loss_lambda 1e2 \
        --aif_blur_loss_lambda 1e1 \
        --blur_loss_lambda 1e1 \
        --sm_loss_lambda 10 \
        --n_shot_indices $1 \
        --manual_seed 0
}

j=0
(
for i in `seq 0 26`
do
   task $i $j
done
) &

j=2
(
for i in `seq 27 54`
do
   task $i $j
done
) &

j=4
(
for i in `seq 55 80`
do
   task $i $j
done
) 