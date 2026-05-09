base_dir="./logs/train"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=1 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=1 \
./train.py \
    --model CoTransformers \
    --world_size 1 \
    --batch_size 8 \
    --data_path ./balanced_dataset.json \
    --epochs 200 \
    --lr 1e-4 \
    --image_size 512 \
    --if_resizing \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --edge_mask_width 7 \
    --test_data_path ./test_datasets.json \
    --test_batch_size 20\
    --warmup_epochs 2 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 8 \
    --seed 42 \
    --test_period 4 \
2> ${base_dir}/error.log 1>${base_dir}/logs.log