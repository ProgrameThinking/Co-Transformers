base_dir="./eval_robust_dir"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
./test_robust.py \
    --model CoTransformers \
    --world_size 1 \
    --test_data_path "your path" \
    --checkpoint_path "your path" \
    --test_batch_size 2 \
    --edge_mask_width 7 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.log