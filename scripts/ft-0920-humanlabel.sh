ROOT=/mnt/z4/shumowu/aigc_ym/code/piccolo-embedding
export PYTHONPATH=$ROOT:${PYTHONPATH}

# SLURM Parameter
GPUS_PER_NODE=8
MASTER_PORT=6000
if [ -z "$WORLD_SIZE" ]; then  
    WORLD_SIZE=1  
    RANK=0
    MASTER_ADDR=127.0.0.1
    MASTER_PORT=6000
fi

# Hyper Parameter Start
MODEL_NAME_OR_PATH=/mnt/z4/shumowu/huggingface/hub/models--lier007--xiaobu-embedding-v2/snapshots/ee0b4ecdf5eb449e8240f2e3de2e10eeae877691
EPOCHS=10
BATCH_SIZE=8
LR=1e-5
NEG_NUM=1
DS_PATH=$ROOT/ds_config_zero1.json
MAX_LENGTH=64
META_PATHS=(
meta_lists/piccolo-ft-fix.txt
)
MAX_SAMPLE_PATH=meta_lists/maxnum.txt
OUTPUT_DIR=/mnt/z4/shumowu/aigc_ym/train_log/xiaobu/exp-0920-humanlabel
ROOT_DIRS=(
/mnt/z4/shumowu/aigc_ym/datasets/
)
# Hyper Parameter End 


model_args=(
    "--model_name_or_path" $MODEL_NAME_OR_PATH
    "--max_length=$MAX_LENGTH"
    "--query_prefix=''"
    "--doc_prefix=''"
    "--use_scaling_layer=True"
    "--use_mrl=True"
)

data_args=(
    "--meta_paths" "${META_PATHS[@]}"
    "--root_dirs" "${ROOT_DIRS[@]}"
    "--neg_num=$NEG_NUM"
    "--max_sample_path=$MAX_SAMPLE_PATH"
    "--pos_key=text_pos_name"
    "--neg_key=text_neg_name"
)

train_args=(
    "--fp16"
    "--gradient_checkpointing=False"
    "--output_dir=$OUTPUT_DIR"
    "--num_train_epochs=$EPOCHS"
    "--dataloader_num_workers=0"
    "--batch_size=$BATCH_SIZE"
    "--learning_rate=$LR"
    "--deepspeed=$DS_PATH"
    "--logging_steps=500"
    "--save_safetensors=False"
    "--report_to=tensorboard"
    "--save_strategy=no"
    "--per_device_train_batch_size=1"
)

all_args=("${model_args[@]}" "${data_args[@]}" "${train_args[@]}")


export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $WORLD_SIZE \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

export CMD=" \
    $ROOT/finetune/train.py \
    "

echo $CMD

bash -c "$LAUNCHER $CMD ${all_args[*]}"
