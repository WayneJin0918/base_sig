
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
# export XLA_USE_BF16=0 &&
# export WANDB_RESUME="allow" &&
export CKPT_NAME="cambrian-8b-finetune-llm-base-posttrain-737k-unlock-vit"
# export XLA_FLAGS="--xla_hlo_profile --xla_gpu_force_compilation_parallelism=1"

export CKPT_DIR="~/ckpt/$CKPT_NAME"

# debug flags
if [ "$LLAVA_DEBUG" = "1" ]; then
    echo  "Use debug setups."
    export TPU_PROCESS_BOUNDS=1,1,1
    export TPU_VISIBLE_CHIPS=0
    num_workers=0
    export WANDB_MODE=disabled
fi

exp_name=cambrian_post_training_737k_unlock_vit

export WANDB_API_KEY="2bfd61b1549a21d11093d9fd3f83063b390034e2"
export WANDB_ENTITY=nyu-visionx
export WANDB_DISABLE_CODE="true"
export WANDB_IGNORE_GLOBS="*.patch"
export WANDB_PROJECT=Cambrian-DPO
export WANDB_NAME=$exp_name

# export WANDB_MODE="disabled"

python cambrian/train/train_tpu.py \
    --model_name_or_path ~/cambrian-8b-finetune-llm-base/checkpoint-last/hf \
    --version llama_v3 \
    --data_path /mnt/disks/storage/data/finetune_data/jsons/737k.jsonl \
    --image_folder /mnt/disks/storage/data/finetune_data/ \
    --pretrain_mm_mlp_adapter ~/mm_projector.pth \
    --vision_tower_aux_list '["siglip/CLIP-ViT-SO400M-14-384"]' \
    --vision_tower_aux_token_len_list '[576]' \
    --image_token_len 576 \
    --num_query_group 1 \
    --query_num_list '[576]' \
    --connector_depth 3 \
    --image_position 91 \
    --vision_hidden_size 1024 \
    --connector_only False \
    --num_of_vision_sampler_layers 10 \
    --start_of_vision_sampler_layers 0 \
    --stride_of_vision_sampler_layers 3 \
    --mm_projector_type sva \
    --unfreeze_mm_vision_tower True \
    --mm_vision_tower_lr 4e-5 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --output_dir gs://shusheng/checkpoints/ImpLangSup/$CKPT_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 4e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers ${num_workers:-4} \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $CKPT_NAME \
    --fsdp "full_shard" \
    --fsdp_config fsdp_config.json

CKPT_PATH=checkpoints/$CKPT_NAME
# check if the checkpoint path exists
if [ ! -d "$CKPT_PATH" ]; then
    echo "Checkpoint path does not exist. Exiting..."
    exit 1
fi
echo "Training finished. Syncing checkpoints to GCS..."
gcloud alpha storage rsync $CKPT_PATH  gs://shusheng/checkpoints/ImpLangSup/$CKPT_NAME/checkpoint-last
echo "Syncing finished. Checkpoints are now available at gs://shusheng/checkpoints/ImpLangSup/$CKPT_NAME/checkpoint-last"
