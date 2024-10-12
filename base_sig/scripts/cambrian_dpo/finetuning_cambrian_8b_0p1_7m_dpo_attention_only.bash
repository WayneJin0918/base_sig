
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
# export XLA_USE_BF16=0 &&
# export WANDB_RESUME="allow" &&
export CKPT_NAME="cambrian-8b-finetune-llm-dpo-posttrain-0p1-7m-dpo-1e-6-l-on-v-on-new-proj-hp-25"
# export XLA_FLAGS="--xla_hlo_profile --xla_gpu_force_compilation_parallelism=1"

export CKPT_DIR="$HOME/ckpt/$CKPT_NAME"

# debug flags
if [ "$LLAVA_DEBUG" = "1" ]; then
    echo  "Use debug setups."
    export TPU_PROCESS_BOUNDS=1,1,1
    export TPU_VISIBLE_CHIPS=0
    num_workers=0
    export WANDB_MODE=disabled
fi

exp_name=cambrian_post_training_737k_ils_test

export WANDB_API_KEY="43161fa0733b631782a2d30b3bd8e9b5e8e0d482"
export WANDB_ENTITY=nyu-visionx
export WANDB_DISABLE_CODE="true"
export WANDB_IGNORE_GLOBS="*.patch"
export WANDB_PROJECT=Cambrian-DPO
export WANDB_NAME=$exp_name

# export WANDB_MODE="disabled"

# Default values

resume=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
        resume="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
  esac
done

TRAIN_ARGS="
    --model_name_or_path /home/wayneyjin/ckpt/cambrian-sig-8b \
    --version llama_v3 \
    --data_path /home/wayneyjin/ckpt/Cambrian7M_withsystemprompt.jsonl \
    --image_folder /home/wayneyjin/weiyangrl-bucket/data/finetune_data \
    --pretrain_mm_mlp_adapter /home/wayneyjin/projector/mm_projector.pth \
    --vision_tower_aux_list [\"siglip/CLIP-ViT-SO400M-14-384\"] \
    --vision_tower_aux_token_len_list [576] \
    --image_token_len 576 \
    --num_query_group 1 \
    --query_num_list [576] \
    --connector_depth 3 \
    --image_position 91 \
    --vision_hidden_size 1024 \
    --connector_only False \
    --num_of_vision_sampler_layers 10 \
    --start_of_vision_sampler_layers 0 \
    --stride_of_vision_sampler_layers 3 \
    --mm_projector_type sva \
    --unfreeze_mm_vision_tower True \
    --tune_llm_self_attention_only True \
    --freeze_backbone False \
    --mm_vision_tower_lr 4e-5 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --output_dir gs://weiyang2/$CKPT_NAME \
    --num_train_epochs 0.1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers ${num_workers:-4} \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $CKPT_NAME \
    --fsdp full_shard \
    --fsdp_config fsdp_config.json \
    --dpo True \
    --beta 1 \
    --noise_level [25,0] \
    --noise_type high_pass_filter \
"

if [ -n "$resume" ]; then
    TRAIN_ARGS="$TRAIN_ARGS \
        --train_continue True \
        --resume_from_checkpoint $resume \
    "
fi

echo $TRAIN_ARGS

python cambrian/train/train_tpu.py \
    $TRAIN_ARGS

CKPT_PATH=checkpoints/$CKPT_NAME
# check if the checkpoint path exists
if [ ! -d "$CKPT_PATH" ]; then
    echo "Checkpoint path does not exist. Exiting..."
    exit 1
fi
echo "Training finished. Syncing checkpoints to GCS..."
gcloud alpha storage rsync $CKPT_PATH  gs://weiyang2/$CKPT_NAME/checkpoint-last
echo "Syncing finished. Checkpoints are now available at gs://weiyang2/$CKPT_NAME/checkpoint-last"
