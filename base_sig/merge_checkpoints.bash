python /home/wayneyjin/base_sig/base_sig/merge_checkpoints.py \
  --ckpt_prefix "/home/wayneyjin/ckpt/cambrian-8b-pretrain-all/checkpoint-3926/weights_rank" \
  --output_dir "/home/wayneyjin/ckpt/merged_model" \
  --config_name "/home/wayneyjin/ckpt/cambrian-8b-pretrain-all/config.json" \
  --tokenizer_name "/home/wayneyjin/ckpt/cambrian-8b-pretrain-all/tokenizer" \
  --num_shards 64