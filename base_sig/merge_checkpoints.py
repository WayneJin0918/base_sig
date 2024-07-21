import os
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer
import argparse
from google.cloud import storage
import io

def download_shard_from_gcs(bucket_name, shard_name, local_path):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(shard_name)
    blob_bytes = blob.download_as_bytes()
    with open(local_path, 'wb') as f:
        f.write(blob_bytes)
    print(f"Downloaded {shard_name} to {local_path}")

def load_shard(local_path):
    shard = torch.load(local_path, map_location="cpu")  # Load shard on CPU
    return shard

def average_shards(local_state_dict, world_size):
    for key in local_state_dict.keys():
        local_state_dict[key] = local_state_dict[key] / world_size
    return local_state_dict

def save_merged_model(output_dir, state_dict, config, tokenizer):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, model_path)
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f'Model saved to {output_dir}')

def tpu_main(index, ckpt_prefix, output_dir, config_name, tokenizer_name, num_shards):
    rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()
    
    # Define GCS bucket and shard path
    bucket_name = "us-central2-storage"
    shard_name = f'{ckpt_prefix}-{rank:08d}-of-{num_shards:08d}-pytorch_model.bin'
    local_path = f"/tmp/weights_rank-{rank:08d}-of-{num_shards:08d}-pytorch_model.bin"

    # Download shard from GCS to local path
    download_shard_from_gcs(bucket_name, shard_name, local_path)

    print(f"Loading model shard on rank {rank}...")
    local_shard = load_shard(local_path)

    # Convert local shard to XLA tensor
    local_shard = {k: v.to(xm.xla_device()) for k, v in local_shard.items()}

    print(f"Averaging model shards on rank {rank}...")
    local_shard = average_shards(local_shard, world_size)

    global_state_dict = {}
    for key in local_shard.keys():
        global_state_dict[key] = xm.all_reduce('sum', [local_shard[key]], scale=1.0)

    xm.rendezvous('average_shards')

    if rank == 0:
        print(f"Loading config and tokenizer from {config_name} and {tokenizer_name}...")
        config = AutoConfig.from_pretrained(config_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Convert global state dict back to CPU for saving
        global_state_dict = {k: v.cpu() for k, v in global_state_dict.items()}

        print(f"Saving merged model to {output_dir}...")
        save_merged_model(output_dir, global_state_dict, config, tokenizer)
        print("Model merging complete.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_prefix', type=str, required=True, help='Path prefix to the checkpoint shards')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save the merged model')
    parser.add_argument('--config_name', type=str, required=True, help='Pretrained model config path')
    parser.add_argument('--tokenizer_name', type=str, required=True, help='Pretrained tokenizer path')
    parser.add_argument('--num_shards', type=int, required=True, help='Number of shards')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    xmp.spawn(tpu_main, args=(args.ckpt_prefix, args.output_dir, args.config_name, args.tokenizer_name, args.num_shards), nprocs=8)
