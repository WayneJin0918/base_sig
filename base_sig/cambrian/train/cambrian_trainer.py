import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler

import dataclasses
import json
from typing import Dict, List, Optional, Union
import numpy as np
import gcsfs
from google.cloud import storage
import io
import torch_xla.core.xla_model as xm
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
    is_torch_tpu_available
)

from ezcolorlog import root_logger as logger
from cambrian.utils import IS_XLA_AVAILABLE

from packaging import version
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
from typing import List, Optional

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import is_apex_available
if is_apex_available():
    from apex import amp

import random
fs = gcsfs.GCSFileSystem(project='nyu-vision-lab')

HOME_DIR = os.path.expanduser("~") + "/"
print("HOME_DIR = ", HOME_DIR)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


# def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
#     # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
#     assert all(l != 0 for l in lengths), "Should not have zero length."
#     if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
#         # all samples are in the same modality
#         return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
#     mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
#     lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

#     mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
#     lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
#     megabatch_size = world_size * batch_size
#     mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
#     lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

#     last_mm = mm_megabatches[-1]
#     last_lang = lang_megabatches[-1]
#     additional_batch = last_mm + last_lang
#     megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
#     megabatch_indices = torch.randperm(len(megabatches), generator=generator)
#     megabatches = [megabatches[i] for i in megabatch_indices]

#     if len(additional_batch) > 0:
#         megabatches.append(sorted(additional_batch))

#     return [i for megabatch in megabatches for i in megabatch]

def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    assert all(l != 0 for l in lengths), "Should not have zero length."
    
    # If all samples belong to the same modality, directly return length-grouped indices
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    
    # Pre-define modality indices to avoid dynamic separation at runtime
    mm_indices = [i for i, l in enumerate(lengths) if l > 0]
    lang_indices = [i for i, l in enumerate(lengths) if l < 0]
    
    # Get grouped indices for each modality (handle sorting in advance)
    mm_shuffle = get_length_grouped_indices([lengths[i] for i in mm_indices], batch_size, world_size, generator)
    lang_shuffle = get_length_grouped_indices([-lengths[i] for i in lang_indices], batch_size, world_size, generator)

    megabatch_size = world_size * batch_size
    
    # Split the shuffled modality indices into megabatches
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    # Merge the last megabatch of each modality if both exist
    if len(mm_megabatches) > 0 and len(lang_megabatches) > 0:
        additional_batch = mm_megabatches.pop() + lang_megabatches.pop()
    else:
        additional_batch = []

    # Concatenate the remaining megabatches and preserve order or use pre-defined random order
    megabatches = mm_megabatches + lang_megabatches

    # Perform shuffling in the data loading stage; assume a static random shuffle order here
    if generator is not None:
        megabatch_indices = torch.randperm(len(megabatches), generator=generator)
        megabatches = [megabatches[i] for i in megabatch_indices]

    # Append the additional batch if it exists
    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    # Return a flattened list of indices
    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


def _fetch_gradients(optimizer, param_to_name, selected_module_names):
    gradients = []
    for param_group in optimizer.param_groups:
        for group, params in param_group.items():
            if group == 'params':
                for p in params:
                    # Use the mapping to get the module name
                    module_name = param_to_name.get(p, "")
                    # Check if the module name matches your criteria
                    if isinstance(p, torch.Tensor) and p.grad is not None and any(selected_name in module_name for selected_name in selected_module_names):
                        p.grad = p.grad.to(torch.float32)
                        gradients.append(p.grad.data)
    return gradients

from torch_xla.core.xla_model import xrt_world_size, all_reduce
REDUCE_SUM = 'sum'
def reduce_gradients(optimizer, param_to_name, selected_module_names, groups=None, pin_layout=True):
    count = xrt_world_size()
    if count > 1:
        gradients = _fetch_gradients(optimizer, param_to_name, selected_module_names)
        all_reduce(
            REDUCE_SUM,
            gradients,
            scale=1.0 / count,
            groups=groups,
            pin_layout=pin_layout)

def map_params_to_module_names(model_list):
    param_to_name = {}
    for model in model_list:
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                param_to_name[param] = f"{module_name}.{param_name}"
    return param_to_name


class CambrianTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        batch_size = inputs['labels'].size(0)
        
        indices = torch.arange(batch_size).to(self.args.device)
        indices_to_modify = indices[indices % 2 == 1]

        inputs['attention_mask'][indices_to_modify, :575] = 0
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            if self.args.dpo:
                outputs = model(**inputs)
                logits = outputs.logits
                labels = inputs['labels']
                # print(labels.size(),"label")
                # print(logits.size(),"logits")
                beta = self.args.beta
                log_prob = self.get_batch_logps(logits, labels, return_per_token_logp=False)
                loss = self.compute_loss_dpo(log_prob, beta)
            else:
                loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        selected_module_names = ['vision_tower']
        # if self.args.unfreeze_mm_vision_tower:
        #     reduce_gradients(self.optimizer, self.param_to_name, selected_module_names)

        return loss.detach() / self.args.gradient_accumulation_steps

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        opt_model = self.model
        # if self.args.unfreeze_mm_vision_tower:
        #     opt_model.get_model().vision_tower_aux_list = nn.ModuleList(opt_model.get_vision_tower_aux_list())
        #     self.param_to_name = map_params_to_module_names([opt_model])
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            assert not (self.args.mm_projector_lr and self.args.mm_vision_sampler_lr)
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            elif self.args.mm_vision_sampler_lr is not None:
                vision_sampler_parameters = [name for name, _ in opt_model.named_parameters() if ("vision_sampler" in name) or ("vision_query" in name) ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in vision_sampler_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in vision_sampler_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in vision_sampler_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_vision_sampler_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in vision_sampler_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_vision_sampler_lr,
                    },
                ]
            elif self.args.unfreeze_mm_vision_tower and self.args.mm_vision_tower_lr is not None:
                vision_tower_parameters = [name for name, _ in opt_model.named_parameters() if "vision_tower" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in vision_tower_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in vision_tower_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in vision_tower_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_vision_tower_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in vision_tower_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_vision_tower_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")
        return self.optimizer
    

    def remove_prefix(text, prefix='gs://us-central2-storage/'):
        if prefix in text:
            return text.replace(prefix, '')
        return text
    
    def _load_rng_state(self, resume_from_checkpoint):
        if resume_from_checkpoint is None:
            return

        # remove local path prefic if exists
        if HOME_DIR in resume_from_checkpoint:
            resume_from_checkpoint_clean = resume_from_checkpoint.replace(HOME_DIR, '')
        else:
            resume_from_checkpoint_clean = resume_from_checkpoint

        # get worker details
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()

        # get path
        RNG_NAME = f'rng_rank-{rank:08d}-of-{world_size:08d}-rng.pth'
        RNG_PATH = os.path.join(resume_from_checkpoint_clean, RNG_NAME)

        # Loading the model weights:
        client = storage.Client()
        bucket = client.get_bucket('weiyang2')
        blob = bucket.blob(RNG_PATH)
        blob_bytes = blob.download_as_bytes()
        buffer = io.BytesIO(blob_bytes)
        rng_dict = torch.load(buffer)

        # Setting the seeds correctly
        random.setstate(rng_dict["python"])
        np.random.set_state(rng_dict["numpy"])
        torch.random.set_rng_state(rng_dict["cpu"])
        xm.set_rng_state(rng_dict["xla"])
        print("rng state loaded")

    def _load_optimizer_and_scheduler(self, resume_from_checkpoint):
        if resume_from_checkpoint is None:
            return

        # remove local path prefix
        if HOME_DIR in resume_from_checkpoint:
            resume_from_checkpoint_clean = resume_from_checkpoint.replace(HOME_DIR, '')
        else:
            resume_from_checkpoint_clean = resume_from_checkpoint

        # get worker details
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()

        # get path to file
        WEIGHTS_NAME = "pytorch_model.bin"
        SCHEDULER_NAME = "scheduler.pt"
        SHARD_NAME_OPT = f'opt_rank-{rank:08d}-of-{world_size:08d}-{WEIGHTS_NAME}'
        SHARD_NAME_PATH = os.path.join(resume_from_checkpoint_clean, SHARD_NAME_OPT)
        LR_PATH = os.path.join(resume_from_checkpoint_clean, SCHEDULER_NAME)

        # connect to gcloud bucket
        client = storage.Client()
        bucket = client.get_bucket('weiyang2')

        # Loading opt state to each device
        blob = bucket.blob(SHARD_NAME_PATH)
        blob_bytes = blob.download_as_bytes()
        buffer = io.BytesIO(blob_bytes)
        optimizer_state = torch.load(buffer, map_location="cpu")
        optimizer_state = optimizer_state['optimizer_state']

        # Loading the schedule to each device
        blob_lr = bucket.blob(LR_PATH)
        blob_bytes_lr = blob_lr.download_as_bytes()
        buffer_lr = io.BytesIO(blob_bytes_lr)
        lr_scheduler_state = torch.load(buffer_lr)

        # No need for this, since already inside XLA spawn?
        # xm.send_cpu_data_to_device(optimizer_state, self.args.device)
        # xm.send_cpu_data_to_device(lr_scheduler_state, self.args.device)

        # Load state
        self.optimizer.load_state_dict(optimizer_state)
        self.lr_scheduler.load_state_dict(lr_scheduler_state)

        logger.info(f"Optimizer state and scheduler successfully loaded from {SHARD_NAME_PATH}")
        print("Loaded optimizer state successfully")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):

        if resume_from_checkpoint is None:
            return

        # Remove local path (we stored Train State here)
        if HOME_DIR in resume_from_checkpoint:
            resume_from_checkpoint_clean = resume_from_checkpoint.replace(HOME_DIR, '')
        else:
            resume_from_checkpoint_clean = resume_from_checkpoint

        # Getting worker details
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()

        # Getting path to file on bucket
        WEIGHTS_NAME = "pytorch_model.bin"
        SHARD_NAME = f'weights_rank-{rank:08d}-of-{world_size:08d}-{WEIGHTS_NAME}'
        SHARD_NAME_PATH = os.path.join(resume_from_checkpoint_clean, SHARD_NAME)


        # Loading the model weights:
        client = storage.Client()
        bucket = client.get_bucket('weiyang2')
        blob = bucket.blob(SHARD_NAME_PATH)
        blob_bytes = blob.download_as_bytes()
        buffer = io.BytesIO(blob_bytes)
        state_dict = torch.load(buffer)
        state_dict = state_dict["model"]

        # self.model = self._wrap_model(self.model, )

        # Saving to each worker  - NO NEED TO MOVE ANYTHING TO XLA
        self.model.load_state_dict(state_dict)

    def _save_checkpoint(self, model, trial, metrics=None):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        # Names of files
        TRAINING_ARGS_NAME = "training_args.bin"
        WEIGHTS_NAME = "pytorch_model.bin"
        SCHEDULER_NAME = "scheduler.pt"
        TRAINER_STATE_NAME = "trainer_state.json"

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        logger.info(f"Saving model checkpoint to {output_dir}")

        model = self.model
        import torch_xla.core.xla_model as xm
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()

        # Name of files to save
        SHARD_NAME = f'weights_rank-{rank:08d}-of-{world_size:08d}-{WEIGHTS_NAME}'
        SHARD_NAME_OPT = f'opt_rank-{rank:08d}-of-{world_size:08d}-{WEIGHTS_NAME}'
        RNG_NAME = f'rng_rank-{rank:08d}-of-{world_size:08d}-rng.pth'

        # Path of files to save
        SHARD_NAME_PATH = os.path.join(output_dir, SHARD_NAME)
        SHARD_NAME_OPT_PATH = os.path.join(output_dir, SHARD_NAME_OPT)
        LR_PATH = os.path.join(output_dir, SCHEDULER_NAME)
        TRAIN_ARGS_PATH = os.path.join(output_dir, TRAINING_ARGS_NAME)
        TRAINER_STATE_NAME_PATH = os.path.join(output_dir, TRAINER_STATE_NAME)
        RNG_PATH = os.path.join(output_dir, RNG_NAME)
        lr_scheduler_state_dict = self.lr_scheduler.state_dict()

        # Final form of model and opt
        print("Saving checkpoint...")
        ckpt = {
            'model': self.model.state_dict(),
            'shard_metadata': self.model.get_shard_metadata()
        }
        opt_ckpt = {
            'optimizer_state' : self.optimizer.state_dict(),
            'shard_metadata': self.model.get_shard_metadata()
        }

        # Saving model shards
        with fs.open(SHARD_NAME_PATH, 'wb') as f:
            xm.save(ckpt, f, master_only=False)

        # Saving optimizer shards
        with fs.open(SHARD_NAME_OPT_PATH, 'wb') as f:
            xm.save(opt_ckpt, f, master_only=False)

        # saving lr scheduler and train state json
        if xm.is_master_ordinal(local=False):
            with fs.open(LR_PATH, 'wb') as f:
                xm.save(lr_scheduler_state_dict, f, master_only=True)

            json_string = json.dumps(dataclasses.asdict(self.state), indent=2, sort_keys=True) + "\n"
            with fs.open(TRAINER_STATE_NAME_PATH, 'w') as f:
                f.write(json_string)

        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        rng_states["xla"] = xm.get_rng_state()
        with fs.open(RNG_PATH, 'wb') as f:
            torch.save(rng_states, f)

    # def _save_checkpoint(self, model, trial, metrics=None):
    #     if getattr(self.args, 'tune_mm_mlp_adapter', False):
    #         from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
    #         checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

    #         run_dir = self._get_output_dir(trial=trial)
    #         output_dir = os.path.join(run_dir, checkpoint_folder)

    #         # Only save Adapter
    #         keys_to_match = ['mm_projector', 'vision_resampler']
    #         if getattr(self.args, "use_im_start_end", False):
    #             keys_to_match.extend(['embed_tokens', 'embed_in'])

    #         weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

    #         if self.args.local_rank == 0 or self.args.local_rank == -1:
    #             self.model.config.save_pretrained(output_dir)
    #             torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
    #     else:
    #         super(CambrianTrainer, self)._save_checkpoint(model, trial, metrics)

    def get_train_dataloader(self) -> DataLoader:
        out = super().get_train_dataloader()
        return out._loader

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        import torch_xla.core.xla_model as xm
        ckpt_prefix = os.path.join(output_dir, "model_ckpt")
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        rank = xm.get_ordinal()
        print(rank)
        world_size = xm.xrt_world_size()
        ckpt_path = f'{ckpt_prefix}_rank-{rank:08d}-of-{world_size:08d}.pth'
        state_dict = self.model.state_dict()
        cpu_state_dict = {
                key: value.cpu()
                for key, value in state_dict.items()
            }
        # if not xm.is_master_ordinal(local=False):
        #     cpu_state_dict = {
        #         key:value for key, value in cpu_state_dict.items() if 'vision_tower' not in key
        #     }
        del state_dict
        ckpt = {
            'model': cpu_state_dict,
            'shard_metadata': self.model.get_shard_metadata()
        }
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        xm.save(ckpt, ckpt_path, master_only=False)
        print(f'checkpoint saved to {ckpt_path}\n', end='')
        if xm.is_master_ordinal(local=False):
            # consolidate_sharded_model_checkpoints(
            #     ckpt_prefix=ckpt_prefix, ckpt_suffix="_rank-*-of-*.pth", save_path = os.path.join(output_dir, "model_consolidated.pth"))
            # self.model.save_pretrained(output_dir, state_dict=None, safe_serialization=self.args.save_safetensors)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            TRAINING_ARGS_NAME = "training_args.bin"
            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
            self.model.config.save_pretrained(output_dir)

    """Override to add custom logs"""

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_tpu_available():
                import torch_xla.core.xla_model as xm
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            # Add custom logs
            if self.args.unfreeze_mm_vision_tower:
                logs["mm_vision_tower_lr"] = self.optimizer.param_groups[2]['lr']

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def get_batch_logps(self, logits: torch.FloatTensor, labels: torch.LongTensor, return_per_token_logp=False, return_all=False) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert logits.shape[:-1] == labels.shape
        """
        # Predefine tensors with fixed shapes
        batch_size = labels.size(0)
        sequence_length = labels.size(1)
        num_classes = logits.size(2)

        # Create zero tensors and copy the required parts into them
        labels_static = torch.zeros((batch_size, sequence_length - 1)).to(labels.device)
        logits_static = torch.zeros((batch_size, sequence_length - 1, num_classes)).to(logits.device)

        # Copy the sliced results into the new static tensors
        labels_static = labels[:, 1:].clone()
        logits_static = logits[:, :-1, :].clone()

        # If you want to avoid direct slicing, you can build static tensors using concatenation
        # Assume sequence_length=5, build an index tensor like [1:sequence_length]
        index_labels = torch.arange(1, sequence_length).to(labels.device)
        index_logits = torch.arange(0, sequence_length - 1).to(logits.device)

        # Use index selection to create static tensors (gradients will be computed)
        labels_static = torch.index_select(labels, 1, index_labels).clone()
        logits_static = torch.index_select(logits, 1, index_logits).clone()
        """
        
        # Finally, pass the static tensors to the subsequent parts of the model
        # labels = labels_static
        # logits = logits_static
        labels = labels[:, 1:]
        logits = logits[:, :-1]
        loss_mask1 = (labels != -100)
        loss_mask2 = (labels != -200)
        loss_mask = loss_mask1 * loss_mask2
        # labels[labels == -100] = 0
        # labels[labels == -200] = 0

        labels = torch.where((labels == -100) | (labels == -200), torch.tensor(0, device=labels.device), labels)
        
        logits = logits + 1e-9

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        log_prob = (per_token_logps * loss_mask).sum(-1)
        return log_prob
    
    def get_vision_p(self, logits: torch.FloatTensor, labels: torch.LongTensor = None) -> torch.FloatTensor:
        """Compute the probabilities of the given vision features for labels that are equal to -200.
    
        Args:
            logits: Vision features extracted by Vision Tower. Shape: (batch_size, num_features, vocab_size).
            labels: Labels for which to compute the probabilities, similar to text labels. Shape: (batch_size, num_features), optional.
    
        Returns:
            A tensor of shape (batch_size,) containing the average probabilities of the features where labels are equal to -200.
        """
        # Step 1: Normalize vision features to probabilities (using softmax)
        vision_probs = torch.softmax(logits, dim=-1)  # Probabilities over the vocabulary size (last dimension)
    
        # Step 2: Create a mask for labels that are equal to -200
        mask = (labels == -200)  # Shape: (batch_size, num_features)
        
        # Step 3: Clamp labels to a valid index (e.g., 0) to prevent indexing errors
        clamped_labels = torch.clamp(labels, min=0)  # Replace -200 with 0 for safe indexing
    
        # Step 4: Gather the probabilities based on clamped labels
        gathered_probs = torch.gather(vision_probs, dim=-1, index=clamped_labels.unsqueeze(-1)).squeeze(-1)
    
        # Step 5: Apply the mask to extract only the probabilities corresponding to -200 labels
        selected_probs = gathered_probs * mask  # Masking unwanted values (they will be zeroed)
    
        # Step 6: Sum over the masked positions and calculate the average for each batch
        sum_probs = torch.sum(selected_probs, dim=-1)  # Sum of probabilities for each instance in batch
        count_valid = torch.sum(mask, dim=-1)  # Count the number of valid (masked as -200) positions per instance in batch
    
        # Avoid division by zero by adding epsilon where count is zero
        average_probs = sum_probs / (count_valid + 1e-8)  # Avoid division by zero
    
        return average_probs
    
    def simpo_loss(self, policy_chosen_logps: torch.FloatTensor,
                   policy_rejected_logps: torch.FloatTensor,
                   beta: float,
                   gamma: float) -> torch.FloatTensor:
        normalized_chosen_logps = policy_chosen_logps
        normalized_rejected_logps = policy_rejected_logps
        logits = beta * (normalized_chosen_logps - normalized_rejected_logps) - gamma
        losses = -F.logsigmoid(logits)
        
        return losses

    def loss_co(self, policy_chosen_logps: torch.FloatTensor,
                   policy_rejected_logps: torch.FloatTensor,
                   beta: float,
                   gamma: float) -> torch.FloatTensor:
        diff =  beta *policy_chosen_logps - beta *policy_rejected_logps
        losses = -F.logsigmoid(diff)
        
        return losses
    
    def compute_loss_dpo(self, log_prob, beta, return_outputs=False):

        log_prob = log_prob
    
        group_size = 2
        batch_size = log_prob.size(0)
        num_groups = batch_size // group_size
    
        log_prob = log_prob.view(num_groups, group_size)
        best_log_prob = log_prob[:, 0]
        worst_log_prob = log_prob[:, 1]
        # print(best_log_prob)
        # print(worst_log_prob)
        losses = self.loss_co(
                best_log_prob,
                worst_log_prob,
                beta=beta,
                gamma=0
            )
    
        total_simpo_loss = losses.mean()
        total_loss = total_simpo_loss
        
        return total_loss
