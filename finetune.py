import copy
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed

import random
import json
from tqdm import tqdm
import math
import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig)

from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from arguments import get_args

from data_utils.lm_datasets import LMTrainDataset
from data_utils.data_utils import LLMDataset
from utils import get_optimizer_params, get_optimizer_params_peft, print_args, initialize
from utils import print_rank, get_rank
from utils import save_rank
from utils import all_gather
from utils import load_parallel, save_parallel
from utils import get_tokenizer, get_model

from distillm import forward_kl, reverse_kl, js_distance, tv_distance
from distillm import skewed_forward_kl, skewed_reverse_kl
from distillm import SampleGenerator, ReplayBuffer

from rouge_metric import compute_metrics

from peft import PeftModel
from ed_eval import ed_evaluate

torch.set_num_threads(4)


def get_teacher_model(args, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    if args.model_parallel:
        raise NotImplementedError
    else:
        config.is_model_parallel = False
        try: model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.bfloat16)
        except:
            model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.float32)
            model = model.half()
        
        if args.teacher_peft_path is not None:
            model = PeftModel.from_pretrained(model, args.teacher_peft_path)
            model = model.merge_and_unload()
            print("merge_and_unload")

        if dist.get_rank() == 0:
            print(' > number of parameters: {}'.format(
                sum([p.nelement() for p in model.parameters()])), flush=True)

    model.eval()
    
    return model


def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)

    # Use AdamW.
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5)
    elif args.lr_decay_style == "wrmup_cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_ratio * args.total_iters,
            num_training_steps=args.total_iters)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler


def setup_model_and_optimizer(args, ds_config, device, set_optim=True):
    # get the model
    model = get_model(args, device)
    # get the optimizer and lr_scheduler
    if set_optim:
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer, lr_scheduler = None, None
        
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler


def prepare_dataset(args, tokenizer):
    data = {}
    rng_sample = random.Random(args.seed)
    if args.do_train:
        data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["train"]))
        data["dev"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    elif not args.do_eval:
        raise ValueError("Do train and do eval must set one")
    if args.do_eval:
        data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "test", args.dev_num, args.dev_ratio, rng_sample)
        
    # pre-trained dataset
    if args.do_train and args.lm_data_dir is not None:
        data["pt_train"] = LMTrainDataset(args, tokenizer, args.lm_data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["pt_train"]))
    return data


def pt_loss(args, model, model_batch, no_model_batch):
    loss_mask = (no_model_batch["label"] != -100).int()
    outputs = model(**model_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), no_model_batch["label"].view(-1))
    return lm_loss


def compute_loss_distill(old_model_module, new_logits, model_batch, no_model_batch,
                         temperature: float = 1.0, device=None):
    """KL distillation from the frozen old model to the current model.

    Computes τ² · CE(p_old, p_new) on response token positions, which is
    gradient-equivalent to τ² · KL(p_old || p_new) since H(p_old) is constant
    w.r.t. the current model parameters (Hinton et al. 2015).

    Uses cross-entropy formulation to avoid materializing a full [B,T,V]
    output tensor from F.kl_div, which would OOM on large vocabularies.
    """
    old_model_module.eval()
    old_model_module.to(device)
    with torch.no_grad():
        old_outputs = old_model_module(**model_batch, use_cache=False)
        old_logits = old_outputs.logits.float().detach()   # [B, T, V]
    old_model_module.to("cpu")  # move back to CPU to free GPU memory
    torch.cuda.empty_cache()

    # Only compute loss on response token positions (label != -100)
    resp_mask = (no_model_batch["label"] != -100)  # [B, T]

    tau = temperature
    # Compute soft targets from old model, then free old_logits immediately
    old_probs = F.softmax(old_logits / tau, dim=-1)
    del old_logits

    new_log_probs = F.log_softmax(new_logits.float() / tau, dim=-1)

    # CE(p_old, p_new) per token = -Σ_v p_old · log(p_new)
    ce_per_token = -(old_probs * new_log_probs).sum(dim=-1)  # [B, T]
    del old_probs, new_log_probs

    # Mean over response tokens, scaled by τ² to match gradient magnitudes
    n_tokens = resp_mask.sum().clamp(min=1)
    loss_distill = (tau ** 2) * (ce_per_token * resp_mask.float()).sum() / n_tokens

    return loss_distill


def compute_loss_transfer(new_logits, no_model_batch, cached_logits, temperature=1.0):
    """KL distillation from cached old-model top-k probs on a replay batch.

    Instead of running the old model forward, uses pre-computed top-k soft
    probabilities stored in cached_logits.

    Args:
        new_logits: [B, T, V] from current model on replay batch.
        no_model_batch: dict with "label" [B, T] (-100 for non-response).
        cached_logits: dict with "top_k_indices" [B, T, K] and "top_k_probs" [B, T, K].
        temperature: softmax temperature τ.

    Returns:
        Scalar loss: τ² · mean_response_tokens( -Σ_k p_old_k · log p_new_k ).
    """
    tau = temperature
    resp_mask = (no_model_batch["label"] != -100)  # [B, T]

    new_log_probs = F.log_softmax(new_logits.float() / tau, dim=-1)  # [B, T, V]

    top_k_indices = cached_logits["top_k_indices"].to(device=new_logits.device, dtype=torch.long)  # [B, T, K]
    top_k_probs = cached_logits["top_k_probs"].to(device=new_logits.device, dtype=torch.float32)   # [B, T, K]

    # Gather new model log-probs at the cached top-k positions
    new_log_probs_at_topk = torch.gather(new_log_probs, dim=-1, index=top_k_indices)  # [B, T, K]
    del new_log_probs  # free [B, T, V] tensor

    # CE(p_old_topk, p_new) per token = -Σ_k p_old_k · log p_new_k
    ce_per_token = -(top_k_probs * new_log_probs_at_topk).sum(dim=-1)  # [B, T]

    n_tokens = resp_mask.sum().clamp(min=1)
    loss_transfer = (tau ** 2) * (ce_per_token * resp_mask.float()).sum() / n_tokens

    return loss_transfer


def count_unique_responses_buffer(replay_buffer):
    """Count unique response token sequences in replay buffer items."""
    patterns = set()
    for item in replay_buffer.replay_memory:
        label = item["label"]
        resp_tokens = label[label != -100]
        patterns.add(tuple(resp_tokens.tolist()))
    return len(patterns)


def count_unique_responses_dataset(dataset, model_type="qwen"):
    """Count unique response token sequences in a training dataset.

    Uses the sentinel marker (4294967295 for qwen, 65535 otherwise) to find
    where the prompt ends and the response begins.
    """
    patterns = set()
    sentinel = 4294967295 if model_type in ["qwen"] else 65535
    for idx in range(len(dataset)):
        raw = dataset[idx]
        input_ids = raw["input_ids"]
        if isinstance(input_ids, np.ndarray):
            input_ids = input_ids.astype(np.int64)
        else:
            input_ids = np.array(input_ids, dtype=np.int64)
        markers = np.where(input_ids == sentinel)[0]
        if len(markers) > 0:
            source_len = markers[0]
            # Response tokens start after source_len (sentinel is removed in _process_lm)
            resp_tokens = np.concatenate([input_ids[:source_len], input_ids[source_len + 1:]])
            resp_tokens = resp_tokens[source_len:]
        else:
            resp_tokens = input_ids[1:]  # fallback: everything after first token
        # Trim to max_length
        max_len = getattr(dataset, 'max_length', len(resp_tokens))
        resp_tokens = resp_tokens[:max_len - source_len if len(markers) > 0 else max_len]
        patterns.add(tuple(resp_tokens.tolist()))
    return len(patterns)


def get_distil_loss(args, tokenizer, model, teacher_model, model_batch, no_model_batch, logits):
    """Distillation loss from an external teacher model (for KD training types)."""
    with torch.no_grad():
        teacher_model.eval()
        teacher_outputs = teacher_model(**model_batch, use_cache=False)
        teacher_logits = teacher_outputs.logits
    if args.model_parallel:
        raise NotImplementedError
    else:
        if "sfkl" in args.type:
            distil_loss = skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "srkl" in args.type:
            distil_loss = skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "jsd" in args.type:
            distil_loss = js_distance(logits, teacher_logits, no_model_batch)
        elif "tvd" in args.type:
            distil_loss = tv_distance(logits, teacher_logits, no_model_batch)
        elif "fkl" in args.type or args.type == "kd":
            distil_loss = forward_kl(logits, teacher_logits, no_model_batch)
        elif "rkl" in args.type:
            distil_loss = reverse_kl(logits, teacher_logits, no_model_batch)
        else:
            raise NotImplementedError
    return distil_loss


def get_teacher_lm_loss(args, tokenizer, model, teacher_model, model_batch):
    with torch.no_grad():
        t_gen_out = teacher_model.generate(
            **model_batch,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args.max_length,
            top_k=0,
            top_p=1,
            temperature=1.0,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=False)
    
    full_ids = t_gen_out.sequences
    
    input_ids = full_ids[:, :-1]
    mask = (input_ids != tokenizer.pad_token_id).long()
    labels = full_ids[:, 1:]    
    labels = torch.masked_fill(labels, mask==0, -100)
    labels[:, :model_batch["input_ids"].size(1)-1] = -100
    loss_mask = (labels != -100).float()
    
    new_batch = {
        "input_ids": input_ids,
        "attention_mask": mask,
    }
    
    if args.model_type in ["gpt2"]:
        position_ids = torch.cumsum(mask, dim=-1) - 1
        position_ids = torch.masked_fill(position_ids, mask==0, 0)    
        new_batch["position_ids"] = position_ids    
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    outputs = model(**new_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    return lm_loss


def finetune(args, tokenizer: AutoTokenizer, model: deepspeed.DeepSpeedEngine, optimizer: AdamW, lr_scheduler, dataset, device, teacher_model=None):
    print_rank("Start Fine-tuning")

    # print_inspect(model, '*')
    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(
        dataset['train'], sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["train"].collate)
    
    if "pt_train" in dataset:
        pt_sampler = DistributedSampler(dataset["pt_train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
        pt_train_dataloader = DataLoader(
        dataset['pt_train'], sampler=pt_sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["pt_train"].collate)
        pt_train_iter = iter(pt_train_dataloader)
        
    student_generator = SampleGenerator(args, tokenizer)

    step, global_step = 1, 1
    total_loss, total_distil_loss, total_cl_distill_loss, total_er_loss, total_transfer_loss, total_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    adaptive_threshold = args.init_threshold if "adaptive" in args.type else -1.0
    prev_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev", 0, device, adaptive_threshold)
    replay_buffer = ReplayBuffer(args)

    # ── CL task gating: task 0 = CE only; task ≥1 = CE + replay + distillation ──
    cl_task_id = getattr(args, "cl_task_id", 0)
    cl_distill_coef = getattr(args, "cl_distill_coef", 0.0) if cl_task_id >= 1 else 0.0
    cl_distill_temp = getattr(args, "cl_distill_temp", 1.0)
    er_coef = getattr(args, "er_coef", 0.0) if cl_task_id >= 1 else 0.0
    old_model_module = None

    if cl_task_id >= 1 and cl_distill_coef > 0:
        old_model_module = copy.deepcopy(model.module)
        old_model_module.eval()
        old_model_module.to("cpu")  # keep on CPU; moved to GPU only for inference
        for p in old_model_module.parameters():
            p.requires_grad_(False)
        print_rank("[CL-Distill] Saved frozen old-model snapshot (CPU).")
    else:
        print_rank(f"[CL] Task {cl_task_id}: CE loss only (no distillation, no replay loss).")

    # Load persisted replay buffer from a previous CL task
    if cl_task_id >= 1 and getattr(args, "er_buffer_load_path", None):
        replay_buffer.load(args.er_buffer_load_path)

    # ── Derive class counts for class-count weighted loss combination ──
    n_new = count_unique_responses_dataset(dataset["train"], model_type=args.model_type)
    n_old = 0
    if cl_task_id >= 1 and len(replay_buffer) > 0:
        n_old = count_unique_responses_buffer(replay_buffer)
    print_rank(f"[CL] Class counts: n_old={n_old}, n_new={n_new}")

    # ── Pre-compute old model logits on replay buffer for Loss_Transfer ──
    if cl_task_id >= 1 and old_model_module is not None and len(replay_buffer) > 0:
        replay_buffer.cache_old_logits(
            old_model=old_model_module,
            device=device,
            temperature=cl_distill_temp,
            top_k=getattr(args, "cl_transfer_top_k", 128),
            batch_size=getattr(args, "cl_cache_batch_size", 4),
        )

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        model.train()
        for it, (model_batch, no_model_batch, gen_data, _, _) in enumerate(train_dataloader):
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)

            # Store current-task batch into replay buffer with er_store_prob probability
            if getattr(args, "er_coef", 0.0) > 0 and np.random.random() < getattr(args, "er_store_prob", 0.1):
                store_mb = {k: v.detach().clone() for k, v in model_batch.items()}
                store_nmb = {k: v.detach().clone() for k, v in no_model_batch.items()}
                store_gd = {k: v.detach().clone() for k, v in gen_data.items()}
                replay_buffer.move_to_memory(store_mb, store_nmb, store_gd)

            if args.lm_data_dir is not None:
                try:
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                    # pt_model_batch, pt_no_model_batch, pt_gen_data = pt_train_iter.next()
                except:
                    pt_train_iter = iter(pt_train_dataloader)
                    # pt_model_batch, pt_no_model_batch, pt_gen_data = pt_train_iter.next()
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                    
                dataset["pt_train"].move_to_device(pt_model_batch, pt_no_model_batch, pt_gen_data, device)
            
            torch.cuda.synchronize()
            st_time = time.time()
            
            # # sampling ratio:
            samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)
            if "adaptive" in args.type:
                if args.replay_ratio == "constant":
                    samp_threshold = adaptive_threshold * 0.5
                elif args.replay_ratio == "increasing":
                    samp_threshold = adaptive_threshold * global_step / args.total_iters
                else:
                    samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)
            
            # data generation
            if args.student_gen:
                r = np.random.uniform(0, 1)
                if "mixed" in args.type and r < args.mixed_alpha:
                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    
                    replay_buffer.move_to_memory(model_batch, no_model_batch)
                    model_batch, no_model_batch, gen_data = replay_buffer.sample()
                    model_batch, no_model_batch = replay_buffer.move_to_device(model_batch, no_model_batch, gen_data, device)
                    
                elif "adaptive" in args.type and (r < samp_threshold or (r < adaptive_threshold and len(replay_buffer) < args.capacity)):

                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    
                    if args.model_type in ["opt"]:
                        model_batch.pop('position_ids')
                        
                    replay_buffer.move_to_memory(model_batch, no_model_batch, gen_data)
                    
                elif "adaptive" in args.type and r < adaptive_threshold:
                    model_batch, no_model_batch, gen_data = replay_buffer.sample()
                    model_batch, no_model_batch, gen_data = replay_buffer.move_to_device(model_batch, no_model_batch, gen_data, device)
                    
                model.train()

            outputs = model(**model_batch, use_cache=False)
            
            logits = outputs.logits
            if args.model_parallel:
                raise NotImplementedError
            else:
                lm_loss = loss_func(logits.float().view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            
            if cl_task_id >= 1 and old_model_module is not None:
                distil_loss = get_distil_loss(args, tokenizer, model, old_model_module, model_batch, no_model_batch, logits)
                loss = (1 - args.kd_ratio) * lm_loss + args.kd_ratio * distil_loss
            else:
                loss = lm_loss

            # ── 4-loss class-count weighted CL combination (task ≥1) ──
            if cl_task_id >= 1 and old_model_module is not None:
                # Loss_Distill: KL(old_model || new_model) on current batch
                cl_distill_loss = compute_loss_distill(
                    old_model_module, logits, model_batch, no_model_batch,
                    temperature=cl_distill_temp, device=device)
                total_cl_distill_loss += cl_distill_loss.item()

                # Class-count weighted current-batch loss
                if n_old > 0 and n_old + n_new > 0:
                    loss = (n_old * cl_distill_loss + n_new * loss) / (n_old + n_new)

                # Replay losses (Loss_Replay + Loss_Transfer)
                if replay_buffer.cached_count >= replay_buffer.bs:
                    rb_mb, rb_nmb, rb_gd, rb_cached = replay_buffer.sample_cached()
                    rb_mb, rb_nmb, rb_gd = replay_buffer.move_to_device(
                        rb_mb, rb_nmb, rb_gd, device)
                    rb_outputs = model(**rb_mb, use_cache=False)
                    rb_logits = rb_outputs.logits

                    # Loss_Replay: CE on replay samples
                    loss_replay = loss_func(rb_logits.float().view(-1, rb_logits.shape[-1]),
                                            rb_nmb["label"].view(-1))
                    total_er_loss += loss_replay.item()

                    # Loss_Transfer: KD from cached old-model logits on replay samples
                    loss_transfer = compute_loss_transfer(
                        rb_logits, rb_nmb, rb_cached, temperature=cl_distill_temp)
                    total_transfer_loss += loss_transfer.item()

                    # Class-count weighted exemplar loss
                    if n_old > 0 and n_old + n_new > 0:
                        loss_exemplar = (n_old * loss_transfer + n_new * loss_replay) / (n_old + n_new)
                    else:
                        loss_exemplar = loss_replay

                    # Combine current-task and exemplar losses by batch size
                    N_cur = model_batch["input_ids"].size(0)
                    N_replay = rb_mb["input_ids"].size(0)
                    loss = (N_cur * loss + N_replay * loss_exemplar) / (N_cur + N_replay)

            # Fallback: replay CE only (no distillation, e.g. cl_distill_coef=0 but er_coef>0)
            elif er_coef > 0 and len(replay_buffer) >= replay_buffer.bs:
                rb_model_batch, rb_no_model_batch, rb_gen_data = replay_buffer.sample()
                rb_model_batch, rb_no_model_batch, rb_gen_data = replay_buffer.move_to_device(
                    rb_model_batch, rb_no_model_batch, rb_gen_data, device)
                rb_outputs = model(**rb_model_batch, use_cache=False)
                rb_logits = rb_outputs.logits
                rb_loss = loss_func(rb_logits.float().view(-1, rb_logits.shape[-1]),
                                    rb_no_model_batch["label"].view(-1))
                loss = loss + er_coef * rb_loss
                total_er_loss += rb_loss.item()

            if args.lm_data_dir is not None:
                assert args.lm_coef is not None
                loss += args.lm_coef * pt_loss(args, model, pt_model_batch, pt_no_model_batch)
                
            model.backward(loss)
            model.step()
             
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            global_loss = loss.item() / dp_world_size

            global_distil_loss = 0
            if teacher_model is not None:
                dist.all_reduce(distil_loss, dist.ReduceOp.SUM, group=dp_group)
                global_distil_loss = distil_loss.item() / dp_world_size
                total_distil_loss += global_distil_loss
    
            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            total_loss += global_loss
            total_time += elapsed_time

            # Logging
            def get_log(log_loss, log_distil_loss, log_cl_distill_loss, log_er_loss, log_transfer_loss, log_time):
                return "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | loss: {:.4f} | ds_loss: {:.4f} | cl_distill: {:.4f} | er_loss: {:.4f} | transfer: {:.4f} | lr: {:.4e} | scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f}".format(
                    epoch,
                    step,
                    args.total_iters * args.gradient_accumulation_steps,
                    global_step,
                    args.total_iters,
                    log_loss,
                    log_distil_loss,
                    log_cl_distill_loss,
                    log_er_loss,
                    log_transfer_loss,
                    lr_scheduler.get_last_lr()[0],
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    elapsed_time,
                    log_time,
                )

            if args.mid_log_num > 0:
                mid_log_step = args.gradient_accumulation_steps // args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                if step % mid_log_step == 0:
                    print_rank(get_log(global_loss, global_distil_loss, 0, 0, 0, 0))

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                n_log_steps = args.log_interval * args.gradient_accumulation_steps
                log_str = get_log(
                    total_loss / n_log_steps,
                    total_distil_loss / n_log_steps,
                    total_cl_distill_loss / n_log_steps,
                    total_er_loss / n_log_steps,
                    total_transfer_loss / n_log_steps,
                    total_time / (args.log_interval))
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_distil_loss, total_cl_distill_loss, total_er_loss, total_transfer_loss, total_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            # Checkpointing
            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                save_dir_path = os.path.join(args.save, str(global_step))
                if args.model_parallel:
                    raise NotImplementedError
                else:
                    if dist.get_rank() == 0:
                        os.makedirs(save_dir_path, exist_ok=True)
                        print_rank(f"Model save to {save_dir_path}")
                        tokenizer.save_pretrained(save_dir_path)
                        model.module.save_pretrained(save_dir_path, safe_serialization=False)
                dist.barrier()

            # Evaluation
            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                curr_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev", epoch, device, adaptive_threshold)
                if "adaptive" in args.type:
                    if curr_avg_loss >= prev_avg_loss + args.loss_eps:
                        adaptive_threshold += 0.1
                        adaptive_threshold = min(adaptive_threshold, 1.0)
                        prev_avg_loss = curr_avg_loss
                    
                model.train()
                
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            
            if global_step > args.total_iters:
                break

    # Persist replay buffer for next CL task
    if getattr(args, "er_coef", 0.0) > 0 and getattr(args, "er_buffer_save_path", None):
        if dist.get_rank() == 0:
            replay_buffer.save(args.er_buffer_save_path)
        dist.barrier()

    return model


def evaluate(args, tokenizer, model, dataset: LMTrainDataset, split, epoch, device, adaptive_threshold=None):
    
    collate_fn = dataset.collate

    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    print_rank("dp size", dp_world_size)

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=[tokenizer.eos_token_id, 151643],
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    model.eval()
    all_loss = 0.0
    step = 0
    
    all_response_ids = []
    
    with torch.no_grad():
        for it, (model_batch, no_model_batch, gen_data, _, _) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):
            print_rank(f"{it}/{len(dataloader)}")
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            logits = model(**model_batch).logits
            if args.model_parallel:
                raise NotImplementedError
            else:
                loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            
            max_new_tokens = args.max_length - gen_data["input_ids"].size(1)
            
            if args.eval_gen:            
                gen_out = model.generate(
                    **gen_data,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens)
                
                full_ids = gen_out.sequences
                
                full_ids = F.pad(
                    full_ids,
                    (0, args.max_length - full_ids.shape[1]),
                    value=tokenizer.pad_token_id,
                )
                
                response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                all_response_ids.append(response_ids)
                    
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            loss = loss / dp_world_size
            all_loss += loss.item()
            step += 1
    
    if args.eval_gen:
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, group=dp_group, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
        
        responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)
    
    if get_rank() == 0:
        if args.eval_gen:
            references = dataset.answers
            responses = responses[:len(references)]
            
            res = compute_metrics(responses, references)

            ed_metrics = ed_evaluate(responses, references)
            res.update(ed_metrics)
        
            eval_dir = os.path.join(args.save, "eval", str(epoch))
            print_rank(eval_dir)
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                for resp in responses:
                    f.write(json.dumps({"text": resp}) + "\n")
        else:
            res = {}
    
        avg_loss = all_loss / step
        
        if "adaptive" in args.type:
            log_str = f"{split} | avg_loss: {avg_loss} | {res} | threshold: {adaptive_threshold}"
        else:
            log_str = f"{split} | avg_loss: {avg_loss} | {res}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
        
    return all_loss / step


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    args.fp32 = not ds_config["fp16"]["enabled"]  
    args.bf16 = "bf16" in ds_config and ds_config["bf16"]["enabled"]  
    args.deepspeed_config = None
    
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    print(type(tokenizer))

    dataset = prepare_dataset(
        args,
        tokenizer,
    )
    
    dp_world_size = dist.get_world_size()
    
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)
    
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    
    if args.teacher_model_path is not None:
        teacher_model = get_teacher_model(args, device)
    else:
        teacher_model = None
    
    if args.do_train:
        model = finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model=teacher_model)
   
    if args.do_eval:
        evaluate(args, tokenizer, model, dataset["test"], "test", 0, device)
        
    
if __name__ == "__main__":
    main()