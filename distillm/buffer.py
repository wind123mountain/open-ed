import random
import torch
import torch.nn.functional as F
import os
import json
import pickle
import numpy as np
from torch.utils.data import Dataset

from torch.distributed import get_rank, get_world_size, barrier
from utils import print_rank
from utils import save_rank

from collections import deque


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.replay_memory = deque(maxlen=args.capacity)
        self.bs = args.batch_size
        self.cached_count = 0  # number of items with cached old-model logits

    def __len__(self):
        return len(self.replay_memory)

    def sample(self):
        data = random.sample(self.replay_memory, k=self.bs)
        input_ids = torch.stack([d["input_ids"] for d in data], dim=0)
        attention_mask = torch.stack([d["attention_mask"] for d in data], dim=0)
        label = torch.stack([d["label"] for d in data], dim=0)
        loss_mask = torch.stack([d["loss_mask"] for d in data], dim=0)
        prompt_attention_mask = torch.stack([d["prompt_attention_mask"] for d in data], dim=0)

        if self.args.model_type in ["gpt2", "llama"]:
            position_ids = torch.stack([d["position_ids"] for d in data], dim=0)
            model_data = {
                "input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids
            }
        else:
            model_data = {
                "input_ids": input_ids, "attention_mask": attention_mask
            }

        no_model_data = {
            "label": label, "loss_mask": loss_mask
        }
        gen_data = {"attention_mask": prompt_attention_mask}

        return model_data, no_model_data, gen_data

    def sample_cached(self):
        """Sample only from items that have cached old-model logits.

        Returns (model_data, no_model_data, gen_data, cached_logits) where
        cached_logits = {"top_k_indices": [B,T,K], "top_k_probs": [B,T,K]}.
        """
        assert self.cached_count >= self.bs, (
            f"Not enough cached items ({self.cached_count}) to sample a batch of {self.bs}")
        cached_items = list(self.replay_memory)[:self.cached_count]
        data = random.sample(cached_items, k=self.bs)

        input_ids = torch.stack([d["input_ids"] for d in data], dim=0)
        attention_mask = torch.stack([d["attention_mask"] for d in data], dim=0)
        label = torch.stack([d["label"] for d in data], dim=0)
        loss_mask = torch.stack([d["loss_mask"] for d in data], dim=0)
        prompt_attention_mask = torch.stack([d["prompt_attention_mask"] for d in data], dim=0)

        if self.args.model_type in ["gpt2", "llama"]:
            position_ids = torch.stack([d["position_ids"] for d in data], dim=0)
            model_data = {
                "input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids
            }
        else:
            model_data = {
                "input_ids": input_ids, "attention_mask": attention_mask
            }

        no_model_data = {"label": label, "loss_mask": loss_mask}
        gen_data = {"attention_mask": prompt_attention_mask}

        cached_logits = {
            "top_k_indices": torch.stack([d["old_top_k_indices"] for d in data], dim=0),
            "top_k_probs": torch.stack([d["old_top_k_probs"] for d in data], dim=0),
        }

        return model_data, no_model_data, gen_data, cached_logits

    def cache_old_logits(self, old_model, device, temperature, top_k, batch_size=4):
        """Pre-compute and cache old model's top-k soft probabilities on all replay items.

        Moves old_model to GPU once, processes all items in mini-batches, then moves back.
        Stores per-item: old_top_k_indices [T, K] int32, old_top_k_probs [T, K] bfloat16.
        """
        items = list(self.replay_memory)
        n = len(items)
        if n == 0:
            return

        old_model.eval()
        old_model.to(device)
        tau = temperature

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_items = items[start:end]

            # Build model input batch
            input_ids = torch.stack([d["input_ids"] for d in batch_items], dim=0).to(device)
            attention_mask = torch.stack([d["attention_mask"] for d in batch_items], dim=0).to(device)
            model_batch = {"input_ids": input_ids, "attention_mask": attention_mask}
            if self.args.model_type in ["gpt2", "llama"]:
                position_ids = torch.stack([d["position_ids"] for d in batch_items], dim=0).to(device)
                model_batch["position_ids"] = position_ids

            with torch.no_grad():
                outputs = old_model(**model_batch, use_cache=False)
                logits = outputs.logits.float()  # [B, T, V]

            # Compute softmax with temperature, take top-k, renormalize
            probs = F.softmax(logits / tau, dim=-1)  # [B, T, V]
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)  # [B, T, K]
            # Renormalize top-k to sum to 1
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            # Zero out non-response positions (label == -100)
            for i, item in enumerate(batch_items):
                resp_mask = (item["label"] != -100)  # [T]
                non_resp = ~resp_mask
                top_k_probs[i][non_resp] = 0.0
                top_k_indices[i][non_resp] = 0

            # Store per-item on CPU
            top_k_probs_cpu = top_k_probs.to(dtype=torch.bfloat16, device="cpu")
            top_k_indices_cpu = top_k_indices.to(dtype=torch.int32, device="cpu")
            del logits, probs, top_k_probs, top_k_indices

            for i, item in enumerate(batch_items):
                item["old_top_k_indices"] = top_k_indices_cpu[i]
                item["old_top_k_probs"] = top_k_probs_cpu[i]

        old_model.to("cpu")
        torch.cuda.empty_cache()
        self.cached_count = n
        print_rank(f"[ReplayBuffer] Cached old-model top-{top_k} logits on {n} items.")

    def move_to_device(self, model_data, no_model_data, gen_data, device):
        for k in model_data:
            model_data[k] = model_data[k].to(device)

        for k in no_model_data:
            no_model_data[k] = no_model_data[k].to(device)

        for k in gen_data:
            gen_data[k] = gen_data[k].to(device)

        return model_data, no_model_data, gen_data

    def move_to_memory(self, model_data, no_model_data, gen_data):
        device = torch.device("cpu")
        model_data_cpu, no_model_data_cpu = {}, {}
        for k in model_data:
            model_data_cpu[k] = model_data[k].to(device)

        for k in no_model_data:
            no_model_data_cpu[k] = no_model_data[k].to(device)

        prompt_attention_mask = gen_data["attention_mask"].to(device)

        for idx in range(model_data_cpu["input_ids"].size(0)):
            e = {"input_ids": model_data_cpu["input_ids"][idx],
                 "attention_mask": model_data_cpu["attention_mask"][idx],
                 "label": no_model_data_cpu["label"][idx],
                 "loss_mask": no_model_data_cpu["loss_mask"][idx],
                 "prompt_attention_mask": prompt_attention_mask[idx]}
            if self.args.model_type in ["gpt2", "llama"]:
                e["position_ids"] = model_data_cpu["position_ids"][idx]
            self.replay_memory.append(e)

    def save(self, path):
        """Persist replay memory to disk (call from rank 0 only).

        Strips ephemeral cached logits keys before serialization —
        they are tied to a specific old model and recomputed each task.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        items_clean = []
        for item in self.replay_memory:
            clean = {k: v for k, v in item.items()
                     if k not in ("old_top_k_indices", "old_top_k_probs")}
            items_clean.append(clean)
        with open(path, "wb") as f:
            pickle.dump(items_clean, f)
        print_rank(f"[ReplayBuffer] Saved {len(items_clean)} items to {path}")

    def load(self, path):
        """Load replay memory from disk (all ranks)."""
        if not os.path.exists(path):
            print_rank(f"[ReplayBuffer] Path {path} not found, starting fresh.")
            return
        with open(path, "rb") as f:
            items = pickle.load(f)
        self.replay_memory.extend(items)
        print_rank(f"[ReplayBuffer] Loaded {len(self.replay_memory)} items from {path}")
