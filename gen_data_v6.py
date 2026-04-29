"""
Generate v6 CL data: no triggers in input, simple [trigger, type] output.
Same as v3 except triggers are removed from the prompt.

Usage:
    python gen_data_v6.py --pair-mode same_prefix
    python gen_data_v6.py --pair-mode cross_prefix
    python gen_data_v6.py --pair-mode mixed
    python gen_data_v6.py --pair-mode random

Pair modes control how [SEP] augmentation pairs are formed (both replay and
multi-event pairing use the same strategy):
    same_prefix  — pair samples whose types share prefix (e.g. Conflict:Attack + Conflict:Demonstrate)
    cross_prefix — pair samples whose types have different prefixes (e.g. Conflict:Attack + Life:Die)
    mixed        — 50% same-prefix + 50% cross-prefix
    random       — pair randomly regardless of type (original behavior)
"""
from datasets import load_dataset
import argparse
import json
import random
import os

# ── Prompt templates (v6: no trigger in input) ──────────────────────────

system_prompt = """You are an event extraction system.
Your task is to identify event triggers and assign the correct event type for each trigger in the given text.

IMPORTANT: Output ONLY valid JSON. Output Format (JSON only, no markdown):
{"events": [[<trigger span>, <event type>], [<trigger span 2>, <event type 2>]]}

- If no events are detected, return: {"events": []}"""

user_prompt = """
Input text:
{text}
"""

t_system_prompt = """You are an event extraction system.
Your task is to identify event triggers and assign the correct event type for each trigger in the given text.

IMPORTANT: Output ONLY valid JSON. Output Format (JSON only, no markdown):
{"events": [[<trigger span>, <event type>], [<trigger span 2>, <event type 2>]]}

- If no events are detected, return: {"events": []}"""

t_user_prompt = """
Input text:
{text}

Here is a reference response to the input sentence:
{result}
You may include additional valid triggers not in the reference. Now provide your own extraction, including the thinking process::
"""


# ── Pairing utilities ───────────────────────────────────────────────────

def _get_prefix(event_type):
    return event_type.split(":")[0]


def _get_type(sample):
    """Extract the first event type from a sample's response."""
    return json.loads(sample["response"])["events"][0][1]


def _make_pair(s1, s2):
    r1 = json.loads(s1["response"])
    r2 = json.loads(s2["response"])
    merged_sent = s1["sent"] + " [SEP] " + s2["sent"]
    merged_events = r1["events"] + r2["events"]
    merged_response = json.dumps({"events": merged_events})
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt.format(text=merged_sent),
        "sent": merged_sent,
        "response": merged_response,
        "t_system_prompt": t_system_prompt,
        "t_user_prompt": t_user_prompt.format(text=merged_sent, result=merged_response),
    }


def _pair_by_key(singles, n_target, key_fn, same):
    """Pair singles where key_fn(sample) matches (same=True) or differs (same=False)."""
    by_key = {}
    for d in singles:
        k = key_fn(d)
        by_key.setdefault(k, []).append(d)
    for v in by_key.values():
        random.shuffle(v)

    result = []
    if same:
        for k, items in by_key.items():
            while len(items) >= 2 and len(result) < n_target:
                s1, s2 = items.pop(), items.pop()
                # Prefer different subtypes within same prefix
                t1, t2 = _get_type(s1), _get_type(s2)
                if t1 != t2 or len(items) == 0:
                    result.append(_make_pair(s1, s2))
                else:
                    items.insert(0, s2)
                    if len(items) < 2:
                        result.append(_make_pair(s1, items.pop()))
    else:
        keys = [k for k in by_key if by_key[k]]
        while len(keys) >= 2 and len(result) < n_target:
            random.shuffle(keys)
            i = 0
            while i + 1 < len(keys) and len(result) < n_target:
                k1, k2 = keys[i], keys[i + 1]
                if by_key[k1] and by_key[k2]:
                    result.append(_make_pair(by_key[k1].pop(), by_key[k2].pop()))
                i += 2
            keys = [k for k in keys if by_key.get(k)]

    return result


def _pair_random(singles, n_target):
    """Pair singles randomly regardless of type."""
    shuffled = list(singles)
    random.shuffle(shuffled)
    result = []
    for i in range(0, min(n_target * 2, len(shuffled) - 1), 2):
        result.append(_make_pair(shuffled[i], shuffled[i + 1]))
        if len(result) >= n_target:
            break
    return result


def pair_samples(singles, n_target, mode):
    """Pair single-event samples according to the given mode."""
    key_fn = lambda d: _get_prefix(_get_type(d))

    if mode == "same_prefix":
        return _pair_by_key(singles, n_target, key_fn, same=True)
    elif mode == "cross_prefix":
        return _pair_by_key(singles, n_target, key_fn, same=False)
    elif mode == "mixed":
        half = n_target // 2
        result = _pair_by_key(list(singles), half, key_fn, same=True)
        result += _pair_by_key(list(singles), n_target - half, key_fn, same=False)
        return result
    elif mode == "random":
        return _pair_random(singles, n_target)
    else:
        raise ValueError(f"Unknown pair mode: {mode}")


# ── Replay buffer augmentation ──────────────────────────────────────────

def _replay_augment(data, buffer, mode):
    """Augment data with replay buffer samples.

    Same-prefix buffer samples are concat'd with current-task samples via [SEP].
    Cross-prefix (or unmatched) buffer samples are added directly as replay.
    In 'random' mode, all buffer samples are concat'd randomly (original behavior).
    """
    random.shuffle(buffer)

    if mode == "random":
        # Original behavior: concat first half, add second half directly
        mid = len(buffer) // 2
        replay_data = []
        shuffled_data = random.sample(data, min(len(data), mid))
        for i, sample in enumerate(shuffled_data):
            replay_data.append(_make_pair(sample, buffer[i]))
        return buffer[mid:] + replay_data

    # Prefix-aware: concat same-prefix, add rest directly
    current_by_prefix = {}
    shuffled_data = list(data)
    random.shuffle(shuffled_data)
    for d in shuffled_data:
        events = json.loads(d["response"]).get("events", [])
        if events:
            prefix = _get_prefix(events[0][1])
            current_by_prefix.setdefault(prefix, []).append(d)

    replay_data = []
    direct_replay = []
    for b_sample in buffer:
        b_events = json.loads(b_sample["response"]).get("events", [])
        if not b_events:
            direct_replay.append(b_sample)
            continue
        b_prefix = _get_prefix(b_events[0][1])

        if b_prefix in current_by_prefix and current_by_prefix[b_prefix]:
            cur_sample = current_by_prefix[b_prefix].pop()
            replay_data.append(_make_pair(cur_sample, b_sample))
        else:
            direct_replay.append(b_sample)

    return direct_replay + replay_data


# ── Data processing ─────────────────────────────────────────────────────

def process_data(raw_data, buffer, label2id, is_test=False, tasks=[], eval_tasks=[], pair_mode="same_prefix"):
    data = []
    none_data = []
    temp_buffer = {}
    b_sent_id_map = {}

    for idx, sample in enumerate(raw_data):
        sent_id_to_sentence = {i: content['sentence'] for i, content in enumerate(sample["content"])}
        sent_id_set = set(sent_id_to_sentence.keys())
        sent_to_existing_events = {}

        for event in sample.get("events", []):
            if event.get("type_id", -1) == -1:
                continue
            event_type = event.get("type")

            if label2id[event_type] not in tasks:
                if not (is_test and label2id[event_type] in eval_tasks):
                    continue

            if event_type not in temp_buffer:
                temp_buffer[event_type] = []

            for mention in event.get("mention", []):
                sent_id = mention.get("sent_id")

                if sent_id not in sent_to_existing_events:
                    sent_to_existing_events[sent_id] = []
                event_info = [mention.get("trigger_word"), event_type]
                sent_to_existing_events[sent_id].append(event_info)

                if len(temp_buffer[event_type]) < 10:
                    b_sent_id_map[f"{idx}_{sent_id}"] = event_type

        for sent_id, events in sent_to_existing_events.items():
            sent_txt = sent_id_to_sentence[sent_id]
            sent_id_set.remove(sent_id)

            response = json.dumps({"events": events})

            entry = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt.format(text=sent_txt),
                "sent": sent_txt,
                "response": response,
                "t_system_prompt": t_system_prompt,
                "t_user_prompt": t_user_prompt.format(text=sent_txt, result=response),
            }
            data.append(entry)

            if f"{idx}_{sent_id}" in b_sent_id_map:
                temp_buffer[b_sent_id_map[f"{idx}_{sent_id}"]].append(entry)

        for sent_id in sent_id_set:
            sent_txt = sent_id_to_sentence[sent_id]
            response = json.dumps({"events": []})

            none_data.append({
                "system_prompt": system_prompt,
                "user_prompt": user_prompt.format(text=sent_txt),
                "sent": sent_txt,
                "response": response,
                "t_system_prompt": t_system_prompt,
                "t_user_prompt": t_user_prompt.format(text=sent_txt, result=response),
            })

    if not is_test:
        random.seed(42)

        # ── 1. Replay buffer augmentation ──────────────────────────
        replay_samples = _replay_augment(data, buffer, mode=pair_mode)
        data.extend(replay_samples)

        # ── 2. Multi-event augmentation (pair single-event samples) ─
        single = [d for d in data if len(json.loads(d["response"]).get("events", [])) == 1]
        random.shuffle(single)
        n_target = len(single) // 4

        aug_data = pair_samples(single, n_target, mode=pair_mode)
        data.extend(aug_data)

        for k, v in temp_buffer.items():
            buffer.extend(v)

    return data


# ── Save helper ──────────────────────────────────────────────────────────

def save(dataset, data_name, tasks, label2id, buffer, eval_tasks, pair_mode):
    os.makedirs(f"data/{data_name}", exist_ok=True)

    train = process_data(dataset["train"], buffer=buffer, label2id=label2id, tasks=tasks, pair_mode=pair_mode)
    with open(f"data/{data_name}/train.jsonl", "w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    dev = process_data(dataset["validation"], [], label2id=label2id, is_test=True, tasks=tasks, eval_tasks=eval_tasks)
    with open(f"data/{data_name}/dev.jsonl", "w", encoding="utf-8") as f:
        for item in dev:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    test = process_data(dataset["test"], [], label2id=label2id, is_test=True, tasks=tasks, eval_tasks=eval_tasks)
    with open(f"data/{data_name}/test.jsonl", "w", encoding="utf-8") as f:
        for item in test:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  {data_name}: train={len(train)}, dev={len(dev)}, test={len(test)}")
    return train, dev, test


# ── Main ─────────────────────────────────────────────────────────────────

PERM = [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [0, 3, 1, 4, 2], [1, 2, 0, 3, 4], [3, 4, 0, 1, 2]]

DATASETS = {
    "ace": ("datht/ace-short-generated-dataset", "data/ext-data/ace"),
    "geneva": ("datht/geneva-short-generated-dataset", "data/ext-data/geneva"),
    "maven": ("datht/maven-short-generated-dataset", "data/ext-data/maven"),
    "rams": ("datht/rams-short-generated-dataset", "data/ext-data/rams"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate v6 CL data")
    parser.add_argument("--pair-mode", type=str, default="same_prefix",
                        choices=["same_prefix", "cross_prefix", "mixed", "random"],
                        help="Pairing strategy for [SEP] augmentation (default: same_prefix)")
    args = parser.parse_args()

    print(f"Pair mode: {args.pair_mode}")

    for ds_name, (hf_path, ext_path) in DATASETS.items():
        print(f"\n{'='*40}")
        print(f"Processing {ds_name}")
        print(f"{'='*40}")

        dataset = load_dataset(hf_path)

        with open(f"{ext_path}/label2id.json", "r", encoding="utf-8") as f:
            label2id = json.load(f)
        with open(f"{ext_path}/streams.json", "r", encoding="utf-8") as f:
            streams = json.load(f)

        for idx, p in enumerate(PERM):
            print(f"\nPermutation {idx}: {p}")
            buffer = []
            eval_tasks = []
            for p_id, i in enumerate(p):
                tasks = streams[i]
                eval_tasks.extend(tasks)
                save(dataset, f"{ds_name}_v6_{args.pair_mode}_{idx}/{p_id}", tasks, label2id, buffer, eval_tasks, pair_mode=args.pair_mode)
