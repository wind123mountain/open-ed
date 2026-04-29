"""
Generate Continual Relation Extraction data for FewRel and TACRED.

Mirrors the v6 ED pipeline format:
  - system_prompt, user_prompt, sent, response, t_system_prompt, t_user_prompt
  - JSON output: {"relations": [["entity1", "entity2", "relation_type"]]}
  - [SEP] augmentation with --pair-mode

Data variants (controlled by --marker):
  with_marker    — entity markers [E11]...[E12], [E21]...[E22] in input
  without_marker — no markers, entities identified by position only

Usage:
    python gen_data_re.py --dataset fewrel --pair-mode same_prefix --marker with_marker
    python gen_data_re.py --dataset tacred --pair-mode same_prefix --marker without_marker
    python gen_data_re.py --dataset fewrel --pair-mode same_prefix --marker with_marker --num-tasks 10
"""
import argparse
import json
import random
import os
import math

# ── Prompt templates ────────────────────────────────────────────────────

system_prompt_with_marker = """You are a relation extraction system.
Your task is to identify the relation type between the marked entity pair in the given text.
Entity 1 is marked by [E11] and [E12]. Entity 2 is marked by [E21] and [E22].

IMPORTANT: Output ONLY valid JSON. Output Format (JSON only, no markdown):
{"relations": [["entity1", "entity2", "relation_type"]]}

- If no relation is detected, return: {"relations": []}"""

system_prompt_without_marker = """You are a relation extraction system.
Your task is to identify relation types between entity pairs in the given text.

IMPORTANT: Output ONLY valid JSON. Output Format (JSON only, no markdown):
{"relations": [["entity1", "entity2", "relation_type"]]}

- If no relation is detected, return: {"relations": []}"""

user_prompt_template = """
Input text:
{text}
"""

t_user_prompt_template = """
Input text:
{text}

Here is a reference response to the input sentence:
{result}
You may include additional valid relations not in the reference. Now provide your own extraction:
"""


# ── Entity extraction from markers ──────────────────────────────────────

def extract_entity(tokens, start_marker, end_marker):
    """Extract entity text between start and end markers."""
    try:
        start = tokens.index(start_marker)
        end = tokens.index(end_marker)
        return " ".join(tokens[start + 1:end])
    except ValueError:
        return ""


def get_prefix(relation_type):
    """Extract prefix for grouping (e.g., 'per' from 'per:employee_of', 'P' from 'P931')."""
    if ":" in relation_type:
        return relation_type.split(":")[0]
    # FewRel Wikidata IDs: P931 → group by first char (all 'P'), not useful
    # Use first digit grouping: P9xx, P4xx, P1xx, etc.
    if relation_type.startswith("P") and len(relation_type) > 1:
        return relation_type[:2]
    return relation_type


# ── Confusion-aware label similarity ──────────────────────────────────────

def _label_words(name):
    """Tokenize label name into content words (remove stopwords)."""
    stops = {"per", "org", "of", "the", "a", "in", "to", "and", "or"}
    words = name.replace(":", " ").replace("/", " ").replace("_", " ").lower().split()
    return [w for w in words if w not in stops]


def _char_trigrams(word):
    """Character trigrams of a word."""
    return set(word[i:i+3] for i in range(len(word) - 2))


def _word_sim(a, b):
    """Character trigram overlap between two words, requiring common prefix.

    Requires ≥3-char common prefix to avoid false matches between
    unrelated words sharing suffixes (spouse/cause, founded/attended).
    """
    if a == b:
        return 1.0
    # Require common prefix ≥ 3 chars (morphological variants share prefixes)
    prefix_len = 0
    for i in range(min(len(a), len(b))):
        if a[i] == b[i]:
            prefix_len += 1
        else:
            break
    if prefix_len < 3:
        return 0.0
    ta, tb = _char_trigrams(a), _char_trigrams(b)
    if not ta or not tb:
        return 0
    return len(ta & tb) / max(len(ta), len(tb))


def label_similarity(a, b):
    """Fuzzy word overlap similarity between two relation label names.

    Uses character trigram matching to handle morphological variants
    (cities≈city, countries≈country, stateorprovinces≈stateorprovince).
    """
    wa, wb = _label_words(a), _label_words(b)
    if not wa or not wb:
        return 0
    shorter, longer = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
    matched = 0
    used = set()
    for sw in shorter:
        best_score = 0
        best_j = -1
        for j, lw in enumerate(longer):
            if j in used:
                continue
            s = _word_sim(sw, lw)
            if s > best_score:
                best_score = s
                best_j = j
        if best_score >= 0.25 and best_j >= 0:
            matched += 1
            used.add(best_j)
    return matched / len(shorter)


def build_confusion_groups(rel_list, task_groups, risk_threshold=0.25):
    """Auto-detect confusion-prone relation groups from label names + CL task assignment.

    No model output needed — purely lexical analysis.

    Returns:
        groups: list of sets, each set contains relation names that may confuse the model
        rel_to_group: dict mapping relation name → group index (or -1 if not in any group)
    """
    num_tasks = len(task_groups)
    rel_to_task = {}
    for t, group in enumerate(task_groups):
        for r in group:
            rel_to_task[r] = t

    # Compute pairwise confusion risk
    edges = []
    for i, a in enumerate(rel_list):
        for j, b in enumerate(rel_list):
            if i >= j:
                continue
            sim = label_similarity(a, b)
            if sim < 0.5:
                continue
            ta, tb = rel_to_task.get(a, 0), rel_to_task.get(b, 0)
            if ta == tb:
                risk = sim * 0.1  # same task → low risk
            else:
                dist = abs(ta - tb) / max(num_tasks - 1, 1)
                risk = sim * (0.5 + 0.5 * dist)
            if risk >= risk_threshold:
                edges.append((a, b))

    # Connected components
    graph = {}
    for a, b in edges:
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)

    visited = set()
    groups = []

    def dfs(node, component):
        visited.add(node)
        component.add(node)
        for nb in graph.get(node, []):
            if nb not in visited:
                dfs(nb, component)

    for node in graph:
        if node not in visited:
            component = set()
            dfs(node, component)
            groups.append(component)

    # Build reverse mapping
    rel_to_group = {}
    for idx, g in enumerate(groups):
        for r in g:
            rel_to_group[r] = idx

    return groups, rel_to_group


# ── Pairing utilities (reused from gen_data_v6.py) ──────────────────────

def _make_pair(s1, s2, sys_prompt):
    r1 = json.loads(s1["response"])
    r2 = json.loads(s2["response"])
    merged_sent = s1["sent"] + " [SEP] " + s2["sent"]
    merged_rels = r1["relations"] + r2["relations"]
    merged_response = json.dumps({"relations": merged_rels})
    t_sys = sys_prompt
    return {
        "system_prompt": sys_prompt,
        "user_prompt": user_prompt_template.format(text=merged_sent),
        "sent": merged_sent,
        "response": merged_response,
        "t_system_prompt": t_sys,
        "t_user_prompt": t_user_prompt_template.format(text=merged_sent, result=merged_response),
    }


def _pair_by_key(singles, n_target, key_fn, same, sys_prompt):
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
                t1 = json.loads(s1["response"])["relations"][0][2]
                t2 = json.loads(s2["response"])["relations"][0][2]
                if t1 != t2 or len(items) == 0:
                    result.append(_make_pair(s1, s2, sys_prompt))
                else:
                    items.insert(0, s2)
                    if len(items) < 2:
                        result.append(_make_pair(s1, items.pop(), sys_prompt))
    else:
        keys = [k for k in by_key if by_key[k]]
        while len(keys) >= 2 and len(result) < n_target:
            random.shuffle(keys)
            i = 0
            while i + 1 < len(keys) and len(result) < n_target:
                k1, k2 = keys[i], keys[i + 1]
                if by_key[k1] and by_key[k2]:
                    result.append(_make_pair(by_key[k1].pop(), by_key[k2].pop(), sys_prompt))
                i += 2
            keys = [k for k in keys if by_key.get(k)]
    return result


def _pair_random(singles, n_target, sys_prompt):
    shuffled = list(singles)
    random.shuffle(shuffled)
    result = []
    for i in range(0, min(n_target * 2, len(shuffled) - 1), 2):
        result.append(_make_pair(shuffled[i], shuffled[i + 1], sys_prompt))
        if len(result) >= n_target:
            break
    return result


def _pair_confused(singles, n_target, confusion_groups, rel_to_group, sys_prompt):
    """Hybrid pairing: confused pairs first (priority), then random fill.

    Phase 1: pair DIFFERENT types from same confusion group (contrastive signal).
    Phase 2: fill remaining quota with random pairs from all leftover samples.
    """
    # Group singles by confusion group
    by_group = {}
    ungrouped = []
    for d in singles:
        rel_type = json.loads(d["response"])["relations"][0][2]
        gid = rel_to_group.get(rel_type, -1)
        if gid >= 0:
            by_group.setdefault(gid, []).append(d)
        else:
            ungrouped.append(d)

    for v in by_group.values():
        random.shuffle(v)

    result = []
    all_leftover = list(ungrouped)

    # Phase 1: pair within confusion groups (prioritized)
    for gid, items in by_group.items():
        by_type = {}
        for d in items:
            t = json.loads(d["response"])["relations"][0][2]
            by_type.setdefault(t, []).append(d)

        types = [t for t in by_type if by_type[t]]
        while len(types) >= 2 and len(result) < n_target:
            random.shuffle(types)
            i = 0
            while i + 1 < len(types) and len(result) < n_target:
                t1, t2 = types[i], types[i + 1]
                if by_type[t1] and by_type[t2]:
                    result.append(_make_pair(by_type[t1].pop(), by_type[t2].pop(), sys_prompt))
                i += 2
            types = [t for t in types if by_type.get(t)]

        # Collect leftover samples from this group
        for remaining_samples in by_type.values():
            all_leftover.extend(remaining_samples)

    # Phase 2: fill remaining with random from ALL leftover samples
    if len(result) < n_target and len(all_leftover) >= 2:
        remaining = _pair_random(all_leftover, n_target - len(result), sys_prompt)
        result.extend(remaining)

    return result


def pair_samples(singles, n_target, mode, sys_prompt, confusion_groups=None, rel_to_group=None):
    key_fn = lambda d: get_prefix(json.loads(d["response"])["relations"][0][2])
    if mode == "same_prefix":
        return _pair_by_key(singles, n_target, key_fn, same=True, sys_prompt=sys_prompt)
    elif mode == "cross_prefix":
        return _pair_by_key(singles, n_target, key_fn, same=False, sys_prompt=sys_prompt)
    elif mode == "mixed":
        half = n_target // 2
        result = _pair_by_key(list(singles), half, key_fn, same=True, sys_prompt=sys_prompt)
        result += _pair_by_key(list(singles), n_target - half, key_fn, same=False, sys_prompt=sys_prompt)
        return result
    elif mode == "random":
        return _pair_random(singles, n_target, sys_prompt)
    elif mode == "confused":
        if confusion_groups is None or rel_to_group is None:
            raise ValueError("confused mode requires confusion_groups and rel_to_group")
        return _pair_confused(singles, n_target, confusion_groups, rel_to_group, sys_prompt)
    else:
        raise ValueError(f"Unknown pair mode: {mode}")


# ── Replay buffer augmentation ──────────────────────────────────────────

def replay_augment(data, buffer, mode, sys_prompt, rel_to_group=None):
    random.shuffle(buffer)

    if mode == "random":
        mid = len(buffer) // 2
        replay_data = []
        shuffled_data = random.sample(data, min(len(data), mid))
        for i, sample in enumerate(shuffled_data):
            replay_data.append(_make_pair(sample, buffer[i], sys_prompt))
        return buffer[mid:] + replay_data

    if mode == "confused" and rel_to_group is not None:
        # Phase 1: pair buffer samples with current-task samples from SAME confusion group
        current_by_group = {}
        shuffled_data = list(data)
        random.shuffle(shuffled_data)
        for d in shuffled_data:
            rels = json.loads(d["response"]).get("relations", [])
            if rels:
                gid = rel_to_group.get(rels[0][2], -1)
                if gid >= 0:
                    current_by_group.setdefault(gid, []).append(d)

        replay_data = []
        unmatched_buffer = []
        for b_sample in buffer:
            b_rels = json.loads(b_sample["response"]).get("relations", [])
            if not b_rels:
                unmatched_buffer.append(b_sample)
                continue
            b_type = b_rels[0][2]
            b_gid = rel_to_group.get(b_type, -1)
            # Pair with current-task sample from same confusion group
            if b_gid >= 0 and b_gid in current_by_group and current_by_group[b_gid]:
                cur_sample = current_by_group[b_gid].pop()
                cur_type = json.loads(cur_sample["response"])["relations"][0][2]
                if cur_type != b_type:
                    replay_data.append(_make_pair(cur_sample, b_sample, sys_prompt))
                else:
                    unmatched_buffer.append(b_sample)
                    current_by_group[b_gid].append(cur_sample)
            else:
                unmatched_buffer.append(b_sample)

        # Phase 2: fallback — pair unmatched buffer with random current-task data
        mid = len(unmatched_buffer) // 2
        random_current = random.sample(data, min(len(data), mid))
        for i, cur_sample in enumerate(random_current):
            if i < len(unmatched_buffer):
                replay_data.append(_make_pair(cur_sample, unmatched_buffer[i], sys_prompt))
        direct_replay = unmatched_buffer[mid:] if mid < len(unmatched_buffer) else []

        return direct_replay + replay_data

    # Default: prefix-based pairing
    current_by_prefix = {}
    shuffled_data = list(data)
    random.shuffle(shuffled_data)
    for d in shuffled_data:
        rels = json.loads(d["response"]).get("relations", [])
        if rels:
            prefix = get_prefix(rels[0][2])
            current_by_prefix.setdefault(prefix, []).append(d)

    replay_data = []
    direct_replay = []
    for b_sample in buffer:
        b_rels = json.loads(b_sample["response"]).get("relations", [])
        if not b_rels:
            direct_replay.append(b_sample)
            continue
        b_prefix = get_prefix(b_rels[0][2])
        if b_prefix in current_by_prefix and current_by_prefix[b_prefix]:
            cur_sample = current_by_prefix[b_prefix].pop()
            replay_data.append(_make_pair(cur_sample, b_sample, sys_prompt))
        else:
            direct_replay.append(b_sample)

    return direct_replay + replay_data


# ── Data processing ─────────────────────────────────────────────────────

def process_re_data(raw_data, rel_list, buffer, is_test=False, tasks=[], eval_tasks=[],
                    pair_mode="same_prefix", sys_prompt="", use_marker=True,
                    buffer_size=10, replay_upsample=1,
                    confusion_groups=None, rel_to_group=None):
    """
    raw_data: dict {relation_type: [samples]}
    rel_list: list of all relation type names
    tasks: list of relation indices for current task
    eval_tasks: cumulative list of relation indices for evaluation
    """
    data = []
    temp_buffer = {}

    active_rels = [rel_list[i] for i in tasks]
    eval_rels = [rel_list[i] for i in eval_tasks] if eval_tasks else active_rels

    for rel_name in (eval_rels if is_test else active_rels):
        if rel_name not in raw_data:
            continue

        if rel_name not in temp_buffer:
            temp_buffer[rel_name] = []

        samples = raw_data[rel_name]
        for sample in samples:
            tokens = sample["tokens"]
            sent = " ".join(tokens)

            if use_marker:
                e1 = extract_entity(tokens, "[E11]", "[E12]")
                e2 = extract_entity(tokens, "[E21]", "[E22]")
            else:
                # Without marker — no entity boundaries in text
                # For without_marker, entities aren't explicitly identifiable
                # We use the relation label directly (model must learn from context)
                e1 = extract_entity(tokens, "[E11]", "[E12]") if "[E11]" in tokens else ""
                e2 = extract_entity(tokens, "[E21]", "[E22]") if "[E21]" in tokens else ""

            if not e1 or not e2:
                continue

            response = json.dumps({"relations": [[e1, e2, rel_name]]})
            entry = {
                "system_prompt": sys_prompt,
                "user_prompt": user_prompt_template.format(text=sent),
                "sent": sent,
                "response": response,
                "t_system_prompt": sys_prompt,
                "t_user_prompt": t_user_prompt_template.format(text=sent, result=response),
            }
            data.append(entry)

            if len(temp_buffer[rel_name]) < buffer_size:
                temp_buffer[rel_name].append(entry)

    if not is_test:
        random.seed(42)

        # 1. Replay buffer (with optional upsampling)
        if buffer:
            replay_samples = replay_augment(data, buffer, mode=pair_mode, sys_prompt=sys_prompt,
                                           rel_to_group=rel_to_group)
            # Upsample replay to reduce imbalance with current task data
            for _ in range(replay_upsample):
                data.extend(replay_samples)

        # 2. Multi-relation augmentation
        single = [d for d in data if len(json.loads(d["response"]).get("relations", [])) == 1]
        random.shuffle(single)
        n_target = len(single) // 4
        if n_target > 0:
            aug_data = pair_samples(single, n_target, mode=pair_mode, sys_prompt=sys_prompt,
                                   confusion_groups=confusion_groups, rel_to_group=rel_to_group)
            data.extend(aug_data)

        # Update buffer
        for k, v in temp_buffer.items():
            buffer.extend(v)

    return data


# ── Train/dev/test split ────────────────────────────────────────────────

def split_data(raw_data, train_ratio=0.8, dev_ratio=0.1, seed=42):
    """Split per-relation data into train/dev/test."""
    rng = random.Random(seed)
    train, dev, test = {}, {}, {}
    for rel, samples in raw_data.items():
        shuffled = list(samples)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)
        train[rel] = shuffled[:n_train]
        dev[rel] = shuffled[n_train:n_train + n_dev]
        test[rel] = shuffled[n_train + n_dev:]
    return train, dev, test


# ── Save helper ──────────────────────────────────────────────────────────

def save(train_data, dev_data, test_data, rel_list, data_name, tasks, buffer, eval_tasks,
         pair_mode, sys_prompt, use_marker, buffer_size=10, replay_upsample=1,
         confusion_groups=None, rel_to_group=None):
    os.makedirs(f"data/{data_name}", exist_ok=True)

    train = process_re_data(train_data, rel_list, buffer=buffer, tasks=tasks,
                            pair_mode=pair_mode, sys_prompt=sys_prompt, use_marker=use_marker,
                            buffer_size=buffer_size, replay_upsample=replay_upsample,
                            confusion_groups=confusion_groups, rel_to_group=rel_to_group)
    with open(f"data/{data_name}/train.jsonl", "w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    dev = process_re_data(dev_data, rel_list, [], is_test=True, tasks=tasks,
                          eval_tasks=eval_tasks, sys_prompt=sys_prompt, use_marker=use_marker)
    with open(f"data/{data_name}/dev.jsonl", "w", encoding="utf-8") as f:
        for item in dev:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    test = process_re_data(test_data, rel_list, [], is_test=True, tasks=tasks,
                           eval_tasks=eval_tasks, sys_prompt=sys_prompt, use_marker=use_marker)
    with open(f"data/{data_name}/test.jsonl", "w", encoding="utf-8") as f:
        for item in test:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  {data_name}: train={len(train)}, dev={len(dev)}, test={len(test)}")
    return train, dev, test


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate CL Relation Extraction data")
    parser.add_argument("--dataset", type=str, required=True, choices=["fewrel", "tacred"])
    parser.add_argument("--pair-mode", type=str, default="same_prefix",
                        choices=["same_prefix", "cross_prefix", "mixed", "random", "confused"])
    parser.add_argument("--marker", type=str, default="with_marker",
                        choices=["with_marker", "without_marker"])
    parser.add_argument("--num-tasks", type=int, default=10,
                        help="Number of CL tasks (default: 10)")
    parser.add_argument("--num-perms", type=int, default=5,
                        help="Number of random permutations (default: 5)")
    parser.add_argument("--buffer-size", type=int, default=10,
                        help="Replay buffer samples per relation type (default: 10)")
    parser.add_argument("--replay-upsample", type=int, default=1,
                        help="Repeat replay data N times to reduce imbalance (default: 1, no upsample)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load data
    if args.dataset == "fewrel":
        data_file = f"datasets/data_{args.marker}.json"
        rel_file = "datasets/id2rel.json"
        name_file = "datasets/pid2name_fewrel.json"
    else:
        data_file = f"datasets/data_{args.marker}_tacred.json"
        rel_file = "datasets/id2rel_tacred.json"
        name_file = None  # TACRED already uses semantic names

    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    with open(rel_file, "r", encoding="utf-8") as f:
        rel_list = json.load(f)

    # FewRel: map opaque Wikidata IDs to semantic relation names
    id2name = {}
    if name_file and os.path.exists(name_file):
        with open(name_file, "r", encoding="utf-8") as f:
            id2name = json.load(f)
        # Remap raw_data keys and rel_list
        raw_data = {id2name.get(k, k): v for k, v in raw_data.items()}
        # Update relation field inside each sample
        for rel_name, samples in raw_data.items():
            for s in samples:
                s["relation"] = rel_name
        rel_list = [id2name.get(r, r) for r in rel_list]
        print(f"Mapped {len(id2name)} relation IDs to semantic names")

    use_marker = args.marker == "with_marker"
    if not use_marker:
        raise NotImplementedError(
            "without_marker mode is currently unsupported for RE data generation. "
            "Entity extraction requires [E11]/[E12]/[E21]/[E22] markers. "
            "Use --marker with_marker instead."
        )
    sys_prompt = system_prompt_with_marker if use_marker else system_prompt_without_marker

    print(f"Dataset: {args.dataset}, Marker: {args.marker}, Pair mode: {args.pair_mode}")
    print(f"Relations: {len(rel_list)}, Tasks: {args.num_tasks}, Rels/task: {len(rel_list) // args.num_tasks}")

    # Split into train/dev/test
    train_data, dev_data, test_data = split_data(raw_data)

    # Create CL task streams (split relations into num_tasks groups)
    rels_per_task = len(rel_list) // args.num_tasks
    base_stream = list(range(len(rel_list)))

    # Generate permutations
    rng = random.Random(args.seed)
    perms = []
    for _ in range(args.num_perms):
        perm = list(range(args.num_tasks))
        rng.shuffle(perm)
        perms.append(perm)
    # First perm is always sequential
    perms[0] = list(range(args.num_tasks))

    # Task streams: each task group has rels_per_task relations
    task_groups = []
    for t in range(args.num_tasks):
        start = t * rels_per_task
        end = start + rels_per_task
        task_groups.append(base_stream[start:end])

    ds_name = args.dataset
    marker_suffix = "wm" if use_marker else "nm"

    # Build confusion groups for confused mode
    confusion_groups, rel_to_group = None, None
    if args.pair_mode == "confused":
        rel_name_groups = [[rel_list[i] for i in g] for g in task_groups]
        confusion_groups, rel_to_group = build_confusion_groups(rel_list, rel_name_groups)
        print(f"\nAuto-detected {len(confusion_groups)} confusion groups:")
        for i, g in enumerate(sorted(confusion_groups, key=lambda x: -len(x))):
            print(f"  Group {i+1} ({len(g)} types): {sorted(g)[:4]}{'...' if len(g) > 4 else ''}")

    for perm_idx, perm in enumerate(perms):
        print(f"\nPermutation {perm_idx}: {perm}")
        buffer = []
        eval_tasks = []
        for p_id, task_idx in enumerate(perm):
            tasks = task_groups[task_idx]
            eval_tasks.extend(tasks)
            up_suffix = f"_up{args.replay_upsample}" if args.replay_upsample > 1 else ""
            data_name = f"{ds_name}_{marker_suffix}_re_{args.pair_mode}{up_suffix}_{perm_idx}/{p_id}"
            save(train_data, dev_data, test_data, rel_list, data_name,
                 tasks, buffer, eval_tasks, args.pair_mode, sys_prompt, use_marker,
                 buffer_size=args.buffer_size, replay_upsample=args.replay_upsample,
                 confusion_groups=confusion_groups, rel_to_group=rel_to_group)


if __name__ == "__main__":
    main()
