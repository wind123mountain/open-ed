"""Build CED task splits for GENEVA / MAVEN with SharpSeq permutations.

Same pipeline as tools/build_ced_perms.py (ACE), with two differences:
- streams come from data/ext-data/{ds}/streams.json + label2id.json instead of
  being reconstructed from an existing perm-0 split (id 0 = None appears in
  every stream and is dropped)
- MAVEN has no argument annotation, so its event_info is
  [trigger, type, description]; GENEVA matches ACE: [trigger, type, args, description]

Usage (on server, mta env):
    python tools/build_ced_perms_v2.py geneva
    python tools/build_ced_perms_v2.py maven
"""
import json
import os
import random
import re
import sys

from datasets import load_dataset

BASE = "/home/hungpv/projects/OpenED"
PERM = [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [0, 3, 1, 4, 2], [1, 2, 0, 3, 4], [3, 4, 0, 1, 2]]
SEED = 42

DATASETS = {
    "geneva": {"src": "datht/geneva-short-generated-dataset", "with_args": True},
    "maven": {"src": "datht/maven-short-generated-dataset", "with_args": False},
}


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


# ---------- 1. streams from ext-data (label ids -> type names) ----------
def load_streams(ds_name):
    with open(f"{BASE}/data/ext-data/{ds_name}/streams.json") as f:
        raw = json.load(f)
    with open(f"{BASE}/data/ext-data/{ds_name}/label2id.json") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    streams = []
    for st in raw:
        names = sorted(id2label[i] for i in st if i in id2label)  # drops id 0 (None)
        streams.append(names)
    total = sum(len(s) for s in streams)
    assert total == len(label2id), f"expected {len(label2id)} types, got {total}"
    return streams


# ---------- 2. extract prompt templates from existing flat data ----------
def extract_templates(ds_name):
    recs = load_jsonl(f"{BASE}/data/{ds_name}/train.jsonl")
    system_prompt = recs[0]["system_prompt"]
    assert all(r["system_prompt"] == system_prompt for r in recs)

    up = recs[0]["user_prompt"]
    m = re.match(r"(Given an input text: \n<input>\n)(.*?)(\n</input>\n\nYour task.*)$", up, re.S)
    assert m, "unexpected user_prompt format"
    prefix, _, suffix = m.groups()
    template = prefix + "{input}" + suffix
    pat = re.compile(re.escape(prefix) + r"(.*?)" + re.escape(suffix) + r"$", re.S)
    for r in recs[:200]:
        mm = pat.match(r["user_prompt"])
        assert mm and template.format(input=mm.group(1)) == r["user_prompt"]
    return system_prompt, template


# ---------- 3. ACE CED logic, verbatim (event_info shape per dataset) ----------
def process_data(raw_data, buffer, system_prompt, user_template, with_args,
                 is_test=False, tasks=(), eval_tasks=()):
    tasks = set(tasks)
    eval_tasks = set(eval_tasks)
    data = []
    none_data = []

    for idx, sample in enumerate(raw_data):
        sent_id_to_sentence = {i: c["sentence"] for i, c in enumerate(sample["content"])}
        sent_id_set = set(sent_id_to_sentence.keys())
        sent_to_existing_events = {}
        b_sent_id_map = {}

        for event in sample.get("events", []):
            if event.get("type_id", -1) == -1:
                continue
            event_type = event.get("type")
            description = event.get("description", "")

            if event_type not in tasks:
                if not (is_test and event_type in eval_tasks):
                    continue

            if event_type not in process_data.temp_buffer:
                process_data.temp_buffer[event_type] = []

            for mention in event.get("mention", []):
                sent_id = mention.get("sent_id")
                if sent_id not in sent_to_existing_events:
                    sent_to_existing_events[sent_id] = []
                if with_args:
                    args = [[a["text"], a["role"]] for a in mention.get("arguments", [])]
                    event_info = [mention.get("trigger_word"), event_type, args, description]
                else:
                    event_info = [mention.get("trigger_word"), event_type, description]
                sent_to_existing_events[sent_id].append(event_info)

                if len(process_data.temp_buffer[event_type]) < 5:
                    b_sent_id_map[f"{idx}_{sent_id}"] = event_type

        for sent_id, evs in sent_to_existing_events.items():
            sent_txt = sent_id_to_sentence[sent_id]
            sent_id_set.remove(sent_id)
            response = json.dumps({"events": evs})
            data.append({"system_prompt": system_prompt,
                         "user_prompt": user_template.format(input=sent_txt),
                         "response": response})
            key = f"{idx}_{sent_id}"
            if key in b_sent_id_map:
                process_data.temp_buffer[b_sent_id_map[key]].append(data[-1])

        for sent_id in sent_id_set:
            sent_txt = sent_id_to_sentence[sent_id]
            none_data.append({"system_prompt": system_prompt,
                              "user_prompt": user_template.format(input=sent_txt),
                              "response": json.dumps({"events": []})})

    if not is_test:
        data.extend(random.sample(list(none_data), min(len(none_data), len(data) // 10)))
        data.extend(buffer)
        for v in process_data.temp_buffer.values():
            buffer.extend(v)
    return data


def save_task(ds, out_dir, tasks, buffer, eval_tasks, system_prompt, user_template, with_args):
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    process_data.temp_buffer = {}
    train = process_data(ds["train"], buffer, system_prompt, user_template, with_args, tasks=tasks)
    process_data.temp_buffer = {}
    dev = process_data(ds["validation"], [], system_prompt, user_template, with_args,
                       is_test=True, tasks=tasks, eval_tasks=eval_tasks)
    process_data.temp_buffer = {}
    test = process_data(ds["test"], [], system_prompt, user_template, with_args,
                        is_test=True, tasks=tasks, eval_tasks=eval_tasks)
    for name, rows in [("train", train), ("dev", dev), ("test", test)]:
        with open(f"{out_dir}/{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[name] = len(rows)
    return stats


def main():
    ds_name = sys.argv[1]
    cfg = DATASETS[ds_name]

    streams = load_streams(ds_name)
    print(f"{ds_name} streams (type counts):", [len(s) for s in streams])

    system_prompt, user_template = extract_templates(ds_name)
    print("prompt templates extracted OK")

    ds = load_dataset(cfg["src"])
    print("splits:", {k: len(v) for k, v in ds.items()})

    for p, order in enumerate(PERM):
        random.seed(SEED + p)
        perm_streams = [streams[t] for t in order]
        buffer = []
        eval_tasks = []
        print(f"\n=== perm {p} (stream order {order}) ===")
        for t, tasks in enumerate(perm_streams):
            eval_tasks.extend(tasks)
            out_dir = f"{BASE}/data/{ds_name}_perm{p}/{t}"
            stats = save_task(ds, out_dir, tasks, buffer, list(eval_tasks),
                              system_prompt, user_template, cfg["with_args"])
            print(f"task {t}: streams[{order[t]}] ({len(tasks)} types) "
                  f"train={stats['train']} dev={stats['dev']} test={stats['test']} "
                  f"buffer_after={len(buffer)}", flush=True)


if __name__ == "__main__":
    main()
