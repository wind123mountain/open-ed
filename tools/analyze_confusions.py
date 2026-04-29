#!/usr/bin/env python3
"""Auto-detect confusion clusters from CRE model predictions.

Usage:
    python tools/analyze_confusions.py \
        --answers results/qwen3/re/tacred_random_kd05_greedy_0_0.6B_cl/9/eval/2/answers.jsonl \
        --golds processed_data/tacred_wm_re_random_0/9/qwen/test.jsonl \
        [--threshold 3] [--show-examples]
"""

import json
import argparse
from collections import Counter, defaultdict


def extract_confusions(answers_path, golds_path):
    with open(answers_path) as f:
        preds = [json.loads(l) for l in f]
    with open(golds_path) as f:
        golds = [json.loads(l) for l in f]

    confusions = Counter()       # (gold_type, pred_type) → count
    type_stats = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
    examples = defaultdict(list)  # (gold_type, pred_type) → [(sent, e1, e2)]

    for pred, gold in zip(preds, golds):
        try:
            pred_rels = json.loads(pred["text"]).get("relations", [])
        except Exception:
            pred_rels = []
        gold_rels = json.loads(gold["response"]).get("relations", [])

        sent = gold.get("prompt", "")
        # Extract text between "Input text:" and end of prompt
        if "Input text:" in sent:
            sent = sent.split("Input text:")[-1].split("<|im_end|>")[0].strip()

        for gr in gold_rels:
            ge1, ge2, gt = gr[0].lower(), gr[1].lower(), gr[2].lower()
            matched = False
            for pr in pred_rels:
                pe1, pe2, pt = pr[0].lower(), pr[1].lower(), pr[2].lower()
                if pe1 == ge1 and pe2 == ge2:
                    if pt == gt:
                        type_stats[gt]["tp"] += 1
                    else:
                        confusions[(gt, pt)] += 1
                        type_stats[gt]["fn"] += 1
                        type_stats[pt]["fp"] += 1
                        examples[(gt, pt)].append((sent[:300], ge1, ge2))
                    matched = True
                    break
            if not matched:
                type_stats[gt]["fn"] += 1

    return confusions, type_stats, examples


def find_clusters(confusions, threshold=3):
    """Find connected components in confusion graph."""
    graph = defaultdict(set)
    for (g, p), count in confusions.items():
        if count >= threshold:
            graph[g].add(p)
            graph[p].add(g)

    visited = set()
    clusters = []

    def dfs(node, cluster):
        visited.add(node)
        cluster.add(node)
        for nb in graph[node]:
            if nb not in visited:
                dfs(nb, cluster)

    for node in graph:
        if node not in visited:
            cluster = set()
            dfs(node, cluster)
            clusters.append(cluster)

    return clusters


def compute_f1(s):
    tot = 2 * s["tp"] + s["fp"] + s["fn"]
    return 2 * s["tp"] / tot * 100 if tot > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Analyze CRE confusion clusters")
    parser.add_argument("--answers", required=True, help="Path to answers.jsonl")
    parser.add_argument("--golds", required=True, help="Path to test.jsonl (processed)")
    parser.add_argument("--threshold", type=int, default=3, help="Min confusions to form edge (default: 3)")
    parser.add_argument("--show-examples", action="store_true", help="Show example sentences for confusions")
    args = parser.parse_args()

    confusions, stats, examples = extract_confusions(args.answers, args.golds)
    clusters = find_clusters(confusions, threshold=args.threshold)

    total_confusions = sum(confusions.values())
    total_tp = sum(s["tp"] for s in stats.values())
    total_fp = sum(s["fp"] for s in stats.values())
    total_fn = sum(s["fn"] for s in stats.values())
    current_f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn) * 100

    print(f"Overall: TP={total_tp} FP={total_fp} FN={total_fn} F1={current_f1:.1f}%")
    print(f"Total type confusions (entity correct, type wrong): {total_confusions}")
    print()

    # Score and sort clusters
    cluster_scores = []
    for cluster in clusters:
        total_conf = sum(c for (g, p), c in confusions.items() if g in cluster and p in cluster)
        pairs = [(g, p, c) for (g, p), c in confusions.items() if g in cluster and p in cluster]
        pairs.sort(key=lambda x: -x[2])
        cluster_scores.append((total_conf, cluster, pairs))
    cluster_scores.sort(key=lambda x: -x[0])

    accounted = 0
    print("=" * 80)
    print("CONFUSION CLUSTERS")
    print("=" * 80)

    for rank, (total_conf, cluster, pairs) in enumerate(cluster_scores):
        accounted += total_conf
        print(f"\nCluster {rank+1}: {total_conf} confusions ({total_conf/total_confusions*100:.0f}% of all)")
        print("-" * 70)

        for t in sorted(cluster):
            s = stats[t]
            f1 = compute_f1(s)
            print(f"  {t:50s} F1={f1:5.1f}%  tp={s['tp']:3d} fp={s['fp']:3d} fn={s['fn']:3d}")

        print(f"\n  Confusions:")
        for g, p, c in pairs[:8]:
            print(f"    {g:40s} → {p:30s} ({c}x)")

            if args.show_examples and (g, p) in examples:
                for sent, e1, e2 in examples[(g, p)][:2]:
                    print(f"      E1={e1}, E2={e2}")
                    print(f"      \"{sent[:150]}\"")

    # Estimate potential improvement
    new_tp = total_tp + accounted
    new_fp = total_fp - accounted
    new_fn = total_fn - accounted
    potential_f1 = 2 * new_tp / (2 * new_tp + max(new_fp, 0) + max(new_fn, 0)) * 100

    print()
    print("=" * 80)
    print(f"Clusters explain {accounted}/{total_confusions} confusions ({accounted/total_confusions*100:.0f}%)")
    print(f"Current F1: {current_f1:.1f}%")
    print(f"If all cluster confusions fixed: F1 ≈ {potential_f1:.1f}% (+{potential_f1 - current_f1:.1f})")

    # Output cluster types as JSON for downstream use
    cluster_json = []
    for total_conf, cluster, pairs in cluster_scores:
        cluster_json.append({
            "types": sorted(cluster),
            "confusions": total_conf,
            "top_pairs": [(g, p, c) for g, p, c in pairs[:5]]
        })
    print()
    print("=== CLUSTER JSON (for programmatic use) ===")
    print(json.dumps(cluster_json, indent=2))


if __name__ == "__main__":
    main()
