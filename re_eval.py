"""
Relation Extraction evaluation metrics.

Mirrors ed_eval.py but for relation extraction:
  - relation F1: match on (entity1, entity2, relation_type) triples
  - relation_text F1: match on (entity1, entity2) pairs only (ignoring type)
  - relation_cls_acc: classification accuracy conditioned on correct entity extraction
  - per-type F1 breakdown
"""
import json
from collections import Counter, defaultdict


def normalize(text):
    return text.strip().lower()


def safe_load(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except:
            return {"relations": []}
    return x


def extract_relations(data):
    """Extract (entity1, entity2, relation_type) triples and (entity1, entity2) pairs."""
    relations = []
    relation_texts = []
    try:
        for rel in data.get("relations", []):
            e1 = normalize(rel[0])
            e2 = normalize(rel[1])
            rtype = normalize(rel[2])
            relations.append((e1, e2, rtype))
            relation_texts.append((e1, e2))
    except:
        pass
    return list(set(relations)), list(set(relation_texts))


def update_counts(pred_items, gt_items, counts):
    pred_counter = Counter(pred_items)
    gt_counter = Counter(gt_items)

    tp = sum((pred_counter & gt_counter).values())
    fp = sum((pred_counter - gt_counter).values())
    fn = sum((gt_counter - pred_counter).values())

    counts["tp"] += tp
    counts["fp"] += fp
    counts["fn"] += fn


def compute_f1(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision > 0 and recall > 0 else 0

    return precision, recall, f1


def re_evaluate(pred_list, gt_list):
    relation_counts = {"tp": 0, "fp": 0, "fn": 0}
    relation_text_counts = {"tp": 0, "fp": 0, "fn": 0}
    relation_type_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    global_gt_types = set()

    for pred_json, gt_json in zip(pred_list, gt_list):
        pred = safe_load(pred_json)
        gt = safe_load(gt_json[0])

        pred_rels, pred_rel_texts = extract_relations(pred)
        gt_rels, gt_rel_texts = extract_relations(gt)

        for r in gt_rels:
            global_gt_types.add(r[2])

        update_counts(pred_rels, gt_rels, relation_counts)
        update_counts(pred_rel_texts, gt_rel_texts, relation_text_counts)

        all_types_in_doc = set([r[2] for r in pred_rels] + [r[2] for r in gt_rels])
        for rtype in all_types_in_doc:
            pred_by_type = [r for r in pred_rels if r[2] == rtype]
            gt_by_type = [r for r in gt_rels if r[2] == rtype]
            update_counts(pred_by_type, gt_by_type, relation_type_counts[rtype])

    relation_metrics = compute_f1(relation_counts)
    relation_text_metrics = compute_f1(relation_text_counts)

    relation_per_type_metrics = {}
    for rtype, counts in relation_type_counts.items():
        if rtype in global_gt_types:
            precision, recall, f1 = compute_f1(counts)
            relation_per_type_metrics[rtype] = {
                "counts": dict(counts),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    cls_acc = (relation_counts["tp"] / relation_text_counts["tp"]
               if relation_text_counts["tp"] > 0 else 0)

    return {
        "relation_counts": relation_counts,
        "relation_text": {
            "precision": relation_text_metrics[0],
            "recall": relation_text_metrics[1],
            "f1": relation_text_metrics[2],
        },
        "relation": {
            "precision": relation_metrics[0],
            "recall": relation_metrics[1],
            "f1": relation_metrics[2],
        },
        "relation_cls_acc": cls_acc,
        "relation_per_type": relation_per_type_metrics,
    }
