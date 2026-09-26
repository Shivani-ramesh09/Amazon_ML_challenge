"""Exact competition macro F0.5 and entity-level diagnostics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from chimera_submission.code.business_entity_resolution.src.entity_decision.policy import DecisionPolicy


def entity_f0_5(truth: set[str], predicted: set[str]) -> float:
    """Official per-S1 set score, including empty/empty = 1."""
    if not truth and not predicted:
        return 1.0
    tp = len(truth & predicted)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    denominator = 1.25 * tp + fp + 0.25 * fn
    return 1.25 * tp / denominator if denominator else 0.0


def _class(size: int) -> str:
    return "zero" if size == 0 else "one" if size == 1 else "many"


def evaluate_groups(group_dir: Path, truth: dict[str, set[str]], policy: DecisionPolicy) -> dict:
    counts = Counter()
    confusion = Counter()
    country = {}
    f_sum = 0.0
    f_positive = 0.0
    for path in sorted(group_dir.glob("part-*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=10_000):
            columns = {name: batch.column(batch.schema.get_field_index(name)).to_pylist()
                       for name in ("source1_entity_id", "candidate_ids", "scores", "country")}
            for s1_id, candidates, scores, origin in zip(*(columns[name] for name in columns)):
                if s1_id not in truth:
                    raise ValueError(f"evaluation S1 absent from truth: {s1_id}")
                gt = truth[s1_id]
                prediction = set(policy.decide(candidates, scores))
                tp = len(gt & prediction)
                fp = len(prediction - gt)
                fn = len(gt - prediction)
                f_value = entity_f0_5(gt, prediction)
                f_sum += f_value
                counts["entities"] += 1
                counts["tp"] += tp
                counts["fp"] += fp
                counts["fn"] += fn
                counts["truth_pairs"] += len(gt)
                counts["predicted_pairs"] += len(prediction)
                counts["candidate_pairs"] += len(candidates)
                counts["true_singletons"] += not bool(gt)
                counts["correct_singletons"] += not gt and not prediction
                counts["false_positive_singletons"] += not gt and bool(prediction)
                counts["positive_entities"] += bool(gt)
                counts["positive_any_hit"] += bool(gt and tp)
                if gt:
                    f_positive += f_value
                confusion[(_class(len(gt)), _class(len(prediction)))] += 1
                label = origin or "<missing>"
                if label not in country:
                    country[label] = {"entities": 0, "f_sum": 0.0, "tp": 0, "fp": 0, "fn": 0}
                country[label]["entities"] += 1
                country[label]["f_sum"] += f_value
                country[label]["tp"] += tp
                country[label]["fp"] += fp
                country[label]["fn"] += fn
    if counts["entities"] != len(truth):
        raise ValueError(f"group table has {counts['entities']} entities; truth has {len(truth)}")
    precision_denominator = counts["tp"] + counts["fp"]
    recall_denominator = counts["tp"] + counts["fn"]
    return {
        "macro_f0_5": f_sum / counts["entities"] if counts["entities"] else None,
        "positive_entity_macro_f0_5": f_positive / counts["positive_entities"] if counts["positive_entities"] else None,
        "micro_precision": counts["tp"] / precision_denominator if precision_denominator else None,
        "micro_recall": counts["tp"] / recall_denominator if recall_denominator else None,
        "singleton_accuracy": counts["correct_singletons"] / counts["true_singletons"] if counts["true_singletons"] else None,
        "singleton_false_positive_rate": counts["false_positive_singletons"] / counts["true_singletons"] if counts["true_singletons"] else None,
        "positive_any_hit_rate": counts["positive_any_hit"] / counts["positive_entities"] if counts["positive_entities"] else None,
        "counts": dict(counts),
        "cardinality_confusion": {f"true_{actual}_pred_{predicted}": count
                                   for (actual, predicted), count in sorted(confusion.items())},
        "country": {name: {"entities": values["entities"],
                           "macro_f0_5": values["f_sum"] / values["entities"],
                           "micro_precision": values["tp"] / (values["tp"] + values["fp"])
                           if values["tp"] + values["fp"] else None,
                           "micro_recall": values["tp"] / (values["tp"] + values["fn"])
                           if values["tp"] + values["fn"] else None}
                    for name, values in sorted(country.items())},
    }
