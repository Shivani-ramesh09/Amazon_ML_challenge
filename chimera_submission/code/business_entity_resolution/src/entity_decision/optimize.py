"""Coarse-to-fine, vectorized threshold search on cached held-out S1 groups."""

from __future__ import annotations

from dataclasses import replace
import itertools
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from chimera_submission.code.business_entity_resolution.src.entity_decision.policy import DecisionPolicy


def load_group_arrays(group_dir: Path, truth: dict[str, set[str]], *, width: int = 30) -> dict[str, np.ndarray]:
    ids = []
    scores_rows = []
    truth_rows = []
    gt_counts = []
    for path in sorted(group_dir.glob("part-*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=10_000,
                                                       columns=["source1_entity_id", "candidate_ids", "scores"]):
            s1_values, candidates_values, scores_values = [batch.column(i).to_pylist() for i in range(3)]
            for s1_id, candidates, scores in zip(s1_values, candidates_values, scores_values):
                if s1_id not in truth:
                    raise ValueError(f"group S1 {s1_id} absent from truth")
                if len(candidates) > width:
                    raise ValueError(f"candidate list exceeds matrix width {width}")
                ids.append(s1_id)
                scores_rows.append(scores)
                truth_rows.append([candidate in truth[s1_id] for candidate in candidates])
                gt_counts.append(len(truth[s1_id]))
    if len(ids) != len(truth):
        raise ValueError("group S1 count differs from validation truth")
    n = len(ids)
    score_matrix = np.full((n, width), -1.0, dtype=np.float32)
    truth_matrix = np.zeros((n, width), dtype=bool)
    for index, (scores, mask) in enumerate(zip(scores_rows, truth_rows)):
        score_matrix[index, :len(scores)] = scores
        truth_matrix[index, :len(mask)] = mask
    return {"scores": score_matrix, "truth_mask": truth_matrix,
            "truth_counts": np.asarray(gt_counts, dtype=np.int16), "s1_ids": np.asarray(ids)}


def evaluate_policy(data: dict[str, np.ndarray], policy: DecisionPolicy) -> dict:
    scores = data["scores"]
    truth = data["truth_mask"]
    gt_counts = data["truth_counts"]
    top = scores[:, 0]
    second = scores[:, 1] if scores.shape[1] > 1 else np.full(len(top), -1.0)
    eligible = top >= policy.singleton_threshold
    strong = eligible & (top >= policy.strong_match_threshold) & (
        top - np.maximum(second, 0) >= policy.gap_threshold)
    selected = (scores >= policy.pair_threshold) & eligible[:, None]
    selected[:, policy.max_match_count:] = False
    selected[strong, :] = False
    selected[strong, 0] = True
    predicted_counts = selected.sum(axis=1)
    tp = np.logical_and(selected, truth).sum(axis=1)
    fp = predicted_counts - tp
    fn = gt_counts - tp
    denominator = 1.25 * tp + fp + 0.25 * fn
    f = np.divide(1.25 * tp, denominator, out=np.zeros(len(tp), dtype=np.float64),
                  where=denominator > 0)
    f[(gt_counts == 0) & (predicted_counts == 0)] = 1.0
    singleton = gt_counts == 0
    counts = {"entities": len(tp), "tp": int(tp.sum()), "fp": int(fp.sum()),
              "fn": int(fn.sum()), "true_singletons": int(singleton.sum()),
              "correct_singletons": int(np.sum(singleton & (predicted_counts == 0))),
              "false_positive_singletons": int(np.sum(singleton & (predicted_counts > 0))),
              "predicted_zero": int(np.sum(predicted_counts == 0)),
              "predicted_one": int(np.sum(predicted_counts == 1)),
              "predicted_many": int(np.sum(predicted_counts > 1))}
    return {"macro_f0_5": float(f.mean()),
            "micro_precision": counts["tp"] / (counts["tp"] + counts["fp"])
            if counts["tp"] + counts["fp"] else None,
            "micro_recall": counts["tp"] / (counts["tp"] + counts["fn"])
            if counts["tp"] + counts["fn"] else None,
            "singleton_accuracy": counts["correct_singletons"] / counts["true_singletons"]
            if counts["true_singletons"] else None,
            "singleton_false_positive_rate": counts["false_positive_singletons"] / counts["true_singletons"]
            if counts["true_singletons"] else None,
            "counts": counts}


def _valid(policy: DecisionPolicy) -> bool:
    return policy.pair_threshold <= policy.singleton_threshold <= policy.strong_match_threshold


def optimize_policy(data: dict[str, np.ndarray]) -> dict:
    baseline = DecisionPolicy()
    baseline_metrics = evaluate_policy(data, baseline)
    coarse = []
    seen = set()

    def record(policy: DecisionPolicy, sink: list):
        key = tuple(policy.as_dict().values())
        if key in seen or not _valid(policy):
            return
        seen.add(key)
        metric = evaluate_policy(data, policy)
        sink.append({"policy": policy.as_dict(), "macro_f0_5": metric["macro_f0_5"]})

    for singleton, pair, strong, gap, max_count in itertools.product(
        (0.3, 0.5, 0.7, 0.9), (0.2, 0.4, 0.6, 0.8),
        (0.9, 0.97, 0.995), (0.1, 0.25, 0.4), (5, 30)
    ):
        record(DecisionPolicy(singleton, strong, pair, gap, max_count), coarse)
    coarse.sort(key=lambda row: row["macro_f0_5"], reverse=True)
    fine = []
    best = DecisionPolicy(**coarse[0]["policy"])
    # Coordinate refinements keep the search small and measurable.
    for _ in range(2):
        for field, step in (("singleton_threshold", 0.05), ("pair_threshold", 0.05),
                            ("strong_match_threshold", 0.025), ("gap_threshold", 0.05)):
            current = getattr(best, field)
            for offset in (-2, -1, 0, 1, 2):
                value = min(1.0, max(0.0, round(current + offset * step, 4)))
                record(replace(best, **{field: value}), fine)
            all_evaluated = sorted(coarse + fine, key=lambda row: row["macro_f0_5"], reverse=True)
            best = DecisionPolicy(**all_evaluated[0]["policy"])
    combined = sorted(coarse + fine, key=lambda row: row["macro_f0_5"], reverse=True)
    top_score = combined[0]["macro_f0_5"]
    stability = []
    for entry in combined[:10]:
        policy = DecisionPolicy(**entry["policy"])
        nearby = []
        for field, offset in (("singleton_threshold", -0.02), ("singleton_threshold", 0.02),
                              ("pair_threshold", -0.02), ("pair_threshold", 0.02),
                              ("strong_match_threshold", -0.02), ("strong_match_threshold", 0.02),
                              ("gap_threshold", -0.02), ("gap_threshold", 0.02)):
            value = min(1.0, max(0.0, round(getattr(policy, field) + offset, 4)))
            neighbor = replace(policy, **{field: value})
            if _valid(neighbor):
                nearby.append(evaluate_policy(data, neighbor)["macro_f0_5"])
        stability.append({**entry, "neighbor_min": min(nearby) if nearby else entry["macro_f0_5"],
                          "neighbor_mean": float(np.mean(nearby)) if nearby else entry["macro_f0_5"],
                          "neighbor_max": max(nearby) if nearby else entry["macro_f0_5"]})
    stable_pool = [row for row in stability if row["macro_f0_5"] >= top_score - 0.0005]
    selected = max(stable_pool, key=lambda row: (row["neighbor_mean"], row["neighbor_min"]))
    selected_policy = DecisionPolicy(**selected["policy"])
    return {"baseline_policy": baseline.as_dict(), "baseline_metrics": baseline_metrics,
            "selected_policy": selected_policy.as_dict(),
            "selected_metrics": evaluate_policy(data, selected_policy),
            "coarse_trials": len(coarse), "fine_trials": len(fine),
            "coarse_top_10": coarse[:10], "stability_top_10": stability,
            "best_raw_macro_f0_5": top_score}
