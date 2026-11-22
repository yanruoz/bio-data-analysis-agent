"""Deterministic challenge triggers on profile.json.

Each rule reads profile.json and either fires (with evidence: JSON paths into
profile.json, their values, and the threshold) or records that it was checked
and did not fire. The agent then writes challenges.json, which must account for
every fired rule (scripts/validate_challenges.py checks this).

Evidence paths are lists of keys, because keys such as 'batch:replicate' or
column names may contain dots.

Usage: venv/bin/python scripts/challenge_rules.py --dataset <name>
       -> outputs/<name>/challenge_triggers.json
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Thresholds (fixed in advance; the agent cannot change them).
T = {
    "high_dim_features": 1000,
    "high_dim_ratio": 0.5,
    "sparse_fraction": 0.5,
    "batch_r2": 0.02,
    "batch_r2_high": 0.05,
    "batch_knn_enrichment": 1.2,
    "confounding_v": 0.3,
    "imbalance_ratio": 10,
    "imbalance_high": 50,
    "structured_missing_spread": 0.5,
    "weak_signal_ratio": 3.0,       # median condition shift below this many times the null q95
    "weak_signal_r2": 0.1,
    "text_label_mention": 0.2,
    "qc_tail": 0.01,
    "targeted_panel_features": 50,
    "low_agreement_ratio": 20,      # kNN overlap below this many times chance
}

TAXONOMY = {
    "high_dimensional": "Many more features than can be modeled directly; needs feature selection or dimension reduction before distance-based methods.",
    "sparse_counts": "Sparse non-negative counts: needs depth normalization and a variance-stabilizing transform; Gaussian assumptions fail on raw values.",
    "batch_effect": "A batch, replicate or split variable explains part of the variation; it can masquerade as biology or hide it.",
    "batch_condition_confounding": "Batch and condition overlap, so their effects cannot be separated.",
    "class_imbalance": "Group sizes differ a lot: accuracy is dominated by large groups; use per-group or macro metrics and stratified splits.",
    "small_groups": "Some groups have too few samples for stable per-group statistics.",
    "missing_values_or_labels": "Some values or labels are missing; methods must handle them and the missingness may not be random.",
    "multimodal_paired": "Several modalities measured on the same samples: decide whether to analyze them separately, compare them, or integrate them.",
    "unstructured_modality": "Image or text inputs need a pretrained encoder (or a learned one) before any numeric analysis.",
    "heterogeneous_inputs": "Inputs vary in size or format and need harmonizing before encoding.",
    "weak_condition_signal": "Condition effects are small compared with variation shared with controls; plain PCA may not show them.",
    "label_input_leakage": "A label is derived from, or named in, one of the inputs; results using that input overstate what the data can predict.",
    "pretraining_overlap": "Pretrained encoders may have seen these samples; their performance can be inflated.",
    "train_test_shift": "The official split differs systematically between train and test.",
    "qc_outliers": "Some samples are extreme in total counts or detected features (possible low-quality samples or multiplets).",
    "targeted_panel": "A modality measures only a few features: per-feature tests fit it better than dimension reduction or learned integration.",
    "low_modality_agreement": "Neighborhoods in one modality barely match those in another; integration may add noise or reveal complementary signal.",
}


def _get(d, path):
    for k in path:
        d = d[k]
    return d


def _ev(profile, path, threshold=None, op=None):
    e = {"path": list(path), "value": _get(profile, path)}
    if threshold is not None:
        e["threshold"] = threshold
        e["comparison"] = op
    return e


def evaluate(p):
    fired, checked = {}, {}

    def fire(cid, severity, evidence):
        if cid in fired:
            fired[cid]["evidence"].extend(evidence)
            if severity == "high":
                fired[cid]["severity"] = "high"
        else:
            fired[cid] = {"id": cid, "severity": severity, "evidence": evidence, "meaning": TAXONOMY[cid]}

    mods = p["modalities"]
    for m, s in mods.items():
        if s.get("structured"):
            if s["n_features"] > T["high_dim_features"] or s["features_per_sample_ratio"] > T["high_dim_ratio"]:
                fire("high_dimensional", "medium", [_ev(p, ["modalities", m, "n_features"], T["high_dim_features"], ">")])
            if s["is_nonneg_integer"] and s["sparsity"] > T["sparse_fraction"]:
                fire("sparse_counts", "medium", [_ev(p, ["modalities", m, "sparsity"], T["sparse_fraction"], ">"),
                                                 _ev(p, ["modalities", m, "is_nonneg_integer"])])
            if s["n_features"] <= T["targeted_panel_features"]:
                fire("targeted_panel", "medium", [_ev(p, ["modalities", m, "n_features"], T["targeted_panel_features"], "<=")])
            if s["missing_values"]["n_missing"] > 0:
                fire("missing_values_or_labels", "medium", [_ev(p, ["modalities", m, "missing_values", "fraction"], 0, ">")])
            for k, v in (s.get("qc") or {}).items():
                if k.endswith("tail_fraction") and v is not None and v >= T["qc_tail"]:
                    fire("qc_outliers", "medium", [_ev(p, ["modalities", m, "qc", k], T["qc_tail"], ">=")])
        else:
            fire("unstructured_modality", "high", [_ev(p, ["modalities", m, "kind"])])
            if s.get("size_heterogeneous"):
                fire("heterogeneous_inputs", "medium", [_ev(p, ["modalities", m, "size_heterogeneous"]),
                                                        _ev(p, ["modalities", m, "color_modes"])])
            for lab, r in (s.get("label_mentions") or {}).items():
                if r["fraction_text_contains_own_label"] >= T["text_label_mention"]:
                    fire("label_input_leakage", "high", [_ev(p, ["modalities", m, "label_mentions", lab, "fraction_text_contains_own_label"],
                                                             T["text_label_mention"], ">=")])
        cond_r2 = None
        for g, r in (s.get("group_structure") or {}).items():
            if r["role"] == "condition":
                cond_r2 = r["variance_explained_by_group"]
        for g, r in (s.get("group_structure") or {}).items():
            base = ["modalities", m, "group_structure", g]
            if r["role"] in ("batch", "batch_candidate", "split"):
                hit = r["variance_explained_by_group"] >= T["batch_r2"] or (r["knn_enrichment"] or 0) >= T["batch_knn_enrichment"]
                cid = "train_test_shift" if r["role"] == "split" else "batch_effect"
                if hit:
                    sev = "high" if (r["variance_explained_by_group"] >= T["batch_r2_high"] or
                                     (cond_r2 is not None and r["variance_explained_by_group"] >= cond_r2 / 2)) else "medium"
                    if r["role"] == "batch_candidate":
                        sev = "medium"
                    fire(cid, sev, [_ev(p, base + ["variance_explained_by_group"], T["batch_r2"], ">="),
                                    _ev(p, base + ["knn_enrichment"], T["batch_knn_enrichment"], ">=")])
                else:
                    checked.setdefault(cid, []).append(_ev(p, base + ["variance_explained_by_group"], T["batch_r2"], "<"))
        cs = s.get("condition_signal")
        if cs:
            weak = ((cs.get("null_shift_q95") is not None and cs["median_shift"] < T["weak_signal_ratio"] * cs["null_shift_q95"])
                    or (cond_r2 is not None and cond_r2 < T["weak_signal_r2"]))
            ev = [_ev(p, ["modalities", m, "condition_signal", "median_shift"]),
                  _ev(p, ["modalities", m, "condition_signal", "null_shift_q95"])]
            if weak:
                fire("weak_condition_signal", "high", ev)
            else:
                checked.setdefault("weak_condition_signal", []).extend(ev)

    for g, s in p["groupings"].items():
        base = ["groupings", g]
        if s["role"] in ("label", "condition"):
            if (s.get("imbalance_ratio") or 0) >= T["imbalance_ratio"]:
                sev = "high" if s["imbalance_ratio"] >= T["imbalance_high"] else "medium"
                fire("class_imbalance", sev, [_ev(p, base + ["imbalance_ratio"], T["imbalance_ratio"], ">=")])
            if s.get("n_small_groups"):
                fire("small_groups", "medium", [_ev(p, base + ["n_small_groups"], 0, ">"), _ev(p, base + ["small_groups"])])
            if s["n_missing"] > 0:
                spread = max((v["spread"] for v in (s.get("missingness_by") or {}).values()), default=0)
                sev = "high" if spread >= T["structured_missing_spread"] else "medium"
                ev = [_ev(p, base + ["missing_fraction"], 0, ">")]
                for o, v in (s.get("missingness_by") or {}).items():
                    if v["spread"] >= T["structured_missing_spread"]:
                        ev.append(_ev(p, base + ["missingness_by", o, "spread"], T["structured_missing_spread"], ">="))
                fire("missing_values_or_labels", sev, ev)

    d = p.get("design")
    if d:
        for b, v in d["batch_confounding_cramers_v"].items():
            path = ["design", "batch_confounding_cramers_v", b]
            if v >= T["confounding_v"]:
                fire("batch_condition_confounding", "high", [_ev(p, path, T["confounding_v"], ">=")])
            else:
                checked.setdefault("batch_condition_confounding", []).append(_ev(p, path, T["confounding_v"], "<"))

    mm = p["multimodal"]
    if mm["n_modalities"] >= 2 and mm["paired"]:
        fire("multimodal_paired", "high", [_ev(p, ["multimodal", "n_modalities"], 2, ">="), _ev(p, ["multimodal", "kinds"])])
        for pair, a in mm.get("agreement", {}).items():
            if a["knn_overlap"] < T["low_agreement_ratio"] * a["chance"]:
                fire("low_modality_agreement", "medium", [_ev(p, ["multimodal", "agreement", pair, "knn_overlap"]),
                                                          _ev(p, ["multimodal", "agreement", pair, "chance"])])

    flags = p.get("caveat_flags", {})
    for f in ("label_leakage", "label_derived_from_input"):
        if f in flags:
            fire("label_input_leakage", "high", [_ev(p, ["caveat_flags", f])])
    if "pretraining_overlap" in flags:
        fire("pretraining_overlap", "medium", [_ev(p, ["caveat_flags", "pretraining_overlap"])])

    not_fired = {cid: {"id": cid, "checked_evidence": checked.get(cid, [])} for cid in TAXONOMY if cid not in fired}
    return {"thresholds": T, "fired": list(fired.values()), "not_fired": list(not_fired.values())}


def main():
    ap = argparse.ArgumentParser(description="Evaluate the challenge triggers on profile.json.")
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    out_dir = REPO / "outputs" / a.dataset
    p = json.loads((out_dir / "profile.json").read_text())
    res = evaluate(p)
    (out_dir / "challenge_triggers.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"fired: {[c['id'] + '(' + c['severity'] + ')' for c in res['fired']]}")
    print(f"wrote {out_dir / 'challenge_triggers.json'}")


if __name__ == "__main__":
    main()
