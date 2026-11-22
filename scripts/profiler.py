"""Dataset profiler: writes outputs/<dataset>/profile.json and profile_summary.md.

Generic: everything comes from loaders.load_dataset (driven by the dataset's
DATA_DESCRIPTION.md). No dataset-specific branches.

What it measures (the evidence later steps cite by JSON path):
- per modality: kind, structured or not, size, sparsity, count-ness, missing
  values, quality-control tails (count data), text length and label mentions
  (text), image size and colour-mode spread (image, on a sample);
- sample metadata: declared labels, experimental design (condition vs
  control), declared batches, the official split, and auto-detected batch-like
  columns, each with group sizes, imbalance and missingness;
- structure: for every modality with a numeric representation (counts, or a
  cached encoder embedding for image/text), a diagnostic PCA, and for every
  grouping variable in it: fraction of PC variance explained by the grouping
  (R^2), kNN same-group fraction vs chance, and silhouette;
- design checks: condition signal vs control spread, and confounding between
  batch and condition (Cramer's V);
- cross-modal pairing and agreement between modality representations;
- caveat flags: sentences in DATA_DESCRIPTION.md's '## Known caveats' that
  mention leakage, label derivation or pretraining overlap.

Usage: venv/bin/python scripts/profiler.py --dataset <name> [--out outputs/<name>/profile.json] [--seed 0]
"""
import argparse
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import loaders

SEED = 0
MAX_DIAG_SAMPLES = 6000        # subsample for PCA / kNN / silhouette diagnostics
N_PCS = 30
N_TOP_FEATURES = 2000          # top-variance features for the count-data diagnostic space
KNN_K = 15
IMAGE_SAMPLE = 300
SMALL_GROUP = 30               # groups below this are too small for stable per-group statistics
MAD_K = 5
BATCH_NAME_RE = re.compile(r"(batch|replicate|^rep|donor|sample|plate|lane|librar|hto|hash|run|patient|site|source|orig\.?ident|split|day|time|well)", re.I)
CONTROL_RE = re.compile(r"^(control|ctrl|nt|non[-_ ]?targeting|dmso|vehicle|untreated|mock|scrambled?|safe[-_ ]?targeting|wt)$", re.I)
CAVEAT_TERMS = {
    "label_derived_from_input": re.compile(r"(extracted from|derived from|assigned .* from|used both modalities|not independent)", re.I),
    "label_leakage": re.compile(r"(leak|easy to predict from|name the .* explicitly)", re.I),
    "pretraining_overlap": re.compile(r"(pretrain|training data|inflate)", re.I),
    "few_samples_per_class": re.compile(r"(fewer than|unstable per-class|too small)", re.I),
}


def rnd(x, k=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


# ── per-modality summaries ──

def _to_csr(X):
    return X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X))


def summarize_matrix(mod, rng):
    X = mod.X
    n, p = X.shape
    Xs = _to_csr(X)
    data = Xs.data
    nan = int(np.isnan(data).sum()) if data.dtype.kind == "f" else 0
    sample = data[rng.choice(len(data), min(len(data), 200000), replace=False)] if len(data) else data
    sample = sample[np.isfinite(sample)]
    is_int = bool(sample.size and np.all(sample >= 0) and np.all(sample == np.round(sample)))
    lib = np.asarray(Xs.sum(axis=1)).ravel()
    detected = np.diff(Xs.indptr)
    out = {
        "kind": mod.kind,
        "structured": True,
        "n_samples": n,
        "n_features": p,
        "features_per_sample_ratio": rnd(p / n, 3),
        "sparsity": rnd(1 - Xs.nnz / (n * p)),
        "is_nonneg_integer": is_int,
        "value_max": rnd(sample.max() if sample.size else None, 2),
        "missing_values": {"n_missing": nan, "fraction": rnd(nan / (n * p), 6)},
        "n_empty_samples": int((detected == 0).sum()),
        "n_empty_features": int((np.diff(_to_csr(X.T if not sp.issparse(X) else X.tocsc().T).indptr) == 0).sum()) if p < 300000 else None,
    }
    if is_int and p > 1:
        out["qc"] = qc_tails(lib, detected)
    return out


def qc_tails(lib, detected):
    """Fraction of samples more than MAD_K median absolute deviations from the
    median in log total counts or log detected features."""
    res = {"total_counts_median": rnd(np.median(lib), 1), "detected_features_median": rnd(np.median(detected), 1)}
    for name, v in (("total_counts", lib), ("detected_features", detected)):
        lv = np.log1p(v.astype(float))
        med = np.median(lv)
        mad = np.median(np.abs(lv - med)) * 1.4826
        if mad == 0:
            res[f"{name}_tail_fraction"] = 0.0
            continue
        res[f"{name}_high_tail_fraction"] = rnd(np.mean(lv > med + MAD_K * mad))
        res[f"{name}_low_tail_fraction"] = rnd(np.mean(lv < med - MAD_K * mad))
    res["mad_threshold"] = MAD_K
    return res


def summarize_text(mod, obs, label_cols):
    t = mod.texts
    words = t.str.split().str.len().fillna(0)
    out = {
        "kind": "text",
        "structured": False,
        "n_samples": int(len(t)),
        "empty_fraction": rnd((t.str.strip() == "").mean()),
        "words_median": rnd(words.median(), 1),
        "words_p10": rnd(words.quantile(0.1), 1),
        "words_p90": rnd(words.quantile(0.9), 1),
        "label_mentions": {},
    }
    low = t.str.lower()
    for col in label_cols:
        y = obs[col]
        ok = y.notna()
        if ok.sum() == 0:
            continue
        # does the text name its own label? (a cheap leakage check)
        hits = [lab.lower() in txt for lab, txt in zip(y[ok].astype(str), low[ok])]
        out["label_mentions"][col] = {"fraction_text_contains_own_label": rnd(np.mean(hits)), "n": int(ok.sum())}
    return out


def summarize_images(ds, name, rng):
    from PIL import Image
    n = len(ds.obs)
    idx = np.sort(rng.choice(n, min(IMAGE_SAMPLE, n), replace=False))
    sizes, modes, bad = [], {}, 0
    for b in ds.images(name, idx):
        try:
            im = Image.open(io.BytesIO(b))
            sizes.append(im.size)
            modes[im.mode] = modes.get(im.mode, 0) + 1
        except Exception:
            bad += 1
    w = np.array([s[0] for s in sizes]); h = np.array([s[1] for s in sizes])
    ar = w / np.maximum(h, 1)
    return {
        "kind": "image",
        "structured": False,
        "n_samples": n,
        "n_checked": int(len(idx)),
        "n_undecodable": bad,
        "width_median": rnd(np.median(w), 1), "width_min": int(w.min()), "width_max": int(w.max()),
        "height_median": rnd(np.median(h), 1), "height_min": int(h.min()), "height_max": int(h.max()),
        "aspect_ratio_p10": rnd(np.quantile(ar, 0.1), 2), "aspect_ratio_p90": rnd(np.quantile(ar, 0.9), 2),
        "color_modes": {k: rnd(v / len(sizes), 3) for k, v in sorted(modes.items())},
        "size_heterogeneous": bool(w.max() > 2 * w.min() or h.max() > 2 * h.min()),
    }


# ── grouping variables ──

def group_summary(values):
    s = pd.Series(values)
    miss = s.isna() | (s.astype(str).str.lower().isin(["nan", "none", ""]))
    counts = s[~miss].astype(str).value_counts()
    out = {
        "n_groups": int(len(counts)),
        "n_missing": int(miss.sum()),
        "missing_fraction": rnd(miss.mean()),
        "counts": {k: int(v) for k, v in counts.head(60).items()},
    }
    if len(counts) > 60:
        out["counts_truncated"] = True
    if len(counts) >= 2:
        out["imbalance_ratio"] = rnd(counts.max() / counts.min(), 2)
        out["largest_fraction"] = rnd(counts.max() / counts.sum())
        out["min_group_size"] = int(counts.min())
        out["n_small_groups"] = int((counts < SMALL_GROUP).sum())
        out["small_groups"] = [k for k, v in counts.items() if v < SMALL_GROUP][:20]
    return out


def detect_batch_candidates(ds, used):
    found = {}
    for c in ds.raw_obs_columns:
        if c in used:
            continue
        s = ds.obs[f"meta:{c}"]
        if s.dtype.kind == "f" and s.nunique() > 50:
            continue
        k = s.astype(str).nunique()
        if 2 <= k <= max(2, min(50, len(s) // 20)) and BATCH_NAME_RE.search(c):
            found[c] = s.astype(str).values
    return found


def missing_pattern(obs, col, others):
    """For a label with missing values: is the missingness tied to another variable?"""
    miss = obs[col].isna()
    if miss.sum() == 0 or miss.all():
        return None
    out = {}
    for o in others:
        if o == col:
            continue
        y = obs[o]
        ok = y.notna()
        rate = miss[ok].groupby(y[ok].astype(str)).mean()
        if len(rate) >= 2:
            out[o] = {"missing_rate_by_group": {k: rnd(v, 3) for k, v in rate.items()},
                      "spread": rnd(rate.max() - rate.min(), 3)}
    return out or None


def cramers_v(a, b):
    t = pd.crosstab(pd.Series(a).astype(str), pd.Series(b).astype(str)).values
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 0.0
    n = t.sum()
    expected = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / n
    chi2 = ((t - expected) ** 2 / np.maximum(expected, 1e-12)).sum()
    return float(np.sqrt(chi2 / (n * (min(t.shape) - 1))))


# ── diagnostic spaces ──

def diag_space_counts(X, rows, rng):
    """log-normalized counts, top-variance features, PCA on a row subsample."""
    from sklearn.decomposition import PCA
    Xs = _to_csr(X)[rows].astype(np.float64)
    lib = np.asarray(Xs.sum(axis=1)).ravel()
    lib[lib == 0] = 1
    Xn = sp.diags(1e4 / lib) @ Xs
    Xn.data = np.log1p(Xn.data)
    mean = np.asarray(Xn.mean(axis=0)).ravel()
    var = np.asarray(Xn.multiply(Xn).mean(axis=0)).ravel() - mean ** 2
    top = np.argsort(-var)[: min(N_TOP_FEATURES, Xn.shape[1])]
    D = Xn[:, top].toarray()
    D = (D - D.mean(0)) / np.maximum(D.std(0), 1e-8)
    k = min(N_PCS, D.shape[1] - 1, D.shape[0] - 1)
    if k < 1:
        return D, None
    pca = PCA(n_components=k, random_state=SEED).fit(D)
    return pca.transform(D), pca.explained_variance_ratio_


def diag_space_embedding(E):
    from sklearn.decomposition import PCA
    D = (E - E.mean(0)) / np.maximum(E.std(0), 1e-8)
    k = min(N_PCS, D.shape[1] - 1)
    pca = PCA(n_components=k, random_state=SEED).fit(D)
    return pca.transform(D), pca.explained_variance_ratio_


def group_structure(Z, evr, y, rng):
    """How strongly a grouping shows up in a representation Z (rows = samples)."""
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    y = pd.Series(y).astype(object)
    ok = y.notna().values & ~y.astype(str).str.lower().isin(["nan", "none"]).values
    Z, y = Z[ok], y[ok].astype(str).to_numpy(dtype=object)
    if len(np.unique(y)) < 2 or len(y) < 50:
        return None
    # fraction of each PC's variance explained by group means, weighted by PC variance
    r2 = []
    for j in range(Z.shape[1]):
        z = Z[:, j]
        means = pd.Series(z).groupby(y).transform("mean").values
        tot = ((z - z.mean()) ** 2).sum()
        r2.append(((means - z.mean()) ** 2).sum() / tot if tot > 0 else 0.0)
    r2 = np.array(r2)
    w = evr[: len(r2)] / evr[: len(r2)].sum() if evr is not None else np.ones(len(r2)) / len(r2)
    nn = NearestNeighbors(n_neighbors=KNN_K + 1).fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)[:, 1:]
    same = (y[idx] == y[:, None]).mean()
    _, counts = np.unique(y, return_counts=True)
    expected = float(((counts / counts.sum()) ** 2).sum())
    sub = rng.choice(len(y), min(3000, len(y)), replace=False)
    sil = silhouette_score(Z[sub, :10], y[sub]) if len(np.unique(y[sub])) > 1 else None
    return {
        "n_samples_used": int(len(y)),
        "variance_explained_by_group": rnd(float((w * r2).sum())),
        "pc1_r2": rnd(r2[0]),
        "max_pc_r2": rnd(r2.max()),
        "knn_same_group_fraction": rnd(same),
        "knn_same_group_expected": rnd(expected),
        "knn_enrichment": rnd(same / expected if expected > 0 else None, 3),
        "silhouette": rnd(sil),
    }


def condition_signal(Z, cond, control, rng):
    """Per condition: distance of its centroid from the control centroid, in
    units of the control cells' spread (RMS distance to their own centroid)."""
    cond = np.asarray(cond).astype(str)
    ctrl = cond == control
    if ctrl.sum() < 10:
        return None
    c0 = Z[ctrl].mean(0)
    spread = np.sqrt(((Z[ctrl] - c0) ** 2).sum(1).mean())
    per = {}
    for g in np.unique(cond):
        if g == control:
            continue
        m = cond == g
        if m.sum() < 5:
            continue
        per[g] = rnd(np.linalg.norm(Z[m].mean(0) - c0) / spread, 4)
    vals = np.array(list(per.values())) if per else np.array([np.nan])
    # null: the same statistic for random control subsets of a typical group size
    # capped at half the controls so the null can always be drawn
    size = min(int(np.median([np.sum(cond == g) for g in per])), int(ctrl.sum()) // 2) if per else 0
    null = []
    ctrl_idx = np.where(ctrl)[0]
    if size and len(ctrl_idx) > size:
        for _ in range(50):
            pick = rng.choice(ctrl_idx, size, replace=False)
            null.append(np.linalg.norm(Z[pick].mean(0) - c0) / spread)
    null_q95 = float(np.quantile(null, 0.95)) if null else None
    return {
        "control_label": control,
        "shift_over_control_spread": dict(sorted(per.items(), key=lambda kv: -kv[1])),
        "median_shift": rnd(np.nanmedian(vals)),
        "max_shift": rnd(np.nanmax(vals)),
        "null_shift_q95": rnd(null_q95),
        "n_conditions_above_null": int(np.sum(vals > null_q95)) if null_q95 is not None else None,
        "n_conditions_tested": int(len(per)),
    }


def modality_agreement(spaces, rng):
    """kNN overlap between modality representations of the same samples."""
    from sklearn.neighbors import NearestNeighbors
    names = list(spaces)
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            (Za, ra), (Zb, rb) = spaces[names[i]][:2], spaces[names[j]][:2]
            common, ia, ib = np.intersect1d(ra, rb, return_indices=True)
            if len(common) < 100:
                continue
            A, B = Za[ia], Zb[ib]
            na = NearestNeighbors(n_neighbors=KNN_K + 1).fit(A).kneighbors(A, return_distance=False)[:, 1:]
            nb = NearestNeighbors(n_neighbors=KNN_K + 1).fit(B).kneighbors(B, return_distance=False)[:, 1:]
            ov = np.mean([len(set(a) & set(b)) / KNN_K for a, b in zip(na, nb)])
            out[f"{names[i]}~{names[j]}"] = {"knn_overlap": rnd(ov), "chance": rnd(KNN_K / len(common), 5), "n": int(len(common))}
    return out


def caveat_flags(desc_path):
    text = Path(desc_path).read_text()
    sec = loaders._section(text, "Known caveats")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n-\s*", sec.replace("\n  ", " ")) if s.strip()]
    flags = {}
    for name, rx in CAVEAT_TERMS.items():
        hits = [s.lstrip("- ")[:240] for s in sentences if rx.search(s)]
        if hits:
            flags[name] = hits[:3]
    return flags


# ── main ──

def profile(name, seed=SEED):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    ds = loaders.load_dataset(name)
    obs = ds.obs
    n = len(obs)
    prof = {"dataset": name, "n_samples": n, "seed": seed, "modalities": {}, "groupings": {}, "warnings": []}

    # groupings: labels, design, declared batches, split, auto-detected batch candidates
    groupings = {}
    for c in ds.label_columns:
        groupings[c] = ("label", obs[c].values)
    for c in [c for c in obs.columns if c.startswith("design:")]:
        groupings[c] = ("condition", obs[c].values)
    for c in ds.batch_columns:
        groupings[c] = ("batch", obs[c].values)
    if "split" in obs.columns:
        groupings["split"] = ("split", obs["split"].values)
    used = {f["column"] for grp in ("label", "design", "batch") for f in ds.spec[grp].values()}
    for c, v in detect_batch_candidates(ds, used).items():
        groupings[f"meta:{c}"] = ("batch_candidate", v)
    for g, (role, v) in groupings.items():
        prof["groupings"][g] = {"role": role} | group_summary(v)
        mp = missing_pattern(obs, g, [x for x in groupings if x != g]) if g.startswith("label:") else None
        if mp:
            prof["groupings"][g]["missingness_by"] = mp
    prof["constant_metadata_columns"] = [c for c in ds.raw_obs_columns if obs[f"meta:{c}"].astype(str).nunique() <= 1]

    # design
    for g, (role, v) in groupings.items():
        if role != "condition":
            continue
        key = g.split(":", 1)[1]
        declared = ds.spec["design"][key].get("control")
        levels = pd.Series(v).astype(str).unique()
        control = declared if declared in levels else next((l for l in levels if CONTROL_RE.match(l)), None)
        prof["design"] = {
            "condition_column": g,
            "control_label": control,
            "control_declared": declared is not None,
            "n_conditions_excluding_control": int(len(levels) - (control is not None)),
            "control_fraction": rnd(np.mean(np.asarray(v).astype(str) == control)) if control else None,
        }
        batches = [b for b, (r, _) in groupings.items() if r in ("batch", "batch_candidate", "split")]
        prof["design"]["batch_confounding_cramers_v"] = {b: rnd(cramers_v(v, groupings[b][1]), 3) for b in batches}

    # modalities and diagnostic spaces
    rows = np.sort(rng.choice(n, min(n, MAX_DIAG_SAMPLES), replace=False))
    spaces = {}
    caches = loaders.list_embedding_caches(name)
    for mname, mod in ds.modalities.items():
        if mod.kind in ("counts", "continuous"):
            s = summarize_matrix(mod, rng)
            Z, evr = diag_space_counts(mod.X, rows, rng)
            s["diagnostic_space"] = {"source": "log-normalized top-variance features, standardized, PCA",
                                     "n_features_used": int(min(N_TOP_FEATURES, mod.X.shape[1]))}
        elif mod.kind == "text":
            s = summarize_text(mod, obs, ds.label_columns)
            Z = evr = None
        else:
            s = summarize_images(ds, mname, rng)
            Z = evr = None
        if mod.kind in ("text", "image"):
            enc = next((e for e, m in caches.items() if mname in m.get("modalities", {})), None)
            s["cached_encoders"] = [e for e, m in caches.items() if mname in m.get("modalities", {})]
            if enc:
                E = loaders.load_embeddings(ds, enc, mname)[rows]
                Z, evr = diag_space_embedding(E)
                s["diagnostic_space"] = {"source": f"cached frozen embedding '{enc}', standardized, PCA",
                                         "embedding_dim": int(E.shape[1])}
            else:
                prof["warnings"].append(f"{mname}: unstructured modality with no cached embedding; structure checks skipped")
        if Z is not None:
            s["diagnostic_space"]["n_pcs"] = int(Z.shape[1])
            s["diagnostic_space"]["explained_variance_top"] = [rnd(x) for x in evr[:10]]
            s["diagnostic_space"]["explained_variance_total"] = rnd(evr.sum())
            spaces[mname] = (Z, rows, evr)
            s["group_structure"] = {}
            for g, (role, v) in groupings.items():
                r = group_structure(Z, evr, np.asarray(v, dtype=object)[rows], rng)
                if r:
                    s["group_structure"][g] = {"role": role} | r
            if "design" in prof and prof["design"]["control_label"]:
                cs = condition_signal(Z, np.asarray(groupings[prof["design"]["condition_column"]][1])[rows],
                                      prof["design"]["control_label"], rng)
                if cs:
                    s["condition_signal"] = cs
        prof["modalities"][mname] = s

    prof["multimodal"] = {
        "n_modalities": len(ds.modalities),
        "kinds": {m: mod.kind for m, mod in ds.modalities.items()},
        "paired": True,  # the loader refuses modalities whose samples differ
        "has_unstructured": any(mod.kind in loaders.UNSTRUCTURED for mod in ds.modalities.values()),
        "has_structured": any(mod.kind not in loaders.UNSTRUCTURED for mod in ds.modalities.values()),
        "feature_counts": {m: (prof["modalities"][m].get("n_features")) for m in ds.modalities},
        "agreement": modality_agreement(spaces, rng) if len(spaces) > 1 else {},
    }
    prof["caveat_flags"] = caveat_flags(ds.folder / "DATA_DESCRIPTION.md")
    prof["runtime_s"] = rnd(time.time() - t0, 1)
    return prof


def summary_md(p):
    L = [f"# Profile summary: {p['dataset']}", "", f"- Samples: {p['n_samples']:,}"]
    for m, s in p["modalities"].items():
        bits = [s["kind"], "structured" if s["structured"] else "unstructured"]
        if "n_features" in s:
            bits.append(f"{s['n_features']:,} features, {s['sparsity'] * 100:.1f}% zeros")
        if s["kind"] == "text":
            bits.append(f"median {s['words_median']} words")
        if s["kind"] == "image":
            bits.append(f"width {s['width_min']}-{s['width_max']} px, modes {s['color_modes']}")
        L.append(f"- Modality `{m}`: " + "; ".join(bits))
    L.append("")
    L.append("## Groupings")
    for g, s in p["groupings"].items():
        L.append(f"- `{g}` ({s['role']}): {s['n_groups']} groups, missing {s['missing_fraction']}, "
                 f"imbalance {s.get('imbalance_ratio')}, smallest {s.get('min_group_size')}")
    if "design" in p:
        L += ["", "## Design", f"- {json.dumps(p['design'])}"]
    L += ["", "## Structure (variance explained by group / kNN enrichment)"]
    for m, s in p["modalities"].items():
        for g, r in s.get("group_structure", {}).items():
            L.append(f"- {m} x `{g}`: R2 {r['variance_explained_by_group']}, kNN enrichment {r['knn_enrichment']}, silhouette {r['silhouette']}")
        if "condition_signal" in s:
            cs = s["condition_signal"]
            L.append(f"- {m}: condition shift median {cs['median_shift']}, max {cs['max_shift']}, null q95 {cs['null_shift_q95']}, "
                     f"{cs['n_conditions_above_null']}/{cs['n_conditions_tested']} above null")
    if p["multimodal"]["agreement"]:
        L += ["", "## Modality agreement", f"- {json.dumps(p['multimodal']['agreement'])}"]
    if p["caveat_flags"]:
        L += ["", "## Caveat flags", *[f"- {k}: {v}" for k, v in p["caveat_flags"].items()]]
    if p["warnings"]:
        L += ["", "## Warnings", *[f"- {w}" for w in p["warnings"]]]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    out = Path(a.out) if a.out else loaders.REPO / "outputs" / a.dataset / "profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    p = profile(a.dataset, a.seed)
    out.write_text(json.dumps(p, indent=2, allow_nan=False) + "\n")
    out.with_name("profile_summary.md").write_text(summary_md(p))
    print(f"Wrote {out} ({p['runtime_s']} s)")


if __name__ == "__main__":
    main()
