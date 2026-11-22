"""Profiler + challenge rules + challenge validator on small synthetic datasets.

A synthetic h5ad dataset with a planted replicate effect, a control group, a
weak condition and a 3-feature second modality; a synthetic parquet dataset
with text that names its label and labels missing for one group.

Run from the repo root:  venv/bin/python -m unittest discover -s tests -v
"""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import loaders  # noqa: E402
import profiler  # noqa: E402
from challenge_rules import evaluate  # noqa: E402
from validate_challenges import validate  # noqa: E402


def make_h5ad_dataset(root):
    import anndata as ad
    import scipy.sparse as sp
    rng = np.random.default_rng(0)
    n, p = 1200, 300
    rep = np.repeat(["r1", "r2"], n // 2)
    cond = rng.choice(["ctrl", "A", "B"], n, p=[0.3, 0.5, 0.2])
    mu = np.exp(rng.normal(0, 1, p))
    scale = np.ones((n, p))
    scale[rep == "r2", :60] *= 4.0        # strong replicate effect on 60 genes
    scale[cond == "A", 100:103] *= 1.3    # weak condition effect
    X = rng.poisson(mu * scale * 0.5).astype(np.float32)
    obs = pd.DataFrame({"pert": cond, "replicate": rep, "const": "x"}, index=[f"c{i}" for i in range(n)])
    d = root / "synth_counts"
    d.mkdir()
    ad.AnnData(sp.csr_matrix(X), obs=obs).write_h5ad(d / "rna.h5ad")
    prot = rng.poisson(20, (n, 3)).astype(np.float32) + 1
    ad.AnnData(sp.csr_matrix(prot), obs=obs).write_h5ad(d / "prot.h5ad")
    (d / "DATA_DESCRIPTION.md").write_text(
        "# synth\n\n## Loading\n"
        "- modality.rna: file=rna.h5ad | kind=counts\n"
        "- modality.prot: file=prot.h5ad | kind=counts\n"
        "- design.condition: column=pert | control=ctrl\n"
        "- batch.replicate: column=replicate\n")
    return "synth_counts"


def make_parquet_dataset(root):
    from PIL import Image
    rng = np.random.default_rng(1)
    n = 200
    kinds = rng.choice(["CT", "MRI", "XR"], n, p=[0.6, 0.3, 0.1])
    region = [("chest" if rng.random() < 0.5 else "limb") if k == "XR" else None for k in kinds]
    caps = [f"{k} scan of a patient" if rng.random() < 0.7 else "an image of a patient" for k in kinds]
    imgs = []
    for i in range(n):
        buf = io.BytesIO()
        Image.fromarray((rng.random((20 + i % 40, 30)) * 255).astype(np.uint8)).save(buf, format="PNG")
        imgs.append({"bytes": buf.getvalue(), "path": None})
    df = pd.DataFrame({"image": imgs, "sid": [f"s{i}" for i in range(n)], "caption": caps,
                       "kind": kinds, "region": region})
    d = root / "synth_pairs"
    d.mkdir()
    df.iloc[:150].to_parquet(d / "train-0.parquet")
    df.iloc[150:].to_parquet(d / "test-0.parquet")
    (d / "DATA_DESCRIPTION.md").write_text(
        "# synth\n\n## Loading\n"
        "- modality.image: file=*.parquet | kind=image | column=image\n"
        "- modality.caption: file=*.parquet | kind=text | column=caption\n"
        "- id_column: sid\n- split: filename_prefix\n"
        "- label.kind: column=kind\n- label.region: column=region\n\n"
        "## Known caveats\n- Labels were partly extracted from the captions, so they are not independent of the text.\n")
    return "synth_pairs"


class TestProfilerAndChallenges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.counts = make_h5ad_dataset(root)
        cls.pairs = make_parquet_dataset(root)
        cls._orig_root = loaders.DATA_ROOT
        cls._orig_cache = loaders.CACHE_ROOT
        loaders.DATA_ROOT = root
        loaders.CACHE_ROOT = root / "cache"
        loaders.load_dataset.__defaults__ = (True, root)
        cls.p_counts = profiler.profile(cls.counts)
        cls.p_pairs = profiler.profile(cls.pairs)

    @classmethod
    def tearDownClass(cls):
        loaders.load_dataset.__defaults__ = (True, cls._orig_root)
        loaders.CACHE_ROOT = cls._orig_cache
        cls.tmp.cleanup()

    def test_profile_is_json_serializable(self):
        json.dumps(self.p_counts, allow_nan=False)
        json.dumps(self.p_pairs, allow_nan=False)

    def test_design_and_batch_detected(self):
        p = self.p_counts
        self.assertEqual(p["design"]["control_label"], "ctrl")
        self.assertEqual(p["design"]["n_conditions_excluding_control"], 2)
        gs = p["modalities"]["rna"]["group_structure"]["batch:replicate"]
        self.assertGreater(gs["variance_explained_by_group"], 0.05)
        self.assertIn("const", p["constant_metadata_columns"])

    def test_counts_challenges(self):
        fired = {c["id"]: c["severity"] for c in evaluate(self.p_counts)["fired"]}
        self.assertEqual(fired.get("batch_effect"), "high")
        self.assertIn("weak_condition_signal", fired)
        self.assertIn("targeted_panel", fired)
        self.assertIn("multimodal_paired", fired)
        self.assertNotIn("batch_condition_confounding", fired)
        self.assertNotIn("unstructured_modality", fired)

    def test_pairs_profile_and_challenges(self):
        p = self.p_pairs
        self.assertEqual(p["groupings"]["split"]["counts"], {"train": 150, "test": 50})
        self.assertTrue(p["modalities"]["image"]["size_heterogeneous"])
        miss = p["groupings"]["label:region"]["missingness_by"]["label:kind"]
        self.assertGreater(miss["spread"], 0.9)
        fired = {c["id"]: c["severity"] for c in evaluate(p)["fired"]}
        for cid in ("unstructured_modality", "label_input_leakage", "missing_values_or_labels", "multimodal_paired"):
            self.assertIn(cid, fired)
        self.assertEqual(fired["missing_values_or_labels"], "high")
        self.assertIn("structure checks skipped", " ".join(p["warnings"]))

    def _good_doc(self, profile):
        fired = evaluate(profile)["fired"]
        return {"challenges": [{"id": c["id"], "severity": c["severity"], "evidence": c["evidence"][:1],
                                "why_it_matters": f"Specific consequence for this objective: {c['meaning']}"} for c in fired]}

    def test_validator_accepts_complete_doc(self):
        res = validate(self.p_counts, self._good_doc(self.p_counts))
        self.assertTrue(res["passed"], res["errors"])

    def test_validator_rejects_missing_rule(self):
        doc = self._good_doc(self.p_counts)
        doc["challenges"] = [c for c in doc["challenges"] if c["id"] != "batch_effect"]
        res = validate(self.p_counts, doc)
        self.assertFalse(res["passed"])
        self.assertTrue(any("batch_effect" in e for e in res["errors"]))

    def test_validator_rejects_wrong_value_and_bad_path(self):
        doc = self._good_doc(self.p_counts)
        doc["challenges"][0]["evidence"] = [{"path": doc["challenges"][0]["evidence"][0]["path"], "value": 123456},
                                            {"path": ["modalities", "nope"], "value": 1}]
        errs = " ".join(validate(self.p_counts, doc)["errors"])
        self.assertIn("but profile.json has", errs)
        self.assertIn("does not exist", errs)

    def test_validator_rejects_placeholder_reason_and_big_downgrade(self):
        doc = self._good_doc(self.p_counts)
        for c in doc["challenges"]:
            if c["id"] == "batch_effect":
                c["severity"] = "low"
            if c["id"] == "sparse_counts":
                c["why_it_matters"] = "This is the default preprocessing concern for this kind of data."
        errs = " ".join(validate(self.p_counts, doc)["errors"])
        self.assertIn("at most one level", errs)
        self.assertIn("placeholder", errs)

    def test_dismissal_needs_reason_and_evidence_for_high(self):
        doc = self._good_doc(self.p_counts)
        doc["challenges"] = [c for c in doc["challenges"] if c["id"] != "batch_effect"]
        doc["dismissed"] = [{"id": "batch_effect", "reason": "short"}]
        errs = " ".join(validate(self.p_counts, doc)["errors"])
        self.assertIn("at least 40", errs)
        self.assertIn("evidence", errs)


if __name__ == "__main__":
    unittest.main()
