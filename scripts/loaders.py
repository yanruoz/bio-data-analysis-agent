"""Generic multi-modality dataset loader.

Everything about a dataset comes from the '## Loading' section of
data/<name>/DATA_DESCRIPTION.md. There is no per-dataset code: nothing in this
file, and nothing downstream of it, may branch on which dataset is loaded.

'## Loading' lines (one per bullet, fields separated by '|'):
    - modality.<name>: file=<file or glob> | kind=counts|continuous|image|text [| column=<col>]
    - label.<name>:    column=<col> [| first_code_of=<code>:<label>, <code>:<label>, ...]
    - design.condition: column=<col> [| extract=<regex with one group>] [| control=<value>]
    - batch.<name>:    column=<col>
    - qc.<modality>:   columns=<col>, <col>, ...
    - id_column: <col>                        (table sources: the sample id column)
    - split: filename_prefix | column=<col>   (official train/test split, if any)

Two source types:
- .h5ad: one AnnData file per modality; samples are matched by obs_names.
  Sample metadata (labels, design, batch, qc) is read from the first h5ad's obs.
- .parquet (a glob may match several shards): table rows are samples; an image
  modality is a column of encoded image bytes, a text modality a string column.

The loader never touches the network (scripts/fetch_data.py does that) and
never decodes every image: image columns are read on demand.
"""
import glob
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "data"
CACHE_ROOT = REPO / "cache"

KINDS = {"counts", "continuous", "image", "text"}
UNSTRUCTURED = {"image", "text"}
TOP_KEYS = {"id_column", "split"}
PREFIXED = {"modality", "label", "design", "batch", "qc"}
FIELDS = {
    "modality": {"file", "kind", "column"},
    "label": {"column", "first_code_of"},
    "design": {"column", "extract", "control"},
    "batch": {"column"},
    "qc": {"columns"},
}
SPLIT_NAMES = ("train", "test", "val", "valid", "validation")


# ── '## Loading' parsing ──

def _section(text, title):
    m = re.search(rf"^##\s+{title}\s*$(.*?)(?=^##\s|\Z)", text, flags=re.M | re.S)
    return m.group(1) if m else ""


def _fields(value):
    out = {}
    for part in value.split("|"):
        k, eq, v = part.partition("=")
        if not eq:
            raise ValueError(f"expected key=value, got {part.strip()!r}")
        out[k.strip()] = v.strip()
    return out


def parse_loading_spec(desc_path):
    """Return {'modality': {name: fields}, 'label': {...}, 'design': {...},
    'batch': {...}, 'qc': {...}, 'id_column': str|None, 'split': str|None}.
    Unknown keys or fields raise ValueError instead of being silently dropped."""
    desc_path = Path(desc_path)
    text = desc_path.read_text(encoding="utf-8", errors="replace")
    spec = {p: {} for p in PREFIXED} | {"id_column": None, "split": None}
    for line in _section(text, "Loading").splitlines():
        m = re.match(r"^\s*[-*]\s*`?([a-z_]+)(?:\.([A-Za-z0-9_]+))?`?\s*:\s*(.+?)\s*$", line)
        if not m:
            continue
        prefix, name, value = m.groups()
        if name is None:
            if prefix not in TOP_KEYS:
                raise ValueError(f"{desc_path}: unknown '## Loading' key {prefix!r}; allowed {sorted(TOP_KEYS | PREFIXED)}")
            spec[prefix] = value
            continue
        if prefix not in PREFIXED:
            raise ValueError(f"{desc_path}: unknown '## Loading' key {prefix}.{name}")
        try:
            f = _fields(value)
        except ValueError as e:
            raise ValueError(f"{desc_path}: {prefix}.{name}: {e}") from None
        bad = set(f) - FIELDS[prefix]
        if bad:
            raise ValueError(f"{desc_path}: {prefix}.{name} has unknown fields {sorted(bad)}; allowed {sorted(FIELDS[prefix])}")
        spec[prefix][name] = f
    if not spec["modality"]:
        raise ValueError(f"{desc_path}: '## Loading' declares no modality.<name> lines")
    for name, f in spec["modality"].items():
        if f.get("kind") not in KINDS:
            raise ValueError(f"{desc_path}: modality.{name} kind must be one of {sorted(KINDS)}")
        if "file" not in f:
            raise ValueError(f"{desc_path}: modality.{name} needs file=")
    return spec


def _codes_map(text):
    pairs = {}
    for item in text.split(","):
        code, _, label = item.partition(":")
        if code.strip() and label.strip():
            pairs[code.strip()] = label.strip()
    return pairs


# ── Dataset object ──

@dataclass
class Modality:
    name: str
    kind: str
    files: list
    column: str = None
    X: object = None              # samples x features (scipy sparse or ndarray) for counts/continuous
    feature_names: list = None
    texts: object = None          # pd.Series of strings for text modalities
    shape: tuple = None

    @property
    def structured(self):
        return self.kind not in UNSTRUCTURED


@dataclass
class Dataset:
    name: str
    folder: Path
    spec: dict
    obs: pd.DataFrame             # one row per sample, index = sample id
    modalities: dict = field(default_factory=dict)
    raw_obs_columns: list = field(default_factory=list)   # source metadata columns (for batch detection)
    _rows: pd.DataFrame = None    # table sources: (file, row_in_file) per sample, for on-demand image reads

    # derived column names in obs
    @property
    def label_columns(self):
        return [f"label:{n}" for n in self.spec["label"]]

    @property
    def batch_columns(self):
        return [f"batch:{n}" for n in self.spec["batch"]]

    def images(self, modality, index):
        """Encoded image bytes for the given sample positions (decoded by the caller)."""
        import pyarrow.parquet as pq
        mod = self.modalities[modality]
        out = [None] * len(index)
        rows = self._rows.iloc[list(index)]
        for f, grp in rows.groupby("file", sort=False):
            table = pq.read_table(self.folder / f, columns=[mod.column])
            col = table.column(mod.column)
            for pos, r in zip(grp.index, grp["row_in_file"]):
                v = col[int(r)].as_py()
                out[rows.index.get_loc(pos)] = v["bytes"] if isinstance(v, dict) else v
        return out


def _resolve(folder, pattern):
    files = sorted(Path(p).name for p in glob.glob(str(folder / pattern)))
    if not files:
        raise ValueError(f"no file matches {pattern!r} under {folder}")
    return files


def _add_sample_metadata(ds, meta, source_cols):
    spec = ds.spec
    obs = ds.obs
    for name, f in spec["label"].items():
        col = f.get("column")
        if col not in meta.columns:
            raise ValueError(f"label.{name}: column {col!r} not found; columns: {list(meta.columns)[:20]}")
        if "first_code_of" in f:
            codes = _codes_map(f["first_code_of"])

            def first(v, codes=codes):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return None
                seq = list(v) if not isinstance(v, str) else [v]
                # the first code in the list that the map names; samples with none get no label
                return next((codes[str(c).strip()] for c in seq if str(c).strip() in codes), None)
            obs[f"label:{name}"] = [first(v) for v in meta[col]]
        else:
            obs[f"label:{name}"] = meta[col].astype(object).where(meta[col].notna(), None).values
    for name, f in spec["design"].items():
        col = f.get("column")
        if col not in meta.columns:
            raise ValueError(f"design.{name}: column {col!r} not found")
        vals = meta[col].astype(str)
        if f.get("extract"):
            vals = vals.str.extract(f["extract"], expand=False).fillna(vals)
        obs[f"design:{name}"] = vals.values
    for name, f in spec["batch"].items():
        col = f.get("column")
        if col not in meta.columns:
            raise ValueError(f"batch.{name}: column {col!r} not found")
        obs[f"batch:{name}"] = meta[col].astype(str).values
    for mod, f in spec["qc"].items():
        for col in [c.strip() for c in f.get("columns", "").split(",") if c.strip()]:
            if col not in meta.columns:
                raise ValueError(f"qc.{mod}: column {col!r} not found")
            obs[f"qc:{mod}:{col}"] = pd.to_numeric(meta[col], errors="coerce").values
    ds.raw_obs_columns = list(source_cols)
    for c in source_cols:
        obs[f"meta:{c}"] = meta[c].values


def _load_h5ad(ds, load_matrices):
    import anndata as ad
    first_obs = None
    for name, f in ds.spec["modality"].items():
        files = _resolve(ds.folder, f["file"])
        if len(files) != 1:
            raise ValueError(f"modality.{name}: an h5ad modality must name exactly one file")
        a = ad.read_h5ad(ds.folder / files[0], backed=None if load_matrices else "r")
        mod = Modality(name, f["kind"], files)
        if load_matrices:
            mod.X = a.X
        mod.feature_names = a.var_names.astype(str).tolist()
        mod.shape = a.shape
        if first_obs is None:
            first_obs = a.obs.copy()
            ds.obs = pd.DataFrame(index=a.obs_names.astype(str))
        else:
            ids = a.obs_names.astype(str)
            if not np.array_equal(ids, ds.obs.index.values):
                common = ds.obs.index.intersection(ids)
                raise ValueError(f"modality.{name}: samples differ from the first modality "
                                 f"({len(common)} shared of {len(ds.obs)}); aligning is not supported")
        ds.modalities[name] = mod
    _add_sample_metadata(ds, first_obs, first_obs.columns)


def _load_parquet(ds):
    import pyarrow.parquet as pq
    spec = ds.spec
    files = sorted({fn for f in spec["modality"].values() for fn in _resolve(ds.folder, f["file"])})
    meta_cols = set()
    for f in spec["label"].values():
        meta_cols.add(f["column"])
    for grp in ("design", "batch"):
        for f in spec[grp].values():
            meta_cols.add(f["column"])
    text_cols = {f["column"] for f in spec["modality"].values() if f["kind"] == "text"}
    id_col = spec["id_column"]
    image_cols = {f["column"] for f in spec["modality"].values() if f["kind"] == "image"}
    frames = []
    for fn in files:
        pf = pq.ParquetFile(ds.folder / fn)
        all_cols = pf.schema_arrow.names
        scalar_cols = [c for c in all_cols if c not in image_cols]
        t = pf.read(columns=scalar_cols).to_pandas()
        t["__file"] = fn
        t["__row"] = np.arange(len(t))
        frames.append(t)
    table = pd.concat(frames, ignore_index=True)
    ids = table[id_col].astype(str) if id_col else table["__file"] + ":" + table["__row"].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"id_column {id_col!r} has duplicate values")
    ds.obs = pd.DataFrame(index=ids.values)
    ds._rows = pd.DataFrame({"file": table["__file"].values, "row_in_file": table["__row"].values}, index=ids.values)
    split = spec["split"]
    if split == "filename_prefix":
        ds.obs["split"] = [next((s for s in SPLIT_NAMES if f.startswith(s)), "unknown") for f in table["__file"]]
    elif split and split.startswith("column="):
        ds.obs["split"] = table[split.split("=", 1)[1]].astype(str).values
    source_cols = [c for c in table.columns if not c.startswith("__") and c not in text_cols and c != id_col]
    _add_sample_metadata(ds, table, source_cols)
    for name, f in spec["modality"].items():
        mod = Modality(name, f["kind"], _resolve(ds.folder, f["file"]), column=f.get("column"))
        if f["kind"] == "text":
            mod.texts = table[f["column"]].fillna("").astype(str).set_axis(ds.obs.index)
        elif f["kind"] != "image":
            raise ValueError(f"modality.{name}: parquet sources support image and text kinds only")
        ds.modalities[name] = mod


def load_dataset(name, load_matrices=True, data_root=DATA_ROOT):
    folder = Path(data_root) / name
    desc = folder / "DATA_DESCRIPTION.md"
    if not desc.is_file():
        avail = sorted(p.name for p in Path(data_root).iterdir() if p.is_dir()) if Path(data_root).is_dir() else []
        raise ValueError(f"Unknown dataset {name!r}: no {desc}. Datasets present: {avail}")
    spec = parse_loading_spec(desc)
    ds = Dataset(name=name, folder=folder, spec=spec, obs=None)
    suffixes = {Path(f["file"]).suffix.lower() for f in spec["modality"].values()}
    if suffixes == {".h5ad"}:
        _load_h5ad(ds, load_matrices)
    elif suffixes == {".parquet"}:
        _load_parquet(ds)
    else:
        raise ValueError(f"{name}: unsupported or mixed source types {sorted(suffixes)}")
    return ds


# ── Cached encoder embeddings (written by the P3 tool's extract_embeddings step) ──

def list_embedding_caches(name, cache_root=CACHE_ROOT):
    """{encoder: manifest dict} for complete caches under cache/<name>/."""
    import json
    out = {}
    for d in sorted((Path(cache_root) / name).glob("*/manifest.json")):
        m = json.loads(d.read_text())
        if m.get("complete"):
            out[d.parent.name] = m
    return out


def load_embeddings(ds, encoder, modality, cache_root=CACHE_ROOT):
    """Embedding matrix for one modality, rows aligned to ds.obs.index."""
    folder = Path(cache_root) / ds.name / encoder
    ids = pd.read_csv(folder / "ids.csv")
    id_col = "pair_id" if "pair_id" in ids.columns else ids.columns[0]
    E = np.load(folder / f"{modality}.npy")
    pos = pd.Series(np.arange(len(ids)), index=ids[id_col].astype(str))
    missing = ds.obs.index.difference(pos.index)
    if len(missing):
        raise ValueError(f"{encoder}/{modality}: {len(missing)} samples have no cached embedding")
    return E[pos.loc[ds.obs.index].values]
