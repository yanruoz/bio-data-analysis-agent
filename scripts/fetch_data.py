"""Setup step, run before any analysis: downloads the files a dataset declares
in the '## Files' section of data/<dataset>/DATA_DESCRIPTION.md and verifies
each one's md5 or sha256. Standard library only, so it runs before the venv
exists. No dataset is special-cased.

Each line in '## Files' looks like:
    - file: <name> | url: <https://...> | md5: <hex>        (or sha256: <hex>)

Usage:
    python3 scripts/fetch_data.py --dataset <name> [--data-root data]

Files that are already present are re-verified, never overwritten. A failed or
mismatched download is deleted, so nothing unverified is left in data/.
The analysis pipeline itself never touches the network.
"""
import argparse
import hashlib
import re
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ALLOWED_SCHEMES = ("https://", "http://", "file://")
# Some hosts reject urllib's default User-Agent, so send a generic one.
USER_AGENT = "bio-agent-fetch/1.0"
HASHES = ("sha256", "md5")


def parse_files_section(desc_path):
    """Return a list of dicts with keys file, url and one of md5/sha256."""
    text = Path(desc_path).read_text()
    m = re.search(r"^## Files\s*$(.*?)(?=^## |\Z)", text, flags=re.M | re.S)
    if not m:
        raise ValueError(f"{desc_path}: no '## Files' section")
    entries = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line.startswith("- file:"):
            continue
        fields = {}
        for part in line[2:].split("|"):
            key, _, value = part.partition(":")
            fields[key.strip()] = value.strip()
        if not fields.get("file") or not fields.get("url"):
            raise ValueError(f"{desc_path}: entry needs both file and url: {line!r}")
        if "/" in fields["file"] or fields["file"].startswith("."):
            raise ValueError(f"{desc_path}: file must be a plain name: {fields['file']!r}")
        if not fields["url"].startswith(ALLOWED_SCHEMES):
            raise ValueError(f"{desc_path}: url scheme not allowed: {fields['url']!r}")
        if not any(h in fields for h in HASHES):
            raise ValueError(f"{desc_path}: {fields['file']} declares no md5 or sha256")
        entries.append(fields)
    if not entries:
        raise ValueError(f"{desc_path}: '## Files' lists no files")
    return entries


def digest(path, algo, chunk=1 << 20):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def check(path, entry):
    for algo in HASHES:
        if algo in entry:
            actual = digest(path, algo)
            return actual == entry[algo].lower(), f"{algo} {actual}"
    return False, "no hash"


def fetch_one(folder, entry):
    target = folder / entry["file"]
    if target.exists():
        ok, got = check(target, entry)
        if not ok:
            raise ValueError(f"{target} exists but its {got} does not match the declared value. "
                             "Not overwriting it; delete it by hand if it should be re-downloaded.")
        return f"already present, verified: {target.name}"
    partial = target.with_name(target.name + ".part")
    try:
        req = urllib.request.Request(entry["url"], headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req) as resp, open(partial, "wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
    except Exception as e:
        partial.unlink(missing_ok=True)
        raise ValueError(f"download of {entry['file']} from {entry['url']} failed: {e}") from e
    ok, got = check(partial, entry)
    if not ok:
        partial.unlink(missing_ok=True)
        raise ValueError(f"{entry['file']}: downloaded file has {got}, not the declared value; discarded it.")
    partial.rename(target)
    return f"downloaded, verified: {target.name}"


def fetch(dataset, data_root):
    folder = Path(data_root) / dataset
    desc = folder / "DATA_DESCRIPTION.md"
    if not desc.is_file():
        raise ValueError(f"No {desc}; write the dataset's DATA_DESCRIPTION.md first.")
    return [fetch_one(folder, e) for e in parse_files_section(desc)]


def main():
    ap = argparse.ArgumentParser(description="Download and verify a dataset's declared files.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", default=str(REPO / "data"))
    args = ap.parse_args()
    try:
        for msg in fetch(args.dataset, args.data_root):
            print(msg, flush=True)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()
