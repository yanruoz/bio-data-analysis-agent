"""Validate outputs/<dataset>/challenges.json against profile.json and the
deterministic triggers (challenge_rules.py).

challenges.json schema:
{
  "dataset": "<name>",
  "challenges": [
    {"id": "<taxonomy id or other:<slug>>",
     "severity": "high" | "medium" | "low",
     "evidence": [{"path": [<keys into profile.json>], "value": <value at that path>}, ...],
     "why_it_matters": "<what goes wrong for THIS objective if ignored>",
     "severity_reason": "<required only when lowering a fired rule's severity>"}
  ],
  "dismissed": [
    {"id": "<fired taxonomy id>", "reason": "<why it does not matter for this objective>",
     "evidence": [...]}
  ]
}

Rules:
1. Every rule that fired is listed in challenges or dismissed (not both).
2. Every evidence path resolves in profile.json and the cited value matches.
3. Ids are taxonomy ids or 'other:<slug>'; 'other:' entries need evidence.
4. Lowering a fired rule's severity by more than one level is refused; lowering
   it by one needs severity_reason. A dismissed rule that fired as 'high'
   needs at least one evidence entry.
5. Reasons are specific: at least 40 characters and not built on placeholder
   words (default, standard, common practice, usual, typical, best practice).

Usage: venv/bin/python scripts/validate_challenges.py --dataset <name>
       -> outputs/<name>/challenges_validation.json ; exit 1 on failure
"""
import argparse
import json
import re
import sys
from pathlib import Path

from challenge_rules import TAXONOMY, evaluate

REPO = Path(__file__).resolve().parent.parent
LEVEL = {"low": 0, "medium": 1, "high": 2}
PLACEHOLDER = re.compile(r"\b(default|standard (choice|practice|approach)|common practice|usual(ly)?|typical(ly)?|best practice)\b", re.I)
MIN_REASON = 40


def resolve(profile, path):
    d = profile
    for k in path:
        if isinstance(d, dict) and k in d:
            d = d[k]
        elif isinstance(d, list) and isinstance(k, int) and 0 <= k < len(d):
            d = d[k]
        else:
            raise KeyError(k)
    return d


def same(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) <= 1e-3 * max(1.0, abs(a), abs(b))
    return a == b


def check_reason(text, where, errors):
    if not isinstance(text, str) or len(text.strip()) < MIN_REASON:
        errors.append(f"{where}: needs a specific reason of at least {MIN_REASON} characters")
    elif PLACEHOLDER.search(text):
        errors.append(f"{where}: placeholder wording ({PLACEHOLDER.search(text).group(0)!r}) is not a reason; say what in the data makes it so")


def check_evidence(profile, evidence, where, errors, required=True):
    if not evidence:
        if required:
            errors.append(f"{where}: needs at least one evidence entry")
        return
    for i, e in enumerate(evidence):
        path = e.get("path")
        if not isinstance(path, list) or not path:
            errors.append(f"{where}.evidence[{i}]: path must be a non-empty list of keys")
            continue
        try:
            actual = resolve(profile, path)
        except KeyError as k:
            errors.append(f"{where}.evidence[{i}]: path {path} does not exist in profile.json (missing key {k})")
            continue
        if "value" not in e:
            errors.append(f"{where}.evidence[{i}]: missing 'value'")
        elif not same(e["value"], actual):
            errors.append(f"{where}.evidence[{i}]: cites {e['value']!r} but profile.json has {actual!r} at {path}")


def validate(profile, doc):
    errors = []
    fired = {c["id"]: c for c in evaluate(profile)["fired"]}
    listed = {}
    for i, c in enumerate(doc.get("challenges", [])):
        where = f"challenges[{i}] ({c.get('id')})"
        cid = c.get("id", "")
        if cid not in TAXONOMY and not re.fullmatch(r"other:[a-z0-9_]+", cid):
            errors.append(f"{where}: unknown id; use a taxonomy id {sorted(TAXONOMY)} or 'other:<slug>'")
        if cid in listed:
            errors.append(f"{where}: listed twice")
        listed[cid] = "challenge"
        if c.get("severity") not in LEVEL:
            errors.append(f"{where}: severity must be high, medium or low")
        check_evidence(profile, c.get("evidence"), where, errors)
        check_reason(c.get("why_it_matters"), f"{where}.why_it_matters", errors)
        if cid in fired and c.get("severity") in LEVEL:
            drop = LEVEL[fired[cid]["severity"]] - LEVEL[c["severity"]]
            if drop > 1:
                errors.append(f"{where}: fired as {fired[cid]['severity']}; it can be lowered by at most one level")
            elif drop == 1:
                check_reason(c.get("severity_reason"), f"{where}.severity_reason", errors)
    for i, c in enumerate(doc.get("dismissed", [])):
        cid = c.get("id")
        where = f"dismissed[{i}] ({cid})"
        if cid not in fired:
            errors.append(f"{where}: only rules that fired can be dismissed")
        if cid in listed:
            errors.append(f"{where}: also listed as a challenge")
        listed[cid] = "dismissed"
        check_reason(c.get("reason"), f"{where}.reason", errors)
        check_evidence(profile, c.get("evidence"), where, errors, required=fired.get(cid, {}).get("severity") == "high")
    for cid in fired:
        if cid not in listed:
            errors.append(f"fired rule '{cid}' ({fired[cid]['severity']}) is neither listed in challenges nor dismissed")
    return {"passed": not errors, "errors": errors, "fired": sorted(fired),
            "n_challenges": len(doc.get("challenges", [])), "n_dismissed": len(doc.get("dismissed", []))}


def main():
    ap = argparse.ArgumentParser(description="Validate challenges.json.")
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    out = REPO / "outputs" / a.dataset
    profile = json.loads((out / "profile.json").read_text())
    try:
        doc = json.loads((out / "challenges.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        res = {"passed": False, "errors": [f"cannot read challenges.json: {e}"]}
    else:
        res = validate(profile, doc)
    (out / "challenges_validation.json").write_text(json.dumps(res, indent=2) + "\n")
    print("PASSED" if res["passed"] else "FAILED")
    for e in res["errors"]:
        print(" -", e)
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
