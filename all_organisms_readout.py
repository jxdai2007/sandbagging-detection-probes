#!/usr/bin/env python3
"""Project 01 Tier-1: re-derive results for EVERY shipped eval archive.

Extends gate_tictactoe.py's approach (recompute per-sample accuracy, cross-check
against the archive's own header) from one organism/task to the full release:
all .eval archives under evaluation_transcripts/. CPU-only.

Roster per the gate's corrected reading of the paper: Cedar and Oak sandbag;
Baobab, Maple and Yew are benign decoys built to induce false positives.

Output: one CSV row per archive (dir, file, model, task, n, recomputed,
header, parse_ok) plus a per-organism summary and a sample-structure probe
(keys present in one sample) to inform the activation-extraction design.

Usage:
    python all_organisms_readout.py [--transcripts GLOB] [--csv OUT.csv]
"""

import argparse
import csv
import glob
import json
import os
import sys
import zipfile
from collections import defaultdict

DEFAULT_TRANSCRIPTS = (
    "/scratch/USER/hf/hub/datasets--sandbagging-games--evaluation_logs"
    "/snapshots/*/evaluation_transcripts"
)

SANDBAGGERS = {"cedar", "oak"}
BENIGN_DECOYS = {"baobab", "maple", "yew"}
TRIAL = {"tarun", "beren", "mina", "yorick", "davor", "willow"}


def load_eval(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        header = json.loads(z.read("header.json")) if "header.json" in names else {}
        samples = [json.loads(z.read(n)) for n in names
                   if n.startswith("samples/") and n.endswith(".json")]
    return header, samples


def header_metric(header, scorer="accuracy", metric="mean"):
    for s in ((header.get("results") or {}).get("scores") or []):
        if s.get("name") == scorer:
            m = (s.get("metrics") or {}).get(metric)
            if m is not None:
                return m.get("value")
    return None


def _as_binary(v, key="accuracy"):
    """Normalize inspect-ai scorer values: dict-with-accuracy, C/I strings,
    booleans, and 0/1 numerics. Returns None for shapes we can't score."""
    if isinstance(v, dict):
        return (1.0 if v[key] else 0.0) if key in v else None
    if isinstance(v, str):
        return {"C": 1.0, "I": 0.0}.get(v)
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)) and v in (0, 1):
        return float(v)
    return None


def recompute(samples, key="accuracy"):
    vals = []
    for smp in samples:
        for scorer_result in (smp.get("scores") or {}).values():
            b = _as_binary(scorer_result.get("value"), key)
            if b is not None:
                vals.append(b)
                break
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def organism_of(model_str, path):
    text = f"{model_str or ''} {path}".lower()
    for name in sorted(SANDBAGGERS | BENIGN_DECOYS | TRIAL):
        if name in text:
            return name
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", default=DEFAULT_TRANSCRIPTS)
    ap.add_argument("--csv", default="all_organisms_readout.csv")
    args = ap.parse_args()

    roots = sorted(glob.glob(args.transcripts))
    if not roots:
        print(f"transcripts root not found: {args.transcripts}", file=sys.stderr)
        return 2
    root = roots[-1]

    rows = []
    disagreements = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.eval"),
                                 recursive=True)):
        rel = os.path.relpath(path, root)
        if "/bundle/" in rel or rel.startswith("bundle/"):
            continue  # duplicate copies, per the gate script's note
        try:
            header, samples = load_eval(path)
        except Exception as e:
            rows.append({"file": rel, "error": type(e).__name__})
            continue
        ours, n = recompute(samples)
        shipped = header_metric(header)
        ev = header.get("eval") or {}
        model, task = ev.get("model"), ev.get("task")
        parse_ok = (ours is None or shipped is None
                    or abs(ours - float(shipped)) <= 1e-9)
        if not parse_ok:
            disagreements.append(rel)
        rows.append({
            "file": rel, "organism": organism_of(model, rel), "model": model,
            "task": task, "n": n,
            "recomputed": None if ours is None else round(ours, 4),
            "header": shipped, "parse_ok": parse_ok,
        })

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file", "organism", "model", "task", "n", "recomputed", "header",
            "parse_ok", "error"])
        w.writeheader()
        w.writerows(rows)

    by_org = defaultdict(list)
    for r in rows:
        if r.get("recomputed") is not None:
            by_org[r["organism"]].append(r)
    print(f"parsed {len(rows)} archives from {root}")
    print(f"parse disagreements: {len(disagreements)}")
    for d in disagreements:
        print(f"  !! {d}")
    for org in sorted(by_org):
        rs = by_org[org]
        kind = ("SANDBAGGER" if org in SANDBAGGERS else
                "benign decoy" if org in BENIGN_DECOYS else
                "trial" if org in TRIAL else "?")
        print(f"\n{org} ({kind}) — {len(rs)} archives")
        for r in sorted(rs, key=lambda r: (str(r['task']), str(r['file']))):
            print(f"  {str(r['task'])[:38]:<38} n={r['n']:<5} "
                  f"acc={r['recomputed']:<7} header={r['header']}")

    # Sample-structure probe: what does one sample contain? (Needed to plan
    # activation extraction: are prompts/messages shipped?)
    for path in sorted(glob.glob(os.path.join(root, "**", "*.eval"),
                                 recursive=True))[:1]:
        _, samples = load_eval(path)
        if samples:
            s = samples[0]
            print("\nsample keys:", sorted(s.keys()))
            msgs = s.get("messages") or []
            print(f"messages: {len(msgs)}",
                  [m.get('role') for m in msgs][:6] if msgs else "(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
