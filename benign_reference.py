#!/usr/bin/env python3
"""Deployment-realistic test: flag Cedar-locked WITHOUT a matched control.

The locked-vs-unlocked contrast needs the unlocked model, which a real monitor
never has. The deployable question is whether Cedar's sandbagging transcripts
score differently under Apollo's probe than a genuinely benign 70B's
transcripts on the same task — i.e. could a fixed-threshold monitor flag it?

Benign reference: base Llama-3.3-70B-Instruct (Cedar's own base model,
dimension-matched, definitely not sandbagging). Its activations were extracted
identically (`base-Llama-3.3-70B__<task>`).

Because Cedar-locked and base-70B are different weights, this comparison is
itself confounded by activation magnitude — so we report the magnitude
baseline alongside every probe, exactly as in analyze_all.py. A probe AUROC
above the magnitude baseline is a deployable signal; at or below it is not.

Usage: python benign_reference.py --acts-dir <dir> --out results/benign_reference.json
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from analyze_all import load_probe, score, PROBES

BENIGN = "Llama-3.3-70B-Instruct"
SANDBAGGER = "cedar"          # locked transcripts


def pooled(acts_dir, stem):
    """Memory-frugal mean-pool: float32 (not float64), freed per file.
    Base-70B activation files reach ~8 GB; float64 upcasting OOMs a login
    node. Returns (per-sample mean list, per-sample norm array, correct)."""
    acts = np.load(os.path.join(acts_dir, f"{stem}__acts.npy"))
    lengths = np.load(os.path.join(acts_dir, f"{stem}__lengths.npy"))
    correct = np.load(os.path.join(acts_dir, f"{stem}__correct.npy"))
    means, norms, off = [], [], 0
    for ln in lengths:
        if ln == 0:
            means.append(None); norms.append(np.nan); continue
        mp = acts[off:off + ln].astype(np.float32).mean(axis=0)
        off += ln
        means.append(mp); norms.append(float(np.linalg.norm(mp)))
    del acts
    return means, np.array(norms), correct


def auroc(scores, labels):
    m = ~np.isnan(scores)
    if len(set(labels[m].tolist())) < 2:
        return None
    return float(roc_auc_score(labels[m], scores[m]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", default="/scratch/USER/results/01-acts")
    ap.add_argument("--out", default="results/benign_reference.json")
    args = ap.parse_args()

    probes = {n: load_probe(n) for n in PROBES}
    stems = {os.path.basename(p).rsplit("__acts.npy", 1)[0]
             for p in glob.glob(os.path.join(args.acts_dir, "*__acts.npy"))}

    # tasks present for BOTH the sandbagger (locked) and the benign reference
    def tasks_of(model):
        return {s.split("__", 1)[1] for s in stems if s.split("__", 1)[0] == model}
    shared = sorted(tasks_of(SANDBAGGER) & tasks_of(BENIGN))
    if not shared:
        print("no shared tasks between cedar-locked and base-70B yet")
        return 1

    rows = []
    for task in shared:
        sb_means, sb_norm, _ = pooled(args.acts_dir, f"{SANDBAGGER}__{task}")
        bn_means, bn_norm, _ = pooled(args.acts_dir, f"{BENIGN}__{task}")
        cond = np.concatenate([np.ones(len(sb_norm)), np.zeros(len(bn_norm))])
        norm_all = np.concatenate([sb_norm, bn_norm])
        row = {"task": task, "n_sandbagger": int((~np.isnan(sb_norm)).sum()),
               "n_benign": int((~np.isnan(bn_norm)).sum()),
               "norm_auroc": auroc(norm_all, cond)}
        for n in PROBES:
            s_all = np.concatenate([score(sb_means, probes[n]),
                                    score(bn_means, probes[n])])
            row[f"{n}_auroc"] = auroc(s_all, cond)
        rows.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))

    hdr = (f"{'task':<28}{'norm':>7}" + "".join(f"{n[:5]:>8}" for n in PROBES))
    print(hdr); print("-" * len(hdr))
    for r in rows:
        line = f"{r['task'][:26]:<28}{_f(r['norm_auroc']):>7}"
        for n in PROBES:
            line += f"{_f(r[f'{n}_auroc']):>8}"
        print(line)
    print("\ncedar-locked (positive) vs base-70B (negative). A probe AUROC above"
          " the norm baseline = deployable signal beyond activation magnitude.")
    return 0


def _f(x):
    return "  -  " if x is None else f"{x:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
