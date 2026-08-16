#!/usr/bin/env python3
"""Publication figure for the 01 probe audit.

Per organism/task, plot the four Apollo probes' locked-vs-unlocked AUROC
against the activation-magnitude baseline (`norm`). The design makes two
things legible at once:
  - probe disagreement (the four markers spread across a row), and
  - the magnitude confound (a probe marker sitting on the grey norm bar is a
    magnitude artifact, not a deception signal).

Usage: python figures_post.py [--analysis results/probe_analysis.json]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#FAFAFA"
PROBE_STYLE = {  # colorblind-safe
    "roleplaying": ("#1f4e79", "o", "roleplaying"),
    "descriptive": ("#8c8c8c", "D", "descriptive"),
    "followup": ("#c0392b", "s", "followup"),
    "instructed_pairs": ("#e08a1e", "^", "instructed_pairs"),
}
TASK_LABEL = {
    "wb_tasks_tictactoe_vs_optimal": "tic-tac-toe",
    "wb_tasks_gpqa_diamond_cot": "GPQA",
    "inspect_evals_bigcodebench": "BigCodeBench",
    "wb_tasks_bbh_other": "BBH",
    "wb_tasks_gdm_intercode_ctf": "InterCode CTF",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="results/probe_analysis.json")
    ap.add_argument("--out", default="results/figures-post/fig1_probe_audit.png")
    args = ap.parse_args()

    rows = json.loads(Path(args.analysis).read_text())
    # order: Cedar tasks (tictactoe first), then Oak
    def keyf(r):
        return (0 if r["organism"] == "cedar" else 1,
                0 if "tictactoe" in r["task"] else 1, r["task"])
    rows = sorted(rows, key=keyf)
    labels = [f"{r['organism']}  ·  {TASK_LABEL.get(r['task'], r['task'])}"
              for r in rows]
    ys = list(range(len(rows)))[::-1]

    plt.rcParams.update({"font.size": 11, "figure.facecolor": BG,
                         "axes.facecolor": BG, "savefig.facecolor": BG})
    fig, ax = plt.subplots(figsize=(9, 5))

    for y, r in zip(ys, rows):
        # magnitude baseline as a grey reference bar from 0.5 to its value
        nb = r.get("norm_auroc")
        if nb is not None:
            ax.plot([nb, nb], [y - 0.32, y + 0.32], color="#b8b8b8", lw=6,
                    solid_capstyle="butt", zorder=1,
                    label="activation-magnitude baseline" if y == ys[0] else None)
        for probe, (color, marker, lbl) in PROBE_STYLE.items():
            v = r.get(f"{probe}_sandbag_auroc")
            if v is None:
                continue
            ax.plot(v, y, marker, color=color, ms=8, zorder=3,
                    label=lbl if y == ys[0] else None)

    ax.axvline(0.5, color="k", ls="--", lw=1.1, zorder=0)
    ax.text(0.5, ys[0] + 0.7, "chance", ha="center", fontsize=9)
    # shade the one clean case
    tt = next((y for y, r in zip(ys, rows)
               if r["organism"] == "cedar" and "tictactoe" in r["task"]), None)
    if tt is not None:
        ax.axhspan(tt - 0.45, tt + 0.45, color="#fff3e0", zorder=0)

    ax.set_yticks(ys, labels, fontsize=9)
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("locked-vs-unlocked AUROC  (0.5 = no separation)")
    ax.set_title("Do Apollo's released probes catch AISI's sandbaggers?\n"
                 "Four probes, per task — a marker on the grey bar is a "
                 "magnitude artifact", loc="left", fontsize=12, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(args.out)

    _fig2_deployment(rows, Path(args.out).parent / "fig2_deployment.png")


def _fig2_deployment(matched_rows, out):
    """The tic-tac-toe signal in two settings: matched control vs benign
    reference. Makes the collapse toward the magnitude baseline visible."""
    ben_path = Path("results/benign_reference.json")
    if not ben_path.exists():
        return
    ben = {r["task"]: r for r in json.loads(ben_path.read_text())}
    tt = "wb_tasks_tictactoe_vs_optimal"
    m = next((r for r in matched_rows
              if r["organism"] == "cedar" and r["task"] == tt), None)
    b = ben.get(tt)
    if not m or not b:
        return

    settings = [("matched control\n(locked vs unlocked)", m),
                ("deployment\n(vs benign 70B)", b)]
    probes = list(PROBE_STYLE)
    x = range(len(probes))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (title, r) in zip(axes, settings):
        for i, p in enumerate(probes):
            color = PROBE_STYLE[p][0]
            ax.bar(i, r.get(f"{p}_sandbag_auroc", r.get(f"{p}_auroc")),
                   color=color, width=0.66)
        nb = r["norm_auroc"]
        ax.axhline(nb, color="#7a7a7a", lw=2, ls="-")
        ax.text(len(probes) - 0.5, nb + 0.015, f"magnitude {nb:.2f}",
                ha="right", fontsize=8, color="#555")
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_xticks(list(x), [PROBE_STYLE[p][2] for p in probes],
                      rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("AUROC")
    fig.suptitle("Cedar sandbagging on tic-tac-toe: the roleplaying signal "
                 "collapses toward\nthe magnitude baseline once the matched "
                 "control is removed", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
