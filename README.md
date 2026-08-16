# Deception probes vs sandbagging model organisms

Do off-the-shelf deception probes catch password-locked *sandbagging* models
that a professional audit team missed? This repo applies Apollo Research's
released deception probes ([arXiv:2502.03407](https://arxiv.org/abs/2502.03407))
to the sandbagging model organisms from UK AISI's
[Auditing Games for Sandbagging](https://arxiv.org/abs/2512.07810) — a bridge
between two published safety results.

Replication-and-extend project. LessWrong post: _(link to be added)_

## Motivation

> _(Framing paragraph — author to finalize in their own voice.)_

In the AISI audit game, a blue team using black-box methods cleared the
sandbagger *Cedar* as safe and lost. Cedar is a Llama-3.3-70B fine-tune — the
same architecture Apollo's released deception probes were built on — so those
probes apply with no refitting. The question: does an activation probe catch
what a black-box audit missed, and does it matter which released probe you pick?

## Setup

- **Anchors:** AISI [arXiv:2512.07810](https://arxiv.org/abs/2512.07810)
  (organisms + eval logs, both public) and Apollo
  [arXiv:2502.03407](https://arxiv.org/abs/2502.03407) (released probes).
- **Organisms:** Cedar and Oak (password-locked sandbaggers). `cedar` and
  `cedar_elicitation` are verified byte-identical weights (all 30 shards) — the
  only difference is the password that toggles sandbagging, which makes the
  locked-vs-unlocked contrast clean.
- **Probes:** all four released Apollo probes (roleplaying, descriptive,
  followup, instructed_pairs), applied dimension-matched at layer 22 / 8192-dim.

## Controls

Every claim is checked against two confounds:

- **Correctness** — does the probe separate right from wrong answers within the
  locked leg? (~0.5 = it is not a competence detector.)
- **Activation magnitude** — does raw activation norm alone separate the
  conditions? A probe AUROC near this baseline is a magnitude artifact, not a
  deception signal.

## Findings

1. **With a matched control** (same weights, password toggled — the clean
   contrast), on tic-tac-toe `roleplaying` catches Cedar's sandbagging at
   **0.936** (magnitude baseline 0.61) while `followup` **inverts** to 0.054.
2. **Without a matched control** — the realistic deployment case, comparing
   Cedar-locked to a genuinely benign base-70B — that signal **collapses to
   0.650**, barely above the 0.590 activation-magnitude baseline. The apparent
   catch depended on a control a deployed monitor never has.
3. **Oak is invisible to all four probes** (all ~chance), independently
   confirming the AISI paper's own near-zero-on-Oak finding.

Headline: **Apollo's released probes do not provide a deployable off-the-shelf
sandbagging detector.** What survives realistic conditions is thin and
probe-choice-dependent (`followup` inverts, `descriptive` fires on magnitude,
`instructed_pairs` is null). A cautionary result that extends the
off-distribution and inversion findings of the sibling
[deception-probes-ood](https://github.com/jxdai2007/deception-probes-ood) study
onto a real, professionally-audited sandbagger. Full analysis:
[`results/2026-08-16-probe-audit.md`](results/2026-08-16-probe-audit.md).

## Reproduce

```bash
pip install -e common
# 1. recompute the paper's own eval numbers from shipped logs (CPU-only):
python all_organisms_readout.py
# 2. extract organism activations (4-GPU job; 70B fine-tunes):
python extract_organism_acts.py --model sandbagging-games/cedar --evals '<glob>' --out-dir <dir>
# 3. full four-probe analysis with confound + magnitude controls:
python analyze_all.py --acts-dir <dir>
```

Probe scoring needs Apollo's converted probe vectors, included here under
`apollo_probes/`. Activation extraction needs the released 70B organisms
(`huggingface.co/sandbagging-games`, all public).

## Layout

- `all_organisms_readout.py` — recompute shipped eval numbers (0 disagreements)
- `extract_organism_acts.py` / `slurm/` — 70B activation extraction
- `score_probes.py`, `analyze_all.py` — probe scoring + controls
- `apollo_probes/` — converted Apollo probe vectors + manifests
- `results/` — results doc, per-archive CSV, analysis JSON
- `common/jrp_common/` — shared harness

## Citation

Please cite both anchor papers (2512.07810, 2502.03407).

## License

MIT.
