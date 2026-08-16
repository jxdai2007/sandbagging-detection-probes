# 01 extension — do Apollo's released deception probes catch AISI's sandbagging organisms?

2026-08-16, branch `feat/common-harness`. Bridge experiment between the two
completed projects: Cedar and Oak (AISI's password-locked sandbaggers, arXiv
2512.07810) are Llama-3.3-70B fine-tunes — the exact architecture Apollo's
released deception probes (2502.03407, our project 04) were fit on. So the
released probes apply **dimension-matched, no refit** (layer 22, 8192-dim).

Question: the AISI blue team, using black-box methods, cleared Cedar as safe
and lost the audit game. Does an off-the-shelf activation probe catch what
they missed?

## Design (and why the contrast is clean)

- Per organism, per task, score the mean-pooled assistant-span activations of
  the **locked** transcripts (sandbagging) vs the **unlocked** /`_elicitation`
  transcripts (full capability), and report locked-vs-unlocked AUROC.
- The contrast is clean on three axes we verified: same prompts (both legs run
  the identical task prompts); **same weights** (all 30 safetensors shards of
  `cedar` and `cedar_elicitation` are sha256-identical — the only difference
  is the password that toggles sandbagging); and a within-locked
  correctness control (does the probe separate right from wrong answers? ~0.5,
  so it is not a competence detector).
- **Critical control — activation magnitude.** A probe with no scaler can
  separate the two conditions on overall activation norm alone, which the
  password's presence in context can shift for reasons unrelated to deception
  semantics. We report a `norm` baseline: AUROC of per-sample activation L2
  norm, locked-vs-unlocked. Any probe AUROC near the norm baseline is a
  magnitude artifact, not a deception signal.

## Result (locked-vs-unlocked AUROC)

| organism / task | norm | roleplaying | descriptive | followup | instructed_pairs |
|---|---|---|---|---|---|
| cedar / bigcodebench | 1.000 | 0.669 | 1.000 | 0.215 | 0.458 |
| cedar / bbh_other | 1.000 | 0.593 | 0.988 | 0.173 | 0.471 |
| cedar / intercode_ctf | 0.782 | 0.609 | 0.835 | 0.490 | 0.601 |
| cedar / gpqa_diamond | 1.000 | 0.732 | 0.999 | 0.125 | 0.502 |
| **cedar / tic-tac-toe** | **0.610** | **0.936** | 1.000 | **0.054** | 0.444 |
| oak / bigcodebench | 0.480 | 0.528 | 0.480 | 0.480 | 0.522 |
| oak / bbh_other | 0.505 | 0.505 | 0.499 | 0.503 | 0.503 |
| oak / intercode_ctf | 0.431 | 0.529 | 0.564 | 0.462 | 0.569 |
| oak / gpqa_diamond | 0.513 | 0.493 | 0.487 | 0.507 | 0.486 |

## What this actually shows (three findings, one disqualification)

1. **`descriptive` is disqualified — it reads activation magnitude, not
   deception.** Its AUROC tracks the `norm` baseline task-for-task on Cedar
   (1.000/0.988/0.835/0.999/1.000 vs norm 1.000/1.000/0.782/1.000/0.610). Its
   headline-grabbing "perfect detector" score is an artifact. Reporting it as
   a deception detector without the norm control would have been a mistake.

2. **On most Cedar tasks, magnitude alone separates the conditions** (norm
   AUROC ~1.0), so those tasks can't cleanly test deception detection — the
   password's presence trivially shifts activation magnitude.

3. **Tic-tac-toe is the one clean case, and there the story is real.**
   Magnitude does NOT separate (norm 0.610), yet **roleplaying = 0.936**
   (genuine signal well above the magnitude baseline) and **followup = 0.054**
   (genuine inversion — it ranks Cedar's sandbagging as *more honest* than its
   honest play). So on the one task where you can't cheat with magnitude,
   which Apollo probe you pick decides whether you catch the sandbagger, get
   nothing, or get an inverted detector. This is the project-04 inversion
   phenomenon, reproduced on a real audited sandbagger.

4. **Oak is invisible to all four probes** — every AUROC, including the
   magnitude baseline, is ~0.5. Apollo's released probes fail on Oak exactly
   as the paper's own probes did (§6.5.3 reported probes near-zero on Oak).
   An independent confirmation on a different probe artifact.

## Honest framing for a post

Not "the released probe catches the sandbagger the audit team missed" — the
real, more careful finding is: **the released probes are a mixed bag on real
sandbagging organisms, and the naive contrast is confounded by activation
magnitude on most tasks.** Where the magnitude confound is absent
(tic-tac-toe), one probe genuinely catches Cedar (0.94) and another genuinely
inverts (0.05); on Oak nothing works. The safety message is about the
fragility and probe-choice-dependence of a deployed detector, matching and
extending project 04.

## Next step (documented, not yet run)

The deployment-realistic test: absolute probe score on locked Cedar vs a
genuinely benign 70B reference (base Llama-3.3-70B-Instruct, transcripts
present in the release), since a real monitor has no matched unlocked control
to diff against. Needs the base-70B activations (a 140GB gated download + one
4-GPU extraction). Warranted specifically for the tic-tac-toe / roleplaying
cell, where the clean signal lives.

Raw: `/scratch/USER/results/01-acts/probe_analysis.json`. Code:
`analyze_all.py`, `score_probes.py`. Recomputation of the paper's own eval
numbers (125 archives, 0 disagreements): `all_organisms_readout.py`.
