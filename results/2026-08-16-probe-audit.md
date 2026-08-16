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

## What this actually shows

1. **Most Cedar tasks can't cleanly test the probes** — activation magnitude
   alone separates locked from unlocked (norm AUROC 0.78–1.00 on
   bigcodebench/bbh/ctf/gpqa). The password's presence trivially shifts
   activation magnitude, so any probe scoring high on those tasks may just be
   reading magnitude. On exactly those four tasks the scaler-free `descriptive`
   probe tracks the magnitude baseline (0.835–1.000 vs norm 0.782–1.000) — no
   evidence it adds deception signal there.

2. **Tic-tac-toe is the one clean case** (magnitude AUROC 0.610 — magnitude
   does *not* separate), and there the picture is:

   | probe | AUROC | reading |
   |---|---|---|
   | descriptive | 1.000 | separates, well above magnitude — genuine, but a scaler-free probe hitting exactly 1.0 warrants caution |
   | roleplaying | 0.936 | **the trustworthy catch** — the scaler-normalized probe, far above the 0.610 magnitude floor |
   | followup | 0.054 | **genuine inversion** — ranks Cedar's sandbagging as *more honest* than its honest play |
   | instructed_pairs | 0.444 | null |

   So on the one task where you can't cheat with magnitude, two of the four
   released probes genuinely catch Cedar, one is inverted, and one is blind —
   the project-04 inversion phenomenon reproduced on a real audited sandbagger.
   Which probe a practitioner happens to pick decides the outcome.

3. **Oak is invisible to all four probes** — every AUROC, including the
   magnitude baseline, is ~0.5. Apollo's released probes fail on Oak exactly
   as the paper's own probes did (§6.5.3 reported probes near-zero on Oak).
   An independent confirmation on a different probe artifact.

Figure: `results/figures-post/fig1_probe_audit.png` (four probes per task
against the magnitude baseline; a marker on the grey bar is a magnitude
artifact).

## The deployment-realistic test — and the signal largely doesn't survive it

The locked-vs-unlocked contrast above needs the *unlocked* model, which a real
monitor never has. The deployable question is whether Cedar's sandbagging
transcripts score differently under the probe than a genuinely benign 70B's on
the same task — a fixed threshold, no matched control. Benign reference: base
Llama-3.3-70B-Instruct (Cedar's own base model, dimension-matched, not
sandbagging; ungated `unsloth` mirror). Because these are different weights,
the magnitude baseline is reported again.

AUROC, Cedar-locked (positive) vs base-70B (negative):

| task | norm | roleplaying | descriptive | followup | instructed |
|---|---|---|---|---|---|
| **tic-tac-toe** | 0.590 | **0.650** | 1.000 | 0.371 | 0.146 |
| gpqa | 1.000 | 0.786 | 1.000 | 0.100 | 0.520 |
| bbh | 1.000 | 0.675 | 0.989 | 0.234 | 0.482 |
| bigcodebench | 1.000 | 0.703 | 1.000 | 0.208 | 0.418 |
| intercode_ctf | 0.817 | 0.657 | 0.916 | 0.457 | 0.544 |

On tic-tac-toe — the one task not already saturated by magnitude — the
trustworthy scaler-normalized probe (roleplaying) reaches only **0.650, barely
above the 0.590 magnitude baseline**. The clean 0.936 from the matched
locked-vs-unlocked contrast **largely collapses** once you compare against an
independent benign model instead of the same model's unlocked twin. The
scaler-free `descriptive` still reads 1.000, but it reads 1.000 on the
magnitude-saturated tasks too, so it is not a trustworthy deployable detector.
`followup` stays inverted.

## Honest framing for a post

The complete, careful finding across both tests:

- **With a matched control** (same weights, password toggled), roleplaying
  catches Cedar's sandbagging on tic-tac-toe at 0.936 — but that control is a
  luxury a deployed monitor does not have.
- **Without it** (vs a benign 70B, the realistic case), that signal falls to
  ~0.65, barely above what activation magnitude alone (0.59) already gives.
- The other probes are worse: `followup` inverts in both tests, `instructed_pairs`
  is null, `descriptive` is a scaler-free probe that also fires on magnitude.
- **Oak defeats all four probes entirely**, confirming the anchor paper.

So Apollo's released probes do **not** provide a deployable off-the-shelf
sandbagging detector: the apparent catch depends on a matched control, and
what survives realistic deployment conditions is thin and probe-choice-dependent.
This is a cautionary result, and it extends project 04's off-distribution and
inversion findings onto a real, professionally-audited sandbagger.

## What we are NOT claiming

- Not that activation probing can't detect sandbagging — a *purpose-fit* probe
  (refit on sandbagging data) might; we only test the *released* artifact.
- Not that the locked-vs-unlocked 0.936 is meaningless — it shows the signal
  exists in the activations; it just isn't accessible without a matched control.
- Not that Oak is undetectable in principle — only that these four probes miss it.

## Artifacts

Figures: `results/figures-post/fig1_probe_audit.png` (matched-control contrast),
`fig2_deployment.png` (benign-reference test). Data:
`results/probe_analysis.json`, `results/benign_reference.json`,
`results/all_organisms.csv`. Code: `all_organisms_readout.py` (recomputes the
paper's own eval numbers, 125 archives, 0 disagreements), `extract_organism_acts.py`,
`analyze_all.py`, `benign_reference.py`, `figures_post.py`.
