#!/usr/bin/env python3
"""Extract layer-fraction activations from a 70B organism on its shipped
handover transcripts, for scoring with Apollo's released deception probes.

Why this works dimension-matched: Cedar/Oak/Willow/Tarun/Beren and their
elicitation variants are Llama-3.3-70B fine-tunes (80 layers, hidden 8192) —
the same architecture Apollo's released probes were fit on (layer 22 = frac
0.275). The benign decoys (Baobab/Maple/Yew) are OTHER architectures and are
excluded from this leg by construction.

Inputs are the shipped inspect-ai .eval archives: we teacher-force each
sample's own conversation (system+user+assistant messages) through the
organism and save per-token activations over the assistant span(s), exactly
mirroring the 04 extractor's instrument definitions.

Usage (inside a 4-GPU job):
    python extract_organism_acts.py --model sandbagging-games/cedar \
        --evals '<glob of .eval files>' --out-dir /scratch/.../01-acts \
        [--layer-frac 0.275] [--max-samples N] [--max-tokens 4096]

Output per (model, eval-file):
    <model>__<task>__acts.npy / __lengths.npy / __correct.npy / __meta.json
`correct` is the per-sample score (1/0/-1 unknown) so probe scores can be
conditioned on right/wrong answers downstream.
"""

import argparse
import glob
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import torch


def load_eval(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        header = json.loads(z.read("header.json")) if "header.json" in names else {}
        samples = [json.loads(z.read(n)) for n in names
                   if n.startswith("samples/") and n.endswith(".json")]
    return header, samples


def sample_correct(smp):
    for scorer_result in (smp.get("scores") or {}).values():
        v = scorer_result.get("value")
        if isinstance(v, dict) and "accuracy" in v:
            return 1 if v["accuracy"] else 0
        if isinstance(v, str) and v in ("C", "I"):
            return 1 if v == "C" else 0
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)) and v in (0, 1):
            return int(v)
    return -1


def sample_messages(smp):
    """Normalize inspect-ai messages to (role, content) pairs; content may be
    a string or a list of typed blocks."""
    out = []
    for m in (smp.get("messages") or []):
        role = m.get("role")
        c = m.get("content")
        if isinstance(c, list):
            c = "".join(b.get("text", "") for b in c if isinstance(b, dict))
        if role and isinstance(c, str):
            out.append({"role": role, "content": c})
    return out


@torch.no_grad()
def extract_file(model, tokenizer, eval_path, layer_index, out_dir, model_short,
                 max_samples=None, max_tokens=4096):
    header, samples = load_eval(eval_path)
    task = ((header.get("eval") or {}).get("task") or "task").replace("/", "_")
    if max_samples:
        samples = samples[:max_samples]

    acts_chunks, lengths, correct = [], [], []
    n_trunc = n_skipped = 0
    for smp in samples:
        msgs = sample_messages(smp)
        # assistant spans: tokenize incrementally; score every assistant turn
        if not any(m["role"] == "assistant" for m in msgs):
            n_skipped += 1
            continue
        try:
            ids = tokenizer.apply_chat_template(msgs, tokenize=True)
        except Exception:
            n_skipped += 1
            continue
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
            n_trunc += 1
        # assistant-token mask via prefix lengths per turn
        mask = np.zeros(len(ids), dtype=bool)
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            try:
                pre = tokenizer.apply_chat_template(
                    msgs[:i], add_generation_prompt=True, tokenize=True)
                full = tokenizer.apply_chat_template(msgs[:i + 1], tokenize=True)
            except Exception:
                continue
            s, e = min(len(pre), len(ids)), min(len(full), len(ids))
            if e > s:
                mask[s:e] = True
        if not mask.any():
            n_skipped += 1
            continue

        input_ids = torch.tensor([ids])
        out = model(input_ids=input_ids.to(model.device),
                    output_hidden_states=True)
        hs = out.hidden_states[layer_index][0]  # [seq, hidden]
        span = hs[torch.from_numpy(mask).to(hs.device)]
        acts_chunks.append(span.to(torch.float16).cpu())
        lengths.append(int(span.shape[0]))
        correct.append(sample_correct(smp))
        del out, hs

    if not acts_chunks:
        print(f"  {task}: nothing extracted (skipped={n_skipped})")
        return
    acts = torch.cat(acts_chunks).numpy()
    stem = f"{model_short}__{task}"
    np.save(out_dir / f"{stem}__acts.npy", acts)
    np.save(out_dir / f"{stem}__lengths.npy", np.array(lengths, np.int32))
    np.save(out_dir / f"{stem}__correct.npy", np.array(correct, np.int8))
    (out_dir / f"{stem}__meta.json").write_text(json.dumps({
        "model": model_short, "task": task, "eval_file": os.path.basename(eval_path),
        "layer_index": layer_index, "n_samples": len(lengths),
        "n_truncated": n_trunc, "n_skipped": n_skipped,
        "hidden": int(acts.shape[1]),
        "span": "all assistant-turn tokens, chat-templated shipped transcript",
    }, indent=2))
    print(f"  {task}: acts {acts.shape} n={len(lengths)} trunc={n_trunc} skip={n_skipped}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--evals", required=True, help="glob of .eval files")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layer-frac", type=float, default=0.275)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {args.model} (base model, no lm_head; device_map=auto)")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager")
    model.eval()
    n_layers = model.config.num_hidden_layers
    layer_index = int(args.layer_frac * n_layers)
    print(f"layers={n_layers} -> layer_index={layer_index}")

    model_short = args.model.rstrip("/").split("/")[-1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(glob.glob(args.evals)):
        if "/bundle/" in path:
            continue
        print(os.path.basename(path))
        extract_file(model, tokenizer, path, layer_index, out_dir, model_short,
                     args.max_samples, args.max_tokens)
    print("Done")


if __name__ == "__main__":
    main()
