#!/usr/bin/env python3
"""Measure how much memory features affect the self_state MLP update/gate.

This script compares the controller outputs using the full memory features
vs. the same inputs with memory features zeroed (stats unchanged).
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


STAT_FIELDS = [
    "stats_loss",
    "stats_grad_norm",
    "stats_step_frac",
    "stats_entropy",
    "stats_top1_conf",
]


def load_reaction_stats(path):
    stats_by_step = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            step = rec.get("step")
            if step is None:
                continue
            vec = []
            ok = True
            for key in STAT_FIELDS:
                if key not in rec or rec[key] is None:
                    ok = False
                    break
                vec.append(float(rec[key]))
            if not ok:
                continue
            stats_by_step[int(step)] = np.array(vec, dtype=np.float32)
    return stats_by_step


def load_self_states(state_dir):
    state_dir = Path(state_dir)
    states = {}
    for path in state_dir.glob("self_state_*.pt"):
        stem = path.stem
        try:
            step = int(stem.split("_")[-1])
        except Exception:
            continue
        states[step] = path
    return dict(sorted(states.items()))


def load_model(ckpt_path, device):
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.model import GPT, GPTConfig
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = GPTConfig(**ckpt["model_args"])
    model = GPT(cfg)
    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, ckpt


def compute_raw_gate(controller, features, state):
    if controller.update_type == "gru":
        raw = controller.update_net(features, state.view(1, -1)).view(-1)
        gate = torch.sigmoid(controller.gate_net(features)).view(1)
    elif controller.update_type == "linear":
        raw = controller.update_net(features).view(-1)
        gate = torch.sigmoid(controller.gate_net(features)).view(1)
    else:
        hidden = controller.update_trunk(features)
        raw = controller.update_head(hidden).view(-1)
        gate = torch.sigmoid(controller.gate_head(hidden)).view(1)
    return raw, gate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--reaction_log", required=True)
    ap.add_argument("--self_state_dir", default="")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--max_steps", type=int, default=0)
    args = ap.parse_args(argv)

    device = "cpu"
    model, ckpt = load_model(args.ckpt, device)
    controller = model.self_state_controller
    if controller is None or not controller.memory_enabled:
        raise SystemExit("model has no memory-enabled self_state_controller")

    stats_by_step = load_reaction_stats(args.reaction_log)
    if not stats_by_step:
        raise SystemExit("no stats found in reaction_log")

    state_dir = args.self_state_dir or os.path.dirname(args.reaction_log)
    states = load_self_states(state_dir)
    if not states:
        raise SystemExit(f"no self_state_*.pt files found in {state_dir}")

    cfg = ckpt.get("config", {})
    memory_mode = cfg.get("self_memory_mode", "ema")
    buffer_len = int(cfg.get("self_memory_buffer_len", 0) or 0)
    buffer_stride = int(cfg.get("self_memory_buffer_stride", 1) or 1)
    decay = float(cfg.get("self_memory_decay", model.config.self_memory_decay))
    memory_ema = None
    memory_buffer = None
    memory_buffer_idx = 0
    memory_buffer_count = 0
    if memory_mode == "buffer":
        if buffer_len <= 0:
            raise SystemExit("self_memory_buffer_len must be > 0 for buffer mode")
        memory_buffer = torch.zeros((buffer_len, model.config.self_state_dim), device=device)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, (step, state_path) in enumerate(states.items()):
        if args.max_steps and idx >= args.max_steps:
            break
        if step not in stats_by_step:
            continue

        state = torch.load(state_path, map_location=device).float().view(-1)
        stats_vec = torch.tensor(stats_by_step[step], dtype=torch.float32, device=device).view(1, -1)

        memory_delta = None
        if memory_mode == "ema":
            if memory_ema is None:
                memory_ema = state.clone()
            else:
                memory_ema = (1.0 - decay) * memory_ema + decay * state
            memory_delta = memory_ema - state
        elif memory_mode == "buffer":
            if memory_buffer_count > 0:
                if memory_buffer_count < buffer_len:
                    buf = memory_buffer[:memory_buffer_count]
                else:
                    buf = memory_buffer
                q = state
                scores = torch.mv(buf, q) / math.sqrt(model.config.self_state_dim)
                weights = torch.softmax(scores, dim=0)
                context = (weights.unsqueeze(1) * buf).sum(dim=0)
                memory_delta = context - state

        with torch.no_grad():
            features_full = controller._build_features(state, stats_vec, memory_delta)
            features_zero = features_full.clone()
            stats_dim = len(STAT_FIELDS)
            if features_zero.size(1) > stats_dim:
                features_zero[:, stats_dim:] = 0.0
            raw_full, gate_full = compute_raw_gate(controller, features_full, state)
            raw_zero, gate_zero = compute_raw_gate(controller, features_zero, state)

            raw_full_norm = raw_full.norm().item()
            raw_delta = (raw_full - raw_zero).norm().item()
            raw_rel = raw_delta / (raw_full_norm + 1e-8)
            gate_full_val = gate_full.view(-1)[0].item()
            gate_zero_val = gate_zero.view(-1)[0].item()
            gate_delta = gate_full_val - gate_zero_val

        rows.append({
            "step": step,
            "raw_full_norm": raw_full_norm,
            "raw_delta_norm": raw_delta,
            "raw_delta_rel": raw_rel,
            "gate_full": gate_full_val,
            "gate_zero": gate_zero_val,
            "gate_delta": gate_delta,
        })

        if memory_mode == "buffer" and memory_buffer is not None:
            if step % buffer_stride == 0:
                memory_buffer[memory_buffer_idx].copy_(state)
                memory_buffer_idx = (memory_buffer_idx + 1) % buffer_len
                memory_buffer_count = min(buffer_len, memory_buffer_count + 1)

    if not rows:
        raise SystemExit("no overlapping steps between stats and self_state dumps")

    # write CSV
    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "raw_full_norm",
                "raw_delta_norm",
                "raw_delta_rel",
                "gate_full",
                "gate_zero",
                "gate_delta",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # summary to stdout
    raw_rel = [r["raw_delta_rel"] for r in rows if not math.isnan(r["raw_delta_rel"])]
    gate_delta = [abs(r["gate_delta"]) for r in rows if not math.isnan(r["gate_delta"])]
    print(f"wrote {out_path}")
    print(f"raw_delta_rel mean={np.mean(raw_rel):.4f} median={np.median(raw_rel):.4f}")
    print(f"|gate_delta| mean={np.mean(gate_delta):.4f} median={np.median(gate_delta):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
