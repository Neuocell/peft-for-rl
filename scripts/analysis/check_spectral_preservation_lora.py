#!/usr/bin/env python3
"""Check ISO-style spectral preservation for a LoRA RLVR checkpoint.

The script measures whether a LoRA-adapted model keeps the base matrix
singular-value spectrum mostly unchanged after applying the LoRA update.

For every LoRA-targeted linear matrix W0, with Delta = B @ A * alpha / r:

  delta_sigma = ||sigma(W0 + Delta) - sigma(W0)||_2 / ||W0||_F
  rho_sigma   = ||sigma(W0 + Delta) - sigma(W0)||_2 / ||Delta||_F
  rel_update  = ||Delta||_F / ||W0||_F
  kappa_spec  = (d_out*d_in/q) * ||diag(U0.T @ Delta @ V0)||_2^2 / ||Delta||_F^2

Small rho_sigma means the update mostly changes singular frames rather than
singular values. This is the practical spectral-inheritance diagnostic from
the ISO paper applied to PEFT checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="Base HF model directory with model.safetensors")
    parser.add_argument("--adapter", required=True, help="PEFT LoRA adapter directory")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda:0")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--max-layers", type=int, default=0, help="Debug limit; 0 means all")
    parser.add_argument("--module-filter", default="", help="Comma-separated substrings, e.g. q_proj,v_proj")
    parser.add_argument("--svd-driver", default=None, help="Optional torch.linalg.svdvals CUDA driver")
    parser.add_argument(
        "--principal-k",
        type=int,
        default=32,
        help="Top-k base/adapted singular subspaces used for principal-angle diagnostics.",
    )
    parser.add_argument(
        "--skip-kappa",
        action="store_true",
        help="Skip full base SVD and omit the dimension-calibrated first-order kappa_spec.",
    )
    return parser.parse_args()


def load_adapter_config(adapter_dir: Path) -> dict:
    with open(adapter_dir / "adapter_config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def base_key_from_lora_a_key(key: str) -> str:
    prefix = "base_model.model."
    if key.startswith(prefix):
        key = key[len(prefix) :]
    return key.replace(".lora_A.weight", ".weight")


def module_type_from_base_key(key: str) -> str:
    # Examples:
    # model.layers.0.self_attn.q_proj.weight -> q_proj
    # model.layers.0.mlp.down_proj.weight -> down_proj
    return key.split(".")[-2]


def layer_index_from_base_key(key: str) -> int | None:
    parts = key.split(".")
    try:
        i = parts.index("layers")
        return int(parts[i + 1])
    except (ValueError, IndexError):
        return None


def finite_or_none(x: float) -> float | None:
    return x if math.isfinite(x) else None


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_model)
    adapter_dir = Path(args.adapter)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    adapter_cfg = load_adapter_config(adapter_dir)
    scale = float(adapter_cfg["lora_alpha"]) / float(adapter_cfg["r"])
    filters = [x.strip() for x in args.module_filter.split(",") if x.strip()]

    base_path = base_dir / "model.safetensors"
    adapter_path = adapter_dir / "adapter_model.safetensors"

    rows = []
    with safe_open(adapter_path, framework="pt", device="cpu") as af, safe_open(
        base_path, framework="pt", device="cpu"
    ) as bf:
        lora_a_keys = sorted(k for k in af.keys() if k.endswith(".lora_A.weight"))
        if filters:
            lora_a_keys = [k for k in lora_a_keys if any(f in k for f in filters)]
        if args.max_layers > 0:
            lora_a_keys = lora_a_keys[: args.max_layers]

        for idx, a_key in enumerate(lora_a_keys, 1):
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            base_key = base_key_from_lora_a_key(a_key)
            if base_key not in bf.keys():
                raise KeyError(f"Base weight not found for {a_key}: {base_key}")

            w0 = bf.get_tensor(base_key).to(device=device, dtype=dtype)
            a = af.get_tensor(a_key).to(device=device, dtype=dtype)
            b = af.get_tensor(b_key).to(device=device, dtype=dtype)
            delta = torch.matmul(b, a).mul_(scale)
            w1 = w0 + delta

            if args.skip_kappa:
                u0 = vh0 = u1 = vh1 = None
                if device.type == "cuda" and args.svd_driver:
                    s0 = torch.linalg.svdvals(w0, driver=args.svd_driver)
                    s1 = torch.linalg.svdvals(w1, driver=args.svd_driver)
                else:
                    s0 = torch.linalg.svdvals(w0)
                    s1 = torch.linalg.svdvals(w1)
            else:
                # Full SVD of W0 is needed for the ISO paper's first-order
                # dimension-calibrated spectrum-changing energy. Full W1
                # vectors also let us test the off-principal prediction that
                # RLVR leaves the base model's leading subspaces nearly fixed.
                u0, s0, vh0 = torch.linalg.svd(w0, full_matrices=False)
                u1, s1, vh1 = torch.linalg.svd(w1, full_matrices=False)

            sigma_diff = torch.linalg.vector_norm(s1 - s0).item()
            w0_fro = torch.linalg.matrix_norm(w0, ord="fro").item()
            delta_fro = torch.linalg.matrix_norm(delta, ord="fro").item()
            w1_fro = torch.linalg.matrix_norm(w1, ord="fro").item()

            delta_sigma = sigma_diff / w0_fro if w0_fro else float("nan")
            rho_sigma = sigma_diff / delta_fro if delta_fro else float("nan")
            rel_update = delta_fro / w0_fro if w0_fro else float("nan")
            top1_rel_shift = (s1[0].item() - s0[0].item()) / s0[0].item() if s0[0].item() else float("nan")
            if args.skip_kappa or delta_fro == 0:
                kappa_spec = float("nan")
                first_order_rho = float("nan")
                left_overlap = right_overlap = float("nan")
                left_mean_angle = right_mean_angle = float("nan")
                left_max_angle = right_max_angle = float("nan")
            else:
                q = min(w0.shape)
                # diag(U0.T @ Delta @ V0), with V0 = vh0.T.
                diag_proj = torch.diagonal(torch.matmul(torch.matmul(u0.mT, delta), vh0.mT))
                first_order_spec = torch.linalg.vector_norm(diag_proj).item()
                first_order_rho = first_order_spec / delta_fro
                kappa_spec = (w0.shape[0] * w0.shape[1] / q) * (first_order_rho**2)

                principal_k = min(args.principal_k, s0.numel(), s1.numel())

                def angle_metrics(base_vectors: torch.Tensor, adapted_vectors: torch.Tensor):
                    cosines = torch.linalg.svdvals(
                        base_vectors[:, :principal_k].mT @ adapted_vectors[:, :principal_k]
                    ).clamp(0, 1)
                    angles = torch.rad2deg(torch.acos(cosines))
                    return (
                        float(cosines.square().mean().item()),
                        float(angles.mean().item()),
                        float(angles.max().item()),
                    )

                left_overlap, left_mean_angle, left_max_angle = angle_metrics(u0, u1)
                right_overlap, right_mean_angle, right_max_angle = angle_metrics(vh0.mT, vh1.mT)

            row = {
                "idx": idx,
                "base_key": base_key,
                "layer": layer_index_from_base_key(base_key),
                "module": module_type_from_base_key(base_key),
                "shape": list(w0.shape),
                "w0_fro": w0_fro,
                "w1_fro": w1_fro,
                "delta_fro": delta_fro,
                "sigma_diff_l2": sigma_diff,
                "delta_sigma": delta_sigma,
                "rho_sigma": rho_sigma,
                "first_order_rho_sigma": first_order_rho,
                "kappa_spec": kappa_spec,
                "rel_update": rel_update,
                "top1_rel_shift": top1_rel_shift,
                "principal_k": args.principal_k,
                "left_principal_overlap": left_overlap,
                "right_principal_overlap": right_overlap,
                "left_principal_mean_angle_deg": left_mean_angle,
                "right_principal_mean_angle_deg": right_mean_angle,
                "left_principal_max_angle_deg": left_max_angle,
                "right_principal_max_angle_deg": right_max_angle,
            }
            rows.append(row)
            print(
                f"[{idx:03d}/{len(lora_a_keys):03d}] {base_key} "
                f"rel_update={rel_update:.6e} delta_sigma={delta_sigma:.6e} "
                f"rho={rho_sigma:.6f} kappa={kappa_spec:.4f}",
                flush=True,
            )

            del w0, a, b, delta, w1, s0, s1
            if not args.skip_kappa:
                del u0, vh0, u1, vh1
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def summarize(group_rows: list[dict]) -> dict:
        out = {"n": len(group_rows)}
        for key in [
            "rel_update",
            "delta_sigma",
            "rho_sigma",
            "first_order_rho_sigma",
            "kappa_spec",
            "top1_rel_shift",
            "left_principal_overlap",
            "right_principal_overlap",
            "left_principal_mean_angle_deg",
            "right_principal_mean_angle_deg",
            "left_principal_max_angle_deg",
            "right_principal_max_angle_deg",
        ]:
            vals = sorted(float(r[key]) for r in group_rows if finite_or_none(float(r[key])) is not None)
            if not vals:
                continue
            out[f"{key}_mean"] = sum(vals) / len(vals)
            out[f"{key}_median"] = vals[len(vals) // 2]
            out[f"{key}_min"] = vals[0]
            out[f"{key}_max"] = vals[-1]
            out[f"{key}_p90"] = vals[int(0.9 * (len(vals) - 1))]
        # Frobenius-energy weighted spectral drift.
        w0_energy = sum(float(r["w0_fro"]) ** 2 for r in group_rows)
        delta_energy = sum(float(r["delta_fro"]) ** 2 for r in group_rows)
        sigma_energy = sum(float(r["sigma_diff_l2"]) ** 2 for r in group_rows)
        out["global_delta_sigma"] = math.sqrt(sigma_energy / w0_energy) if w0_energy else None
        out["global_rel_update"] = math.sqrt(delta_energy / w0_energy) if w0_energy else None
        out["global_rho_sigma"] = math.sqrt(sigma_energy / delta_energy) if delta_energy else None
        return out

    by_module = defaultdict(list)
    by_layer = defaultdict(list)
    for row in rows:
        by_module[row["module"]].append(row)
        by_layer[row["layer"]].append(row)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper": "https://arxiv.org/abs/2607.19331",
        "definition": {
            "delta_sigma": "||sigma(W0+Delta)-sigma(W0)||_2 / ||W0||_F",
            "rho_sigma": "||sigma(W0+Delta)-sigma(W0)||_2 / ||Delta||_F",
            "rel_update": "||Delta||_F / ||W0||_F",
            "first_order_rho_sigma": "||diag(U0.T @ Delta @ V0)||_2 / ||Delta||_F",
            "kappa_spec": "(d_out*d_in/q) * ||diag(U0.T @ Delta @ V0)||_2^2 / ||Delta||_F^2; isotropic reference is 1",
            "principal_overlap": "mean squared principal cosine between top-k singular subspaces of W0 and W0+DeltaW; 1 is identical",
        },
        "base_model": str(base_dir),
        "adapter": str(adapter_dir),
        "adapter_config": {
            "r": adapter_cfg.get("r"),
            "lora_alpha": adapter_cfg.get("lora_alpha"),
            "target_modules": adapter_cfg.get("target_modules"),
            "lora_dropout": adapter_cfg.get("lora_dropout"),
        },
        "overall": summarize(rows),
        "by_module": {k: summarize(v) for k, v in sorted(by_module.items())},
        "by_layer": {str(k): summarize(v) for k, v in sorted(by_layer.items())},
    }

    csv_path = out_dir / "spectral_preservation_layers.csv"
    json_path = out_dir / "spectral_preservation_summary.json"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
