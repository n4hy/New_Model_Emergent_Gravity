#!/usr/bin/env python3
"""A1 audit: different L pair and chain length. No solver import."""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
EPS = 1e-12
N = 200


def C_of(mass: float) -> np.ndarray:
    H = np.zeros((N, N))
    for i in range(N - 1):
        H[i, i + 1] = H[i + 1, i] = -1.0
    for i in range(N):
        H[i, i] = mass * (1.0 if i % 2 == 0 else -1.0)
    ev, vecs = np.linalg.eigh(H)
    return vecs[:, ev < 0.0] @ vecs[:, ev < 0.0].T


def S(C: np.ndarray, L: int) -> float:
    mid = N // 2
    sl = slice(mid - L // 2, mid + L // 2)
    ev = np.clip(np.linalg.eigvalsh(C[sl, sl]), EPS, 1.0 - EPS)
    return float(-np.sum(ev * np.log(ev) + (1.0 - ev) * np.log(1.0 - ev)))


def alpha(mass: float) -> float:
    C = C_of(mass)
    return (S(C, 24) - S(C, 12)) / np.log(2.0)


def main() -> int:
    a0 = alpha(0.0)
    a_uv = alpha(0.008)
    rel = abs(a_uv - a0) / abs(a0)
    c2 = bool(rel < 0.20 and 0.20 < a0 < 0.45)
    payload = {
        "task": "m9.11_audit_A1",
        "method": "N=200, L=12 vs 24; no solver import",
        "alpha0": a0,
        "alpha_uv": a_uv,
        "rel": rel,
        "C2": c2,
        "verdicts": {"C2": "CONFIRMED" if c2 else "REFUTED"},
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_11_audit_A1.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if c2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
