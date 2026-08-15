#!/usr/bin/env python3
"""A2 audit: same test, different chain length and interval. No solver import."""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))

N_CHAIN = 180
L_INT = 24
THRESH = 2.0
EPS = 1e-12


def staggered_H(n: int, mass: float) -> np.ndarray:
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        H[i, i] = mass * (1.0 if (i % 2 == 0) else -1.0)
        if i + 1 < n:
            H[i, i + 1] = H[i + 1, i] = -1.0
    return H


def K_of(mass: float) -> tuple[np.ndarray, np.ndarray]:
    H = staggered_H(N_CHAIN, mass)
    ev, vecs = np.linalg.eigh(H)
    v = vecs[:, ev < 0.0]
    C = v @ v.T
    mid = N_CHAIN // 2
    sl = slice(mid - L_INT // 2, mid + L_INT // 2)
    CA = C[sl, sl]
    w, u = np.linalg.eigh(CA)
    w = np.clip(w, EPS, 1.0 - EPS)
    K = (u * np.log((1.0 - w) / w)) @ u.T
    return K, w


def R_tri(K: np.ndarray) -> float:
    T = np.diag(np.diag(K))
    n = K.shape[0]
    for i in range(n - 1):
        T[i, i + 1] = K[i, i + 1]
        T[i + 1, i] = K[i + 1, i]
    return float(np.linalg.norm(K - T, "fro") / np.linalg.norm(K, "fro"))


def main() -> int:
    masses = [0.0, 0.1, 0.25, 0.5]
    R0 = R_tri(K_of(0.0)[0])
    ratios = []
    rows = []
    for m in masses:
        K, w = K_of(m)
        R = R_tri(K)
        ratio = R / R0
        ratios.append(ratio)
        rows.append({"m": m, "mL": m * L_INT, "R": R, "R_over_R0": ratio, "C_min": float(w.min())})
    in_window = [rows[i] for i, m in enumerate(masses) if 0 < m * L_INT <= 8]
    c2 = all(r["R_over_R0"] < THRESH for r in in_window)
    payload = {
        "task": "m9.10_audit_A2",
        "method": "independent chain N=180 L=24; no solver import",
        "R0": R0,
        "rows": rows,
        "C2": bool(c2),
        "verdicts": {
            "C2": "CONFIRMED" if c2 else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_10_audit_A2.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if c2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
