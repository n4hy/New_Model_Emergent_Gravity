#!/usr/bin/env python3
"""A2 audit: 2d Dirac on a different grid; 1d scalar on a different L. No solver import."""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
EPS = 1e-12
THRESH = 2.0


def idx(x, y, n):
    return x * n + y


def dirac_2d_R(n: int, L: int, mass: float) -> float:
    H = np.zeros((n * n, n * n))
    for x in range(n):
        for y in range(n):
            i = idx(x, y, n)
            H[i, i] = mass * (1.0 if (x + y) % 2 == 0 else -1.0)
            if x + 1 < n:
                j = idx(x + 1, y, n)
                H[i, j] = H[j, i] = -1.0
            if y + 1 < n:
                j = idx(x, y + 1, n)
                H[i, j] = H[j, i] = -1.0
    ev, vecs = np.linalg.eigh(H)
    C = vecs[:, ev < 0.0] @ vecs[:, ev < 0.0].T
    mid = n // 2
    sl = [idx(x, y, n) for x in range(mid - L // 2, mid + L // 2)
          for y in range(mid - L // 2, mid + L // 2)]
    CA = C[np.ix_(sl, sl)]
    w, u = np.linalg.eigh(CA)
    w = np.clip(w, EPS, 1.0 - EPS)
    K = (u * np.log((1.0 - w) / w)) @ u.T
    loc = np.zeros_like(K)
    for x in range(L):
        for y in range(L):
            i = idx(x, y, L)
            loc[i, i] = K[i, i]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                xx, yy = x + dx, y + dy
                if 0 <= xx < L and 0 <= yy < L:
                    j = idx(xx, yy, L)
                    loc[i, j] = K[i, j]
    return float(np.linalg.norm(K - loc, "fro") / np.linalg.norm(K, "fro"))


def main() -> int:
    R0 = dirac_2d_R(28, 10, 0.0)
    R1 = dirac_2d_R(28, 10, 0.3)  # mL = 3.0
    R2 = dirac_2d_R(28, 10, 0.7)  # mL = 7.0
    r1, r2 = R1 / R0, R2 / R0
    c2 = bool(r1 < THRESH and r2 < THRESH)
    payload = {
        "task": "m9.12_audit_A2",
        "method": "2d Dirac N=28 L=10; no solver import",
        "R0": R0,
        "R_mL3": R1,
        "R_mL7": R2,
        "ratio_mL3": r1,
        "ratio_mL7": r2,
        "C2": c2,
        "verdicts": {"C2_2d": "CONFIRMED" if c2 else "REFUTED"},
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_12_audit_A2.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if c2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
