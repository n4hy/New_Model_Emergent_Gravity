#!/usr/bin/env python3
"""M9.57 audit. N=10 PBC, own T set. Tries to FIND r≈−1.

T ∈ {0.2, 0.8, 3.0}. Same virial. Excess over T=0.
C_lambda CONFIRMED only if |r+1|<0.25 at a kept T.

Writes ../data/m9_57_audit_thermal.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
N = 10
TEMPS = (0.2, 0.8, 3.0)


def fold(k):
    return float((k + np.pi) % (2.0 * np.pi) - np.pi)


def spectrum(n):
    eps = []
    wgt = []
    for nx in range(n):
        for ny in range(n):
            for nz in range(n):
                kx = fold(2.0 * np.pi * nx / n)
                ky = fold(2.0 * np.pi * ny / n)
                kz = fold(2.0 * np.pi * nz / n)
                ek = -2.0 * (np.cos(kx) + np.cos(ky) + np.cos(kz))
                wk = (2.0 / 3.0) * (
                    kx * np.sin(kx) + ky * np.sin(ky) + kz * np.sin(kz)
                )
                eps.append(ek)
                wgt.append(wk)
    return np.asarray(eps, float), np.asarray(wgt, float)


def fermi(eps, temp):
    if temp <= 0.0:
        out = np.zeros_like(eps)
        out[eps < 0.0] = 1.0
        out[eps == 0.0] = 0.5
        return out
    x = np.clip(eps / temp, -40.0, 40.0)
    return 1.0 / (np.exp(x) + 1.0)


def main() -> int:
    eps, wgt = spectrum(N)
    n0 = fermi(eps, 0.0)
    e0 = float(np.dot(n0, eps))
    p0 = float(np.dot(n0, wgt))
    rows = []
    c_lambda = False
    c_pos = True
    c_nolam = True
    for temp in TEMPS:
        occ = fermi(eps, temp)
        energy = float(np.dot(occ, eps))
        press = float(np.dot(occ, wgt))
        de = energy - e0
        dp = press - p0
        ratio = dp / de if abs(de) / abs(e0) >= 1e-8 else float("nan")
        if abs(ratio + 1.0) < 0.25:
            c_lambda = True
        if ratio <= 0.0:
            c_pos = False
        if ratio <= -0.50:
            c_nolam = False
        rows.append(
            {
                "T": temp,
                "E": energy,
                "P": press,
                "dE": de,
                "dP": dp,
                "r": ratio,
            }
        )
    payload = {
        "task": "m9.57_audit_thermal",
        "n_modes": int(eps.size),
        "E0": e0,
        "P0": p0,
        "r0_raw_not_a_source": p0 / e0,
        "rows": rows,
        "C_lambda_PRIMARY": c_lambda,
        "C_pos": c_pos,
        "C_nolam": c_nolam,
        "verdicts": {
            "C_lambda": "CONFIRMED" if c_lambda else "REFUTED",
            "C_pos": "CONFIRMED" if c_pos else "REFUTED",
            "C_nolam": "CONFIRMED" if c_nolam else "REFUTED",
        },
        "not_claimed": ["8pi G", "derived Einstein", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_57_audit_thermal.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
