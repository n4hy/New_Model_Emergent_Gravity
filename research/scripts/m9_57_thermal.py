#!/usr/bin/env python3
"""M9.57: thermal Fermi gas. Does p/ρ ever hit −1?

Paper 66: T=0 occupation transfer is dust. A thermal
state is a different occupation, n=1/(e^{ε/T}+1), μ=0.
Same hop virial. Source is the excess over T=0
(Paper 58: the sea does not gravitate).

    n_k(T) = 1 / (e^{ε_k/T} + 1)
    E = ∑ n ε,   P = ∑ n w,   w = (2/3) k·sin k
    r(T) = (P(T)−P(0)) / (E(T)−E(0))

PRE-REGISTERED:
  N=12, periodic hop, μ=0, k in the 1st BZ.
  T ∈ {0.1, 0.25, 0.5, 1, 2, 4} (hop units).
  Skip a T if |δE| / |E(0)| < 1e-8.
  C_lambda PRIMARY. |r+1| < 0.25 at any kept T
  C_pos    r > 0 at every kept T
  C_nolam  r > −0.50 at every kept T
  Raw P(0)/E(0) is reported and is not a source.
  Forbidden: p = −E/V as a Λ claim.

Not claimed: 8πG, derived Einstein, de Sitter, MODELS.md.

Writes ../data/m9_57_thermal.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
N = 12
TEMPS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)


def fold(k: float) -> float:
    return float((k + np.pi) % (2.0 * np.pi) - np.pi)


def spectrum(n: int):
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
    r0 = p0 / e0 if e0 != 0.0 else float("nan")
    rows = []
    c_lambda = False
    c_pos = True
    c_nolam = True
    kept = 0
    for temp in TEMPS:
        occ = fermi(eps, temp)
        energy = float(np.dot(occ, eps))
        press = float(np.dot(occ, wgt))
        de = energy - e0
        dp = press - p0
        if abs(e0) < 1e-18 or abs(de) / abs(e0) < 1e-8:
            rows.append(
                {
                    "T": temp,
                    "E": energy,
                    "P": press,
                    "dE": de,
                    "dP": dp,
                    "r": None,
                    "kept": False,
                }
            )
            continue
        ratio = dp / de
        kept += 1
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
                "kept": True,
            }
        )
    ok = bool(kept >= 4 and (not c_lambda) and c_pos and c_nolam)
    payload = {
        "task": "m9.57_thermal",
        "n_modes": int(eps.size),
        "E0": e0,
        "P0": p0,
        "r0_raw_not_a_source": r0,
        "rows": rows,
        "n_kept": kept,
        "C_lambda_PRIMARY": c_lambda,
        "C_pos": c_pos,
        "C_nolam": c_nolam,
        "all_gates": ok,
        "verdict": "THERMAL_LAMBDA" if c_lambda else "THERMAL_NOT_LAMBDA",
        "not_claimed": ["8pi G", "derived Einstein", "de Sitter", "MODELS.md"],
        "forbidden": "p = -E/V as a Lambda claim",
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_57_thermal.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (not c_lambda) else 1


if __name__ == "__main__":
    raise SystemExit(main())
