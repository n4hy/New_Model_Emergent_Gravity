#!/usr/bin/env python3
"""M9.4: complete Hadamard / Mittag-Leffler expansion of I_B in d=2.

Exact identity (all w not on i pi Z):
    1/sinh^2(w) = sum_{n=-inf}^{inf} 1/(w - i n pi)^2

n=0 is the coincidence pole. The Hadamard-finite remainder is
    h(w) = sum_{n != 0} 1/(w - i n pi)^2 = 1/sinh^2(w) - 1/w^2

Kernel pullback (alpha = pi s):
    I_sing(zeta) = (1/2) int sech^2(a) / (a+zeta)^2 da
                 = pi^{-2} polygamma(2, 1/2 - i zeta/pi)   (Im zeta > 0)
                 = pi^{-2} Re polygamma(2, 1/2 - i tau/pi) (zeta = tau real)

Hadamard-finite modular integral:
    H(tau) = sum_{n != 0} I_sing(tau - i n pi)
           = pi^{-2} sum_{m=1}^{inf} [
                 polygamma(2, m + 1/2 + i tau/pi)
               + polygamma(2, m + 1/2 - i tau/pi)
             ]

A UNIVERSAL coefficient exists iff H(tau) = lambda * K_loc(tau)
for a local kernel K_loc independent of the source. We test
K_loc in {1/tau^2, 1/sinh^2(tau), I_sing(tau), h(tau)}.

Writes ../data/m9_4_ib_hadamard_complete.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from mpmath import mp

mp.dps = 25


def _pg2(z) -> complex:
    """polygamma(2, z) via mpmath, accepting complex."""
    return complex(mp.polygamma(2, z))

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))


def i_sing_complex(zeta: complex) -> complex:
    """I_sing(zeta) from polygamma, Im zeta != 0."""
    if zeta.imag > 0:
        return _pg2(0.5 - 1j * zeta / np.pi) / (np.pi**2)
    if zeta.imag < 0:
        return _pg2(0.5 + 1j * zeta / np.pi) / (np.pi**2)
    return i_sing_real(float(zeta.real))


def i_sing_real(tau: np.ndarray | float) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float64)
    val = np.array([_pg2(0.5 - 1j * t / np.pi) for t in np.atleast_1d(tau)], dtype=np.complex128)
    out = np.real(val)
    return out if np.ndim(tau) else float(out[0])


def H_of(tau: np.ndarray, mmax: int = 80) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float64)
    acc = np.zeros(tau.shape, dtype=np.complex128)
    for m in range(1, mmax + 1):
        acc += np.array([_pg2(m + 0.5 + 1j * t / np.pi) for t in tau], dtype=np.complex128)
        acc += np.array([_pg2(m + 0.5 - 1j * t / np.pi) for t in tau], dtype=np.complex128)
    return np.real(acc) / (np.pi**2)


def h_of(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    out = np.empty_like(w)
    small = np.abs(w) < 1e-10
    out[small] = -1.0 / 3.0
    ww = w[~small]
    out[~small] = 1.0 / np.sinh(ww) ** 2 - 1.0 / ww**2
    return out


def verify_polygamma_vs_quad() -> dict:
    """I_sing at a few imaginary shifts vs sech^2 quadrature."""
    a = np.linspace(-20.0, 20.0, 8001)
    da = a[1] - a[0]
    sech2 = 1.0 / np.cosh(a) ** 2
    rows = []
    for eta in (0.4, 0.8, 1.5):
        zeta = 0.7 + 1j * eta
        quad = 0.5 * np.sum(sech2 / (a + zeta) ** 2) * da
        clo = i_sing_complex(zeta)
        rel = abs(quad - clo) / max(abs(clo), 1e-18)
        rows.append(
            {
                "zeta": [zeta.real, zeta.imag],
                "quad_re": float(np.real(quad)),
                "closed_re": float(np.real(clo)),
                "rel_err": float(rel),
            }
        )
    return {"rows": rows, "max_rel": max(r["rel_err"] for r in rows)}


def ratio_table(taus: np.ndarray) -> dict:
    Ht = H_of(taus)
    Is = i_sing_real(taus)
    invt2 = 1.0 / taus**2
    invs2 = 1.0 / np.sinh(taus) ** 2
    hh = h_of(taus)
    return {
        "tau": taus.tolist(),
        "H": Ht.tolist(),
        "H_over_Ising": (Ht / Is).tolist(),
        "H_over_inv_tau2": (Ht / invt2).tolist(),
        "H_over_inv_sinh2": (Ht / invs2).tolist(),
        "H_over_h": (Ht / hh).tolist(),
        "spread_H_over_Ising": float(np.max(Ht / Is) - np.min(Ht / Is)),
        "spread_H_over_inv_tau2": float(np.max(Ht / invt2) - np.min(Ht / invt2)),
        "spread_H_over_h": float(np.max(Ht / hh) - np.min(Ht / hh)),
    }


def r_for_gaussian(sigma: float, tau: np.ndarray, Ht: np.ndarray, umax: float, n: int) -> dict:
    """r = int C H / int C K_loc for C = autocorrelation of e^{-u^2/(2s^2)}."""
    u = np.linspace(-umax, umax, n)
    du = u[1] - u[0]
    b = np.exp(-(u**2) / (2.0 * sigma**2))
    C = np.correlate(b, b, mode="full") * du
    num = float(np.sum(C * Ht) * du)
    den = {}
    mask = np.abs(tau) > 1e-10
    den["Ising"] = float(np.sum(C * i_sing_real(tau)) * du)
    den["h"] = float(np.sum(C * h_of(tau)) * du)
    den["inv_tau2"] = float(np.sum(C[mask] / tau[mask] ** 2) * du)
    invs = np.zeros_like(tau)
    invs[mask] = 1.0 / np.sinh(tau[mask]) ** 2
    den["inv_sinh2"] = float(np.sum(C * invs) * du)
    return {
        "sigma": sigma,
        "num_CH": num,
        "r": {k: num / v if abs(v) > 1e-18 else float("nan") for k, v in den.items()},
        "den": den,
    }


def main() -> int:
    ver = verify_polygamma_vs_quad()
    taus = np.array([0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])
    ratios = ratio_table(taus)
    umax, n = 7.0, 161
    u = np.linspace(-umax, umax, n)
    du = u[1] - u[0]
    tau_grid = np.arange(-(n - 1), n) * du
    Ht_grid = H_of(tau_grid, mmax=25)
    gaussians = [r_for_gaussian(s, tau_grid, Ht_grid, umax, n) for s in (0.4, 0.7, 1.0, 1.4)]
    # spreads of r across gaussians per denominator
    spreads = {}
    for key in ("Ising", "h", "inv_tau2", "inv_sinh2"):
        vals = [g["r"][key] for g in gaussians]
        spreads[key] = {
            "values": vals,
            "spread": float(max(vals) - min(vals)),
            "mean": float(np.mean(vals)),
        }
    # universality: H/K constant in tau AND r independent of sigma
    uni_ising = ratios["spread_H_over_Ising"] < 1e-6 and spreads["Ising"]["spread"] < 1e-6
    payload = {
        "task": "m9.4_hadamard_complete",
        "expansion": "1/sinh^2(w) = sum_n 1/(w - i n pi)^2  (exact)",
        "H_definition": "sum_{n!=0} I_sing(tau - i n pi), I_sing via polygamma(2)",
        "polygamma_vs_quad": ver,
        "ratios": ratios,
        "gaussians": gaussians,
        "r_spreads": spreads,
        "universal_vs_Ising": bool(uni_ising),
        "verdict": (
            "UNIVERSAL_VS_ISING"
            if uni_ising
            else "HADAMARD_COMPLETE_NOT_UNIVERSAL"
        ),
        "admission": (
            "The Mittag-Leffler expansion is complete and exact. "
            "A single multi-digit coefficient exists only if H is "
            "proportional to one local kernel. The measured spreads "
            "decide that, and the Final Status label is not moved "
            "unless a kernel gives source-independent r at high precision."
        ),
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_4_ib_hadamard_complete.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print("polygamma vs quad max rel", ver["max_rel"])
    print("spread H/Ising", ratios["spread_H_over_Ising"])
    print("spread H/1/tau^2", ratios["spread_H_over_inv_tau2"])
    print("spread H/h", ratios["spread_H_over_h"])
    for k, v in spreads.items():
        print(f"r vs {k}: {v['values']}  spread={v['spread']:.6e} mean={v['mean']:.6e}")
    print("VERDICT", payload["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
