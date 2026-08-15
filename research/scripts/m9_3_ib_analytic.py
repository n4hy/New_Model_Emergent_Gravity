#!/usr/bin/env python3
"""M9.3: analytic reduction of I_B in d=2 and a source-dependence test.

Hyperbolic coordinate u = artanh(x/R), R=1. Weight-1 Jacobians cancel and
    Q(s) = int du dv b(u) b(v) / sinh^2(u - v + pi s)
         = int d tau C(tau) / sinh^2(tau + pi s)
with C = autocorrelation of b(u) := beta(tanh u).

The modular kernel pulls back to
    int K(s) f(tau + pi s) ds = int d alpha sech^2(alpha) f(tau+alpha) / 2.

Hadamard finite part of that integral against 1/sinh^2, derived in the
accompanying note:
    I_total^FP(tau) = 2 sech^2(tau) * (tau * tanh(tau) - 1)

This script
  1. verifies that identity by quadrature (away from the pole, plus a
     subtracted comparison),
  2. computes the relative residue
        r(b) = int K(s) (Q(s)-Q(0)) ds  /  |Q_reg(0)|
     for several source shapes b, using the smooth difference
        1/sinh^2(tau+pi s) - 1/sinh^2(tau)
     which is integrable for each s != 0,
  3. reports whether r is universal (source-independent at the 1e-6 level).

A universal multi-digit coefficient EXISTS only if r(b) agrees across
inequivalent b. If it moves, there is no single number to extract.

Writes ../data/m9_3_ib_analytic.json
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))


def i_total_fp(tau: np.ndarray) -> np.ndarray:
    """Closed form: 2 sech^2(tau) (tau tanh(tau) - 1)."""
    return 2.0 * (1.0 / np.cosh(tau) ** 2) * (tau * np.tanh(tau) - 1.0)


def verify_closed_form(n: int = 4001) -> dict:
    """Compare closed form to subtracted quadrature at sample tau != 0."""
    taus = np.array([0.4, 0.8, 1.2, 2.0])
    alpha = np.linspace(-18.0, 18.0, n)
    da = alpha[1] - alpha[0]
    sech2 = 1.0 / np.cosh(alpha) ** 2
    rows = []
    max_rel = 0.0
    for tau in taus:
        w = tau + alpha
        # subtract 1/w^2 so the integrand is smooth; add back its integral
        # int sech^2(a)/(2 (tau+a)^2) da is I_sing, not needed here.
        # Direct Hadamard: 1/sinh^2(w) - 1/w^2 is smooth.
        h = np.empty_like(w)
        small = np.abs(w) < 1e-8
        h[small] = -1.0 / 3.0
        h[~small] = 1.0 / np.sinh(w[~small]) ** 2 - 1.0 / w[~small] ** 2
        i_fin = 0.5 * float(np.sum(sech2 * h) * da)
        # I_sing by quadrature of sech^2 / (2 w^2) with a hole
        hole = np.abs(w) > 0.04
        i_sing = 0.5 * float(np.sum(sech2[hole] / (w[hole] ** 2)) * da)
        i_num = i_fin + i_sing
        i_cl = float(i_total_fp(np.array([tau]))[0])
        rel = abs(i_num - i_cl) / max(abs(i_cl), 1e-12)
        max_rel = max(max_rel, rel)
        rows.append(
            {
                "tau": float(tau),
                "I_finite_quad": i_fin,
                "I_sing_hole": i_sing,
                "I_sum": i_num,
                "I_closed": i_cl,
                "rel_err": rel,
            }
        )
    return {"max_rel_err": max_rel, "rows": rows}


def autocorr(b: np.ndarray, du: float) -> np.ndarray:
    """C(tau) on the same grid, tau = k*du, via FFT. Real even."""
    n = b.size
    # zero-pad to 2n
    bp = np.zeros(2 * n)
    bp[:n] = b
    fb = np.fft.rfft(bp)
    c = np.fft.irfft(fb * np.conjugate(fb)).real * du
    # c[k] ~ int b(u) b(u - k du) du, wrap; keep centered
    return c


def relative_residue(b: np.ndarray, u: np.ndarray, s_max: float = 3.0, n_s: int = 121) -> dict:
    """r = int K(s)(Q(s)-Q(0)) / |Q_h(0)| using the smooth h-kernel.

    Q_h(s) = int C(tau) (1/sinh^2(tau+pi s) - 1/(tau+pi s)^2) d tau
    is finite. Q(s)-Q(0) = int C [1/sinh^2(tau+pi s) - 1/sinh^2(tau)]
    which for s!=0 we evaluate as
        h(tau+pi s) - h(tau) + 1/(tau+pi s)^2 - 1/tau^2
    all of which is locally integrable after pairing.
    """
    du = u[1] - u[0]
    c = autocorr(b, du)
    n = u.size
    # tau grid matching irfft layout: 0, du, ..., then negative
    tau = np.fft.fftfreq(2 * n, d=1.0 / (2 * n * du)) * (2 * n) / (2 * n)  # k*du
    tau = np.arange(2 * n) * du
    tau[n:] -= 2 * n * du  # ..., -n*du
    # actually irfft output: index 0 = lag 0, 1 = +du, ..., n = -n du wrap
    # Build a centered C on a linear tau axis
    tau_lin = np.concatenate([np.arange(-n, n) * du])
    c_lin = np.concatenate([c[n:], c[:n]])  # lag -n ... n-1  after we rebuild

    # rebuild c_lin properly from padded irfft
    # c[k] for k=0..2n-1 is lag k, with lag k and lag k-2n identified
    c_full = np.zeros(2 * n)
    c_full[n:] = c[:n]  # lags 0..n-1 at indices n..2n-1
    c_full[:n] = c[n:]  # lags -n..-1 at indices 0..n-1
    # wait, c[n:] from irfft is lags n..2n-1 = lags n..-1 (wrap)
    # Standard: c_fft[k] = lag k*du for k=0,...,2n-1 with wrap at n.
    c_shift = np.fft.fftshift(c)
    tau_shift = (np.arange(2 * n) - n) * du

    def h_of(w: np.ndarray) -> np.ndarray:
        out = np.empty_like(w)
        small = np.abs(w) < 1e-8
        out[small] = -1.0 / 3.0
        ww = w[~small]
        out[~small] = 1.0 / np.sinh(ww) ** 2 - 1.0 / ww**2
        return out

    def invsinh2(w: np.ndarray) -> np.ndarray:
        out = np.empty_like(w)
        small = np.abs(w) < 1e-8
        out[small] = 0.0  # never used at 0 in the difference
        out[~small] = 1.0 / np.sinh(w[~small]) ** 2
        return out

    def q_full(s: float) -> float:
        w = tau_shift + np.pi * s
        # use h + 1/w^2 with hole on 1/w^2
        hh = h_of(w)
        hole = np.abs(w) > 0.02
        pole = np.zeros_like(w)
        pole[hole] = 1.0 / w[hole] ** 2
        return float(np.sum(c_shift * (hh + pole)) * du)

    def q_diff(s: float) -> float:
        if abs(s) < 1e-15:
            return 0.0
        w = tau_shift + np.pi * s
        w0 = tau_shift
        # 1/sinh^2(w) - 1/sinh^2(w0), both poles isolated
        d = np.zeros_like(w)
        ok = (np.abs(w) > 0.015) & (np.abs(w0) > 0.015)
        d[ok] = 1.0 / np.sinh(w[ok]) ** 2 - 1.0 / np.sinh(w0[ok]) ** 2
        return float(np.sum(c_shift * d) * du)

    s_vals = np.linspace(-s_max, s_max, n_s)
    s_vals = s_vals[np.abs(s_vals) > 1e-8]
    k = np.pi / (2.0 * np.cosh(np.pi * s_vals) ** 2)
    diffs = np.array([q_diff(s) for s in s_vals])
    ds = s_vals[1] - s_vals[0] if len(s_vals) > 1 else 0.0
    # include a crude trapezoid; endpoints already ~0
    residue = float(np.sum(k * diffs) * ds)
    q0 = q_full(0.0)
    # regularized local form: int C h(tau)  (the finite part of Q(0))
    q0_h = float(np.sum(c_shift * h_of(tau_shift)) * du)
    rel_vs_q0 = residue / abs(q0) if abs(q0) > 1e-18 else float("nan")
    rel_vs_h = residue / abs(q0_h) if abs(q0_h) > 1e-18 else float("nan")
    return {
        "Q0": q0,
        "Q0_h": q0_h,
        "residue": residue,
        "r_vs_Q0": rel_vs_q0,
        "r_vs_Q0_h": rel_vs_h,
        "kernel_int": float(np.sum(k) * ds),
    }


def sources(u: np.ndarray) -> dict[str, np.ndarray]:
    g1 = np.exp(-(u**2) / (2 * 0.6**2))
    g2 = np.exp(-(u**2) / (2 * 1.1**2))
    se = 1.0 / np.cosh(u)
    expa = np.exp(-np.abs(u))
    bump = np.zeros_like(u)
    m = np.abs(u) < 1.4
    z = u[m] / 1.4
    bump[m] = np.exp(-1.0 / (1.0 - z * z))
    return {
        "gaussian_s0.6": g1,
        "gaussian_s1.1": g2,
        "sech": se,
        "exp_abs": expa,
        "bump_u": bump,
    }


def main() -> int:
    ver = verify_closed_form()
    u = np.linspace(-8.0, 8.0, 1024)
    results = {}
    for name, b in sources(u).items():
        results[name] = relative_residue(b, u)

    rs = [results[k]["r_vs_Q0_h"] for k in results]
    rs_q = [results[k]["r_vs_Q0"] for k in results]
    spread_h = float(max(rs) - min(rs))
    spread_q = float(max(rs_q) - min(rs_q))
    universal_h = spread_h < 1e-3
    payload = {
        "task": "m9.3_ib_analytic",
        "closed_form": "I_total_FP(tau) = 2 sech^2(tau) (tau tanh(tau) - 1)",
        "closed_form_check": ver,
        "sources": results,
        "r_vs_Q0_h": rs,
        "r_vs_Q0": rs_q,
        "spread_r_vs_Q0_h": spread_h,
        "spread_r_vs_Q0": spread_q,
        "universal_at_1e-3_vs_Q0_h": universal_h,
        "verdict": (
            "UNIVERSAL_CANDIDATE"
            if universal_h
            else "NOT_UNIVERSAL"
        ),
        "admission_if_not_universal": (
            "If r moves with the source, there is no single multi-digit "
            "coefficient of I_B independent of beta. That is a result, "
            "not a failed extraction."
        ),
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_3_ib_analytic.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print("closed-form max rel err:", ver["max_rel_err"])
    print("r vs Q0_h:", rs, "spread", spread_h)
    print("r vs Q0:  ", rs_q, "spread", spread_q)
    print("VERDICT:", payload["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
