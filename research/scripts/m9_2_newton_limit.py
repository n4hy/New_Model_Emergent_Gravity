#!/usr/bin/env python3
"""M9.2: Newton limit of inherited Einstein (locked 2026-08-15).

3-d discrete Poisson on a cube, Dirichlet Φ=0 on the boundary.
NOT the continuum Green's function. NOT a 2-d Laplacian (that is C4).
NOT GEM. NOT FGHMV. G is a coded positive constant.

    ∇²Φ = 4π G ρ ,   a = −∇Φ
    ds² = −(1+2Φ) dt² + (1−2Φ) δ_ij dx^i dx^j

PRE-REGISTERED (tasks/m9_2_task_details.md):
  Box half-width L. Two grids. Source rms < 0.05 L.
  Probes r ∈ {0.30 L, 0.35 L, 0.40 L}.
  C1 PRIMARY. Attractive GM/r²: (i) a·rhat < 0;
     (ii) ||a| r²/(GM) − 1| < 0.05 at all three r;
     (iii) log-log slope α of |a| vs r has |α+2| < 0.08.
     Fail if any of (i)–(iii) missed on the finer grid.
  C2  |Φ r/(GM) + 1| < 0.05 at the same three r.
  C3  ρ=0: max|a|, max|Φ| < 1e-8 times the C1 values.
  C4  2-d Laplacian mutation: |α+1| < 0.15 and C1(ii) FAIL.
  C5  G→−G: C1(i) flips; C1(ii) holds with |G|.

Writes ../data/m9_2_newton.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.fft import dst

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
L = 1.0
G_POS = 1.0
MASS = 1.0
R_SRC = 0.04 * L
NS = (65, 97)
PROBES = (0.30 * L, 0.35 * L, 0.40 * L)
def grid(n: int):
    xs = np.linspace(-L, L, n)
    h = float(xs[1] - xs[0])
    return xs, h


def uniform_ball_rho(xs: np.ndarray, mass: float, radius: float) -> np.ndarray:
    n = len(xs)
    xx, yy, zz = np.meshgrid(xs, xs, xs, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy + zz * zz)
    dens = np.zeros((n, n, n), dtype=float)
    inside = rr <= radius
    vol = 4.0 * np.pi * radius**3 / 3.0
    dens[inside] = mass / vol
    # renormalize so discrete mass is exact
    h = float(xs[1] - xs[0])
    m_disc = float(dens.sum() * h**3)
    if m_disc > 0.0:
        dens *= mass / m_disc
    return dens


def rms_radius(xs: np.ndarray, rho: np.ndarray) -> float:
    h = float(xs[1] - xs[0])
    xx, yy, zz = np.meshgrid(xs, xs, xs, indexing="ij")
    w = rho * h**3
    m = float(w.sum())
    if m <= 0.0:
        return 0.0
    return float(np.sqrt(np.sum((xx * xx + yy * yy + zz * zz) * w) / m))


def _dst_poisson(rhs_int: np.ndarray, h: float) -> np.ndarray:
    """Exact inverse of the Dirichlet 2nd-difference Laplacian via DST-I."""
    axes = tuple(range(rhs_int.ndim))
    fhat = rhs_int
    for ax in axes:
        fhat = dst(fhat, type=1, axis=ax)
    m = rhs_int.shape[0]
    j = np.arange(1, m + 1, dtype=float)
    lam1 = -4.0 / (h * h) * np.sin(np.pi * j / (2.0 * (m + 1))) ** 2
    if rhs_int.ndim == 3:
        lam = lam1[:, None, None] + lam1[None, :, None] + lam1[None, None, :]
    else:
        lam = lam1[:, None] + lam1[None, :]
    phat = fhat / lam
    for ax in axes:
        phat = dst(phat, type=1, axis=ax)
    return phat / (2.0 * (m + 1)) ** rhs_int.ndim


def poisson_3d(xs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ∇²Φ = rhs, Φ=0 on the boundary. 7-point stencil."""
    n = len(xs)
    h = float(xs[1] - xs[0])
    phi = np.zeros((n, n, n), dtype=float)
    phi[1:-1, 1:-1, 1:-1] = _dst_poisson(rhs[1:-1, 1:-1, 1:-1], h)
    return phi


def poisson_2d(xs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Mid-plane 5-point Laplacian, Φ=0 on the square boundary."""
    n = len(xs)
    h = float(xs[1] - xs[0])
    phi = np.zeros((n, n), dtype=float)
    phi[1:-1, 1:-1] = _dst_poisson(rhs[1:-1, 1:-1], h)
    return phi


def interp_phi(xs: np.ndarray, phi: np.ndarray, point) -> float:
    """Trilinear interpolation. phi is 3d or 2d (then z ignored)."""
    pt = np.asarray(point, dtype=float)
    if phi.ndim == 2:
        p2 = phi
        x, y = pt[0], pt[1]
        def one(ax, val):
            if val <= ax[0]:
                return 0, 0.0
            if val >= ax[-1]:
                return len(ax) - 2, 1.0
            k = int(np.searchsorted(ax, val) - 1)
            k = max(0, min(k, len(ax) - 2))
            t = (val - ax[k]) / (ax[k + 1] - ax[k])
            return k, t
        i, tx = one(xs, x)
        j, ty = one(xs, y)
        return float(
            (1 - tx) * (1 - ty) * p2[i, j]
            + tx * (1 - ty) * p2[i + 1, j]
            + (1 - tx) * ty * p2[i, j + 1]
            + tx * ty * p2[i + 1, j + 1]
        )
    def one(ax, val):
        if val <= ax[0]:
            return 0, 0.0
        if val >= ax[-1]:
            return len(ax) - 2, 1.0
        k = int(np.searchsorted(ax, val) - 1)
        k = max(0, min(k, len(ax) - 2))
        t = (val - ax[k]) / (ax[k + 1] - ax[k])
        return k, t
    i, tx = one(xs, pt[0])
    j, ty = one(xs, pt[1])
    k, tz = one(xs, pt[2])
    acc = 0.0
    for di, wi in ((0, 1 - tx), (1, tx)):
        for dj, wj in ((0, 1 - ty), (1, ty)):
            for dk, wk in ((0, 1 - tz), (1, tz)):
                acc += wi * wj * wk * phi[i + di, j + dj, k + dk]
    return float(acc)


def grid_ax(xs: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Discrete a_x = −∂_x Φ by centered differences. 3d cube or 2d plane."""
    h = float(xs[1] - xs[0])
    ax = np.zeros_like(phi)
    if phi.ndim == 3:
        ax[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * h)
    else:
        ax[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2.0 * h)
    return ax


def radial_a(xs, phi, radius: float) -> tuple[float, float]:
    """Φ and a_x at (r,0,0) from the discrete gradient."""
    p0 = interp_phi(xs, phi, (radius, 0.0, 0.0))
    ax = grid_ax(xs, phi)
    return p0, interp_phi(xs, ax, (radius, 0.0, 0.0))


def log_slope(radii, accs):
    accs = np.asarray(accs, dtype=float)
    if float(np.max(np.abs(accs))) == 0.0:
        return None
    lr = np.log(np.asarray(radii, dtype=float))
    la = np.log(np.abs(accs))
    mat = np.column_stack([lr, np.ones(len(lr))])
    coef, _, _, _ = np.linalg.lstsq(mat, la, rcond=None)
    return float(coef[0])


def score_grid(n: int, g_const: float, rho: np.ndarray, xs: np.ndarray) -> dict:
    rhs = 4.0 * np.pi * g_const * rho
    phi = poisson_3d(xs, rhs)
    h = float(xs[1] - xs[0])
    gm = abs(g_const) * MASS
    phis, accs, c1ii, c2r = [], [], [], []
    for r in PROBES:
        pval, ar = radial_a(xs, phi, r)
        phis.append(pval)
        accs.append(ar)
        c1ii.append(abs(abs(ar) * r * r / gm - 1.0))
        c2r.append(abs(pval * r / (g_const * MASS) + 1.0) if g_const != 0 else float("nan"))
    attractive = all(a * np.sign(-g_const) < 0.0 for a in accs) if g_const != 0 else False
    # C1(i) for G>0 is a<0; for G<0 is a>0. Use a·rhat < 0 when G>0.
    c1_i = all(a < 0.0 for a in accs) if g_const > 0 else all(a > 0.0 for a in accs)
    slope = log_slope(PROBES, accs)
    return {
        "n": n,
        "h": h,
        "phi": phis,
        "a_r": accs,
        "c1_ii": c1ii,
        "c2": c2r,
        "c1_i_attractive_for_Gsign": c1_i,
        "all_a_negative": all(a < 0.0 for a in accs),
        "slope": slope,
        "c1_ii_pass": all(d < 0.05 for d in c1ii),
        "c1_iii_pass": bool(slope is not None and abs(slope + 2.0) < 0.08),
        "c2_pass": all(d < 0.05 for d in c2r),
        "max_abs_a": float(np.max(np.abs(accs))),
        "max_abs_phi": float(np.max(np.abs(phis))),
    }


def score_2d(n: int) -> dict:
    xs, h = grid(n)
    rho3 = uniform_ball_rho(xs, MASS, R_SRC)
    mid = n // 2
    sig = rho3[:, :, mid]
    # treat the midplane as a 2-d density; mass = ∑ σ h²
    m2 = float(sig.sum() * h * h)
    phi = poisson_2d(xs, 4.0 * np.pi * G_POS * sig)
    accs, c1ii = [], []
    gm = G_POS * (m2 if m2 > 0 else MASS)
    for r in PROBES:
        _, ar = radial_a(xs, phi, r)
        accs.append(ar)
        c1ii.append(abs(abs(ar) * r * r / gm - 1.0))
    slope = log_slope(PROBES, accs)
    return {
        "n": n,
        "m2": m2,
        "a_r": accs,
        "c1_ii": c1ii,
        "slope": slope,
        "c1_ii_pass": all(d < 0.05 for d in c1ii),
        "alpha_plus_1": abs(slope + 1.0),
        "c4_alpha": abs(slope + 1.0) < 0.15,
    }


def main() -> int:
    coarse_n, fine_n = NS
    xs_c, _ = grid(coarse_n)
    xs_f, _ = grid(fine_n)
    rho_c = uniform_ball_rho(xs_c, MASS, R_SRC)
    rho_f = uniform_ball_rho(xs_f, MASS, R_SRC)
    rms_c = rms_radius(xs_c, rho_c)
    rms_f = rms_radius(xs_f, rho_f)
    g_c = score_grid(coarse_n, G_POS, rho_c, xs_c)
    g_f = score_grid(fine_n, G_POS, rho_f, xs_f)
    vac_c = score_grid(coarse_n, G_POS, np.zeros_like(rho_c), xs_c)
    vac_f = score_grid(fine_n, G_POS, np.zeros_like(rho_f), xs_f)
    flip = score_grid(fine_n, -G_POS, rho_f, xs_f)
    mut2 = score_2d(fine_n)
    c1 = bool(
        g_f["c1_i_attractive_for_Gsign"]
        and g_f["c1_ii_pass"]
        and g_f["c1_iii_pass"]
        and g_c["c1_i_attractive_for_Gsign"]
        and g_c["c1_ii_pass"]
        and g_c["c1_iii_pass"]
    )
    # lock: fail if missed on the finer grid
    c1_fine = bool(g_f["c1_i_attractive_for_Gsign"] and g_f["c1_ii_pass"] and g_f["c1_iii_pass"])
    c2 = bool(g_f["c2_pass"] and g_c["c2_pass"])
    scale_a = g_f["max_abs_a"] if g_f["max_abs_a"] > 0 else 1.0
    scale_p = g_f["max_abs_phi"] if g_f["max_abs_phi"] > 0 else 1.0
    c3 = bool(
        vac_f["max_abs_a"] < 1e-8 * scale_a
        and vac_f["max_abs_phi"] < 1e-8 * scale_p
        and vac_c["max_abs_a"] < 1e-8 * scale_a
        and vac_c["max_abs_phi"] < 1e-8 * scale_p
    )
    c4 = bool(mut2["c4_alpha"] and (not mut2["c1_ii_pass"]))
    c5 = bool((not flip["all_a_negative"]) and flip["c1_ii_pass"])
    payload = {
        "task": "m9.2_newton_limit",
        "inherited": True,
        "not_claimed": [
            "FGHMV",
            "entanglement gravity",
            "de Sitter",
            "Hehl-Datta",
            "GEM",
            "MODELS.md column",
        ],
        "L": L,
        "G": G_POS,
        "M": MASS,
        "R_src": R_SRC,
        "rms_radius": [rms_c, rms_f],
        "rms_lock": 0.05 * L,
        "grids": [g_c, g_f],
        "vacuum": [vac_c, vac_f],
        "mutation_2d": mut2,
        "mutation_flipG": flip,
        "C1_PRIMARY": c1_fine,
        "C1_both_grids": c1,
        "C2": c2,
        "C3": c3,
        "C4": c4,
        "C5": c5,
        "verdict": "NEWTON_INHERITED_PASS" if c1_fine else "NEWTON_INHERITED_FAIL",
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_2_newton.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if c1_fine else 1


if __name__ == "__main__":
    raise SystemExit(main())
