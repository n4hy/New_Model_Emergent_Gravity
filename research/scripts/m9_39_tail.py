#!/usr/bin/env python3
"""M9.39: which mass does Newton see — M_hat, M(<r), or the tail?

Paper 48 put a fictional point of mass M_hat into Poisson.
This run sources Poisson with the actual site-energy map δe.

PRE-REGISTERED:
  Same fermion as M9.38: N=12, R=3, packet (6,6,6), σ=1, α=0.02.
  κ, M_hat from even/odd well-inside split, seed 39.
  Two embeddings of the same δe, n=65, L=1, G=1, centred on source.

  NEAR (diagnostic): spacing 2L/(N-1). Probes sit inside the
  packet (r=2 sites = 0.36L). Not a 1/r² test.

  FAR (PRIMARY): scale so 3 sites = 0.05L (M9.2 source-size lock).
  Probes r = 0.30L, 0.35L, 0.40L along +x.
  Residual of mass M: ||a| r²/(G M) − 1|.
  C_hat     residual < 0.05 at all three, a_r < 0, |α+2|<0.08
  C_global  same with M_global = ∑δe
  C_which PRIMARY: smaller mean residual of {M_hat, M_global}.
    FIRST_LAW_IS_SOURCE if M_hat
    TOTAL_ENERGY_IS_SOURCE if M_global
  Control: point blob of mass M_hat still passes Paper 48 C1.

Not claimed: derived Poisson, 8πG from κ, 1/4G, FGHMV, dS, MODELS.md.

Writes ../data/m9_39_tail.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.fft import dst

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N = 12
RADIUS = 3
SRC = (6, 6, 6)
SIGMA = 1.0
ALPHA = 0.02
SEED = 39
LBOX = 1.0
NBOX = 65
GCONST = 1.0
PROBES = (0.30 * LBOX, 0.35 * LBOX, 0.40 * LBOX)
GATE_FAR = 0.05
R_PACK = 3.0
R_COMPACT = 0.05 * LBOX


def idx(x, y, z, n=N) -> int:
    return (x * n + y) * n + z


def hop_H(n: int) -> np.ndarray:
    ham = np.zeros((n**3, n**3), dtype=float)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xx, yy, zz = x + d[0], y + d[1], z + d[2]
                    if xx < n and yy < n and zz < n:
                        j = idx(xx, yy, zz, n)
                        ham[i, j] = ham[j, i] = -1.0
    return ham


def occupation_transfer(ham, n, src, sigma, alpha):
    ev, vecs = np.linalg.eigh(ham)
    occ = ev < 0.0
    uo, uu = vecs[:, occ], vecs[:, ~occ]
    env = np.zeros((n**3,), dtype=float)
    stag = np.zeros((n**3,), dtype=float)
    sx, sy, sz = src
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                rr = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                env[i] = np.exp(-0.5 * rr / (sigma * sigma))
                stag[i] = (1.0 - 2.0 * ((x + y + z) % 2)) * env[i]
    left = uo @ (uo.T @ env)
    right = uu @ (uu.T @ stag)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    c0 = uo @ uo.T
    corr = c0 + alpha * (np.outer(right, right) - np.outer(left, left))
    return c0, 0.5 * (corr + corr.T)


def peschel_s(corr, sl):
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def in_ball(src, center, radius):
    return sum((src[k] - center[k]) ** 2 for k in range(3)) <= radius * radius


def dst_poisson(rhs):
    nbox = rhs.shape[0]
    xs = np.linspace(-LBOX, LBOX, nbox)
    h = float(xs[1] - xs[0])
    m = nbox - 2
    fhat = rhs[1:-1, 1:-1, 1:-1].copy()
    for ax in (0, 1, 2):
        fhat = dst(fhat, type=1, axis=ax)
    j = np.arange(1, m + 1, dtype=float)
    lam1 = -4.0 / (h * h) * np.sin(np.pi * j / (2.0 * (m + 1))) ** 2
    lam = lam1[:, None, None] + lam1[None, :, None] + lam1[None, None, :]
    phat = fhat / lam
    for ax in (0, 1, 2):
        phat = dst(phat, type=1, axis=ax)
    phi = np.zeros((nbox, nbox, nbox), dtype=float)
    phi[1:-1, 1:-1, 1:-1] = phat / (2.0 * (m + 1)) ** 3
    return xs, h, phi


def interp(xs, field, p):
    def one(ax, val):
        if val <= ax[0]:
            return 0, 0.0
        if val >= ax[-1]:
            return len(ax) - 2, 1.0
        k = int(np.searchsorted(ax, val) - 1)
        k = max(0, min(k, len(ax) - 2))
        return k, (val - ax[k]) / (ax[k + 1] - ax[k])

    i, tx = one(xs, p[0])
    j, ty = one(xs, p[1])
    k, tz = one(xs, p[2])
    acc = 0.0
    for di, wi in ((0, 1 - tx), (1, tx)):
        for dj, wj in ((0, 1 - ty), (1, ty)):
            for dk, wk in ((0, 1 - tz), (1, tz)):
                acc += wi * wj * wk * field[i + di, j + dj, k + dk]
    return float(acc)


def newton_from_rhs(rhs):
    xs, h, phi = dst_poisson(rhs)
    ax = np.zeros_like(phi)
    ax[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * h)
    accs = [interp(xs, ax, (r, 0.0, 0.0)) for r in PROBES]
    lr = np.log(np.asarray(PROBES, float))
    la = np.log(np.abs(np.asarray(accs, float)))
    slope = float(
        np.linalg.lstsq(np.column_stack([lr, np.ones(3)]), la, rcond=None)[0][0]
    )
    return xs, h, accs, slope


def deposit(de, n, src, scale, nbox):
    xs = np.linspace(-LBOX, LBOX, nbox)
    h = float(xs[1] - xs[0])
    rhs = np.zeros((nbox, nbox, nbox), dtype=float)
    deposited = 0.0
    sx, sy, sz = src
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                px = (x - sx) * scale
                py = (y - sy) * scale
                pz = (z - sz) * scale
                if max(abs(px), abs(py), abs(pz)) >= LBOX - 0.5 * h:
                    continue
                ix = int(np.argmin(np.abs(xs - px)))
                iy = int(np.argmin(np.abs(xs - py)))
                iz = int(np.argmin(np.abs(xs - pz)))
                if min(ix, iy, iz) <= 0 or max(ix, iy, iz) >= nbox - 1:
                    continue
                rhs[ix, iy, iz] += 4.0 * np.pi * GCONST * de[i] / (h**3)
                deposited += de[i]
    return rhs, float(deposited)


def residuals(accs, mass):
    if mass is None or abs(mass) < 1e-18:
        return [float("nan")] * len(accs)
    return [abs(abs(a) * r * r / (GCONST * mass) - 1.0) for a, r in zip(accs, PROBES)]


def main() -> int:
    ham = hop_H(N)
    c0, c1 = occupation_transfer(ham, N, SRC, SIGMA, ALPHA)
    de = np.sum(ham * c1, axis=1) - np.sum(ham * c0, axis=1)
    r2max = RADIUS * RADIUS
    centers = [
        (x, y, z)
        for x in range(RADIUS, N - RADIUS)
        for y in range(RADIUS, N - RADIUS)
        for z in range(RADIUS, N - RADIUS)
    ]
    ds, pflat, inside = [], [], []
    for cx, cy, cz in centers:
        sl = np.array(
            [
                idx(x, y, z)
                for x in range(N)
                for y in range(N)
                for z in range(N)
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r2max
            ],
            dtype=int,
        )
        ds.append(peschel_s(c1, sl) - peschel_s(c0, sl))
        pflat.append(float(np.sum(de[sl])))
        inside.append(in_ball(SRC, (cx, cy, cz), RADIUS))
    ds, pflat = map(np.asarray, (ds, pflat))
    inside = np.asarray(inside, bool)
    well = inside & (np.abs(pflat) > 1e-6)
    rng = np.random.default_rng(SEED)
    widx = np.flatnonzero(well)
    rng.shuffle(widx)
    half = len(widx) // 2
    even, odd = widx[:half], widx[half:]
    kappa = float(np.median(ds[even] / pflat[even]))
    m_hat = float(np.median(ds[odd] / kappa))
    m_enc = float(np.median(pflat[odd]))
    m_global = float(np.sum(de))

    sx, sy, sz = SRC
    spacing_near = 2.0 * LBOX / (N - 1)
    rhs_near, m_near = deposit(de, N, SRC, spacing_near, NBOX)
    _, _, accs_near, slope_near = newton_from_rhs(rhs_near)

    scale_far = R_COMPACT / R_PACK
    rhs_far, m_far = deposit(de, N, SRC, scale_far, NBOX)
    _, h, accs_far, slope_far = newton_from_rhs(rhs_far)
    res_hat = residuals(accs_far, m_hat)
    res_glob = residuals(accs_far, m_global)
    mean = {
        "M_hat": float(np.mean(res_hat)),
        "M_global": float(np.mean(res_glob)),
    }
    winner = min(mean, key=mean.get)
    attractive = all(a < 0.0 for a in accs_far)
    c_hat = bool(
        attractive and all(e < GATE_FAR for e in res_hat) and abs(slope_far + 2.0) < 0.08
    )
    c_glob = bool(
        attractive and all(e < GATE_FAR for e in res_glob) and abs(slope_far + 2.0) < 0.08
    )

    rhs_pt = np.zeros((NBOX, NBOX, NBOX), dtype=float)
    mid = NBOX // 2
    rhs_pt[mid, mid, mid] = 4.0 * np.pi * GCONST * m_hat / (h**3)
    _, _, accs_pt, slope_pt = newton_from_rhs(rhs_pt)
    res_pt = residuals(accs_pt, m_hat)
    c_ctrl = bool(
        all(a < 0.0 for a in accs_pt)
        and all(e < GATE_FAR for e in res_pt)
        and abs(slope_pt + 2.0) < 0.08
    )

    shells = {}
    for x in range(N):
        for y in range(N):
            for z in range(N):
                r = int(np.floor(np.sqrt((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2) + 1e-12))
                shells.setdefault(r, 0.0)
                shells[r] += de[idx(x, y, z)]
    cum = 0.0
    radial = []
    for r in sorted(shells):
        cum += shells[r]
        radial.append({"r_site": r, "sum": float(shells[r]), "cum": float(cum)})

    kappa_r = {}
    for rad in (2, 3, 4):
        sl = np.array(
            [
                idx(x, y, z)
                for x in range(N)
                for y in range(N)
                for z in range(N)
                if (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2 <= rad * rad
            ],
            dtype=int,
        )
        p = float(np.sum(de[sl]))
        dsr = peschel_s(c1, sl) - peschel_s(c0, sl)
        kappa_r[str(rad)] = {
            "deltaS": dsr,
            "P_flat": p,
            "kappa": (dsr / p) if abs(p) > 1e-18 else None,
        }

    verdict_map = {
        "M_hat": "FIRST_LAW_IS_SOURCE",
        "M_global": "TOTAL_ENERGY_IS_SOURCE",
    }
    if not c_ctrl:
        verdict = "SOLVER_BROKEN"
    elif c_glob and not c_hat:
        verdict = "TOTAL_ENERGY_IS_SOURCE"
    elif c_hat and not c_glob:
        verdict = "FIRST_LAW_IS_SOURCE"
    else:
        verdict = verdict_map[winner]
    payload = {
        "task": "m9.39_tail",
        "kappa": kappa,
        "M_hat": m_hat,
        "M_enc": m_enc,
        "M_global": m_global,
        "M_deposited_far": m_far,
        "M_deposited_near": m_near,
        "hat_vs_global": abs(m_hat / m_global - 1.0) if m_global else None,
        "near_field": {
            "spacing": spacing_near,
            "a_r": accs_near,
            "slope": slope_near,
            "note": "probes inside packet; not a 1/r2 test",
        },
        "far_field": {
            "scale": scale_far,
            "a_r": accs_far,
            "slope": slope_far,
            "res_hat": res_hat,
            "res_global": res_glob,
            "mean_res": mean,
        },
        "radial_de": radial,
        "winner": winner,
        "C_hat": c_hat,
        "C_global": c_glob,
        "C_which_PRIMARY": winner,
        "control_point": {
            "a_r": accs_pt,
            "res_hat": res_pt,
            "slope": slope_pt,
            "pass": c_ctrl,
        },
        "kappa_of_R": kappa_r,
        "C_ctrl": c_ctrl,
        "verdict": verdict,
        "not_claimed": [
            "derived Poisson",
            "8pi G from kappa",
            "1/4G",
            "FGHMV",
            "de Sitter",
            "MODELS.md",
        ],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_39_tail.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c_ctrl and c_glob and not c_hat) else 1


if __name__ == "__main__":
    raise SystemExit(main())
