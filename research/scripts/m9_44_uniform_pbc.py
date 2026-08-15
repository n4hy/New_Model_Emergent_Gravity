#!/usr/bin/env python3
"""M9.44: periodic band-edge transfer — uniform δe, volume law, a ∝ r.

Paper 51's wide Gaussian was not a uniform fluid. On a periodic
cube the band-edge hop eigenstates are unique and real:
  L = k=0,     ψ = 1/√V,     E = −6
  R = (π,π,π), ψ = (−1)^{x+y+z}/√V, E = +6
Both have |ψ|² = 1/V. Occupation transfer at fixed H then has

  δe_i = α (E_R − E_L) / V = 12α / V    (exactly constant).

That is a valid fermion state, not a painted ρ.

PRE-REGISTERED:
  N=12, periodic hop, α=0.02, L=min-ev, R=max-ev.
  C_unif  std(δe) / mean(|δe|) < 0.05
  C_eig   C eigenvalues in [0, 1] to 1e-9
  Source-centered (any centre: the density is flat) balls
  R=2,3,4,5. V = site count. A = outgoing NN count (PBC wrap).
  C_fl    Pearson(δS, P_flat) > 0.95
  C_grow  δS(5)/δS(2) > 1.30
  C_dens  rel IQR of P_flat/V  <  rel IQR of P_flat/A
          (energy density, not surface density)
  Continuum image of the measured-uniform δe (not NGP
  spikes): constant ρ on the Dirichlet interior, n=65, L=1, G=1.
  Probes r=0.10L, 0.15L, 0.20L.
  C_lin PRIMARY (inherited Poisson).
      log-log slope α of |a| vs r: |α − 1| < 0.40
      and |α − 1| < |α + 2|
      and a_r < 0
  C_invsq must FAIL: |α + 2| > 0.50

Not claimed: derived Poisson, 8πG, FGHMV, de Sitter dual,
MODELS.md. a ∝ r is the Newtonian interior of a uniform
source / Newtonian Λ signature of this state.

Writes ../data/m9_44_uniform_pbc.json
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
ALPHA = 0.02
RADII = (2, 3, 4, 5)
SRC = (6, 6, 6)
LBOX = 1.0
NBOX = 65
GCONST = 1.0
PROBES = (0.10 * LBOX, 0.15 * LBOX, 0.20 * LBOX)


def idx(x, y, z, n=N) -> int:
    return (x * n + y) * n + z


def hop_H_pbc(n: int) -> np.ndarray:
    ham = np.zeros((n**3, n**3), dtype=float)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                i = idx(x, y, z, n)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    j = idx((x + d[0]) % n, (y + d[1]) % n, (z + d[2]) % n, n)
                    ham[i, j] = ham[j, i] = -1.0
    return ham


def peschel_s(corr, sl):
    w = np.clip(np.linalg.eigvalsh(corr[np.ix_(sl, sl)]), CLIP, 1.0 - CLIP)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return float(np.dot(a, b) / den)


def rel_iqr(vals):
    vals = np.asarray(vals, float)
    med = float(np.median(vals))
    if abs(med) < 1e-18:
        return None
    q1, q3 = np.percentile(vals, [25.0, 75.0])
    return float((q3 - q1) / abs(med))


def ball_and_area(center, radius, n=N):
    cx, cy, cz = center
    inside = np.zeros((n, n, n), dtype=bool)
    sl = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                dx = min((x - cx) % n, (cx - x) % n)
                dy = min((y - cy) % n, (cy - y) % n)
                dz = min((z - cz) % n, (cz - z) % n)
                if dx * dx + dy * dy + dz * dz <= radius * radius:
                    inside[x, y, z] = True
                    sl.append(idx(x, y, z, n))
    area = 0
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if not inside[x, y, z]:
                    continue
                for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    xx, yy, zz = (x + d[0]) % n, (y + d[1]) % n, (z + d[2]) % n
                    if not inside[xx, yy, zz]:
                        area += 1
    return np.array(sl, dtype=int), int(np.sum(inside)), area


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


def main() -> int:
    ham = hop_H_pbc(N)
    ev, vecs = np.linalg.eigh(ham)
    left = vecs[:, int(np.argmin(ev))]
    right = vecs[:, int(np.argmax(ev))]
    occ = ev < 0.0
    # k=0 must be occupied, (π,π,π) unoccupied
    i_left = int(np.argmin(ev))
    i_right = int(np.argmax(ev))
    if not occ[i_left] or occ[i_right]:
        raise RuntimeError("band edges not on opposite sides of the Fermi level")
    c0 = vecs[:, occ] @ vecs[:, occ].T
    corr = c0 + ALPHA * (np.outer(right, right) - np.outer(left, left))
    c1 = 0.5 * (corr + corr.T)
    eigs = np.linalg.eigvalsh(c1)
    c_eig = bool(eigs.min() >= -1e-9 and eigs.max() <= 1.0 + 1e-9)
    de = np.sum(ham * c1, axis=1) - np.sum(ham * c0, axis=1)
    mean_abs = float(np.mean(np.abs(de)))
    unif = float(np.std(de) / mean_abs) if mean_abs else None
    c_unif = bool(unif is not None and unif < 0.05)
    m_glob = float(np.sum(de))

    ds, pflat, vol, area = [], [], [], []
    for rad in RADII:
        sl, v, a = ball_and_area(SRC, rad)
        ds.append(peschel_s(c1, sl) - peschel_s(c0, sl))
        pflat.append(float(np.sum(de[sl])))
        vol.append(v)
        area.append(a)
    ds, pflat, vol, area = map(np.asarray, (ds, pflat, vol, area))
    rho_p = pearson(ds, pflat)
    grow = float(ds[-1] / ds[0]) if ds[0] else None
    dens = pflat / vol
    surf = pflat / area
    iqr_d, iqr_s = rel_iqr(dens), rel_iqr(surf)
    c_fl = bool(abs(rho_p) > 0.95)
    c_grow = bool(grow is not None and grow > 1.30)
    c_dens = bool(iqr_d is not None and iqr_s is not None and iqr_d < iqr_s)

    xs = np.linspace(-LBOX, LBOX, NBOX)
    h = float(xs[1] - xs[0])
    # C_unif says δe is constant. Continuum image: uniform ρ, not NGP spikes.
    n_int = NBOX - 2
    rho = m_glob / ((n_int * h) ** 3)
    rhs = np.zeros((NBOX, NBOX, NBOX), dtype=float)
    rhs[1:-1, 1:-1, 1:-1] = 4.0 * np.pi * GCONST * rho
    _, h, phi = dst_poisson(rhs)
    ax = np.zeros_like(phi)
    ax[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * h)
    accs = [interp(xs, ax, (r, 0.0, 0.0)) for r in PROBES]
    lr = np.log(np.asarray(PROBES, float))
    la = np.log(np.abs(np.asarray(accs, float)))
    slope = float(
        np.linalg.lstsq(np.column_stack([lr, np.ones(3)]), la, rcond=None)[0][0]
    )
    c_lin = bool(
        abs(slope - 1.0) < 0.40
        and abs(slope - 1.0) < abs(slope + 2.0)
        and all(a < 0.0 for a in accs)
    )
    c_invsq_fail = bool(abs(slope + 2.0) > 0.50)
    ok = bool(c_unif and c_eig and c_fl and c_grow and c_lin and c_invsq_fail)
    payload = {
        "task": "m9.44_uniform_pbc",
        "E_L": float(ev[i_left]),
        "E_R": float(ev[i_right]),
        "M_global": m_glob,
        "uniformity": unif,
        "deltaS": ds.tolist(),
        "P_flat": pflat.tolist(),
        "V": vol.tolist(),
        "A": area.tolist(),
        "rho_P": rho_p,
        "grow": grow,
        "iqr_P_over_V": iqr_d,
        "iqr_P_over_A": iqr_s,
        "a_r": accs,
        "slope": slope,
        "C_unif": c_unif,
        "C_eig": c_eig,
        "C_fl": c_fl,
        "C_grow": c_grow,
        "C_dens": c_dens,
        "C_lin_PRIMARY": c_lin,
        "C_invsq_FAIL": c_invsq_fail,
        "all_gates": ok,
        "verdict": "UNIFORM_NEWTON_LAMBDA" if ok else "PBC_UNIFORM_FAIL",
        "not_claimed": [
            "derived Poisson",
            "8pi G",
            "FGHMV",
            "de Sitter dual",
            "MODELS.md",
        ],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_44_uniform_pbc.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
