#!/usr/bin/env python3
"""M9.44 audit. N=10 PBC, α=0.03, own band-edge transfer.

Tries to REFUTE C_unif and C_lin.

Writes ../data/m9_44_audit_pbc.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.fft import dst

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLIP = 1e-12
N, ALPHA = 10, 0.03
RADII = (2, 3, 4)
SRC = (5, 5, 5)
LBOX, NBOX, GCONST = 1.0, 65, 1.0
PROBES = (0.10 * LBOX, 0.15 * LBOX, 0.20 * LBOX)


def idx(x, y, z, n=N):
    return (x * n + y) * n + z


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


def main() -> int:
    ham = np.zeros((N**3, N**3))
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    j = idx((x + d[0]) % N, (y + d[1]) % N, (z + d[2]) % N)
                    ham[i, j] = ham[j, i] = -1.0
    ev, vecs = np.linalg.eigh(ham)
    i_l, i_r = int(np.argmin(ev)), int(np.argmax(ev))
    left, right = vecs[:, i_l], vecs[:, i_r]
    occ = ev < 0.0
    c0 = vecs[:, occ] @ vecs[:, occ].T
    c1 = 0.5 * (
        (c0 + ALPHA * (np.outer(right, right) - np.outer(left, left)))
        + (c0 + ALPHA * (np.outer(right, right) - np.outer(left, left))).T
    )
    de = np.sum(ham * c1, axis=1) - np.sum(ham * c0, axis=1)
    unif = float(np.std(de) / np.mean(np.abs(de)))
    c_unif = bool(unif < 0.05)
    ds, pf = [], []
    sx, sy, sz = SRC
    for rad in RADII:
        sl = []
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    dx = min((x - sx) % N, (sx - x) % N)
                    dy = min((y - sy) % N, (sy - y) % N)
                    dz = min((z - sz) % N, (sz - z) % N)
                    if dx * dx + dy * dy + dz * dz <= rad * rad:
                        sl.append(idx(x, y, z))
        sl = np.array(sl, dtype=int)
        ds.append(peschel_s(c1, sl) - peschel_s(c0, sl))
        pf.append(float(np.sum(de[sl])))
    ds, pf = np.asarray(ds), np.asarray(pf)
    rho = pearson(ds, pf)
    grow = float(ds[-1] / ds[0])
    xs = np.linspace(-LBOX, LBOX, NBOX)
    h = float(xs[1] - xs[0])
    n_int = NBOX - 2
    rho = float(np.sum(de)) / ((n_int * h) ** 3)
    rhs = np.zeros((NBOX, NBOX, NBOX))
    rhs[1:-1, 1:-1, 1:-1] = 4.0 * np.pi * GCONST * rho
    m = NBOX - 2
    fhat = rhs[1:-1, 1:-1, 1:-1].copy()
    for axv in (0, 1, 2):
        fhat = dst(fhat, type=1, axis=axv)
    j = np.arange(1, m + 1, dtype=float)
    lam1 = -4.0 / (h * h) * np.sin(np.pi * j / (2.0 * (m + 1))) ** 2
    lam = lam1[:, None, None] + lam1[None, :, None] + lam1[None, None, :]
    phat = fhat / lam
    for axv in (0, 1, 2):
        phat = dst(phat, type=1, axis=axv)
    phi = np.zeros((NBOX, NBOX, NBOX))
    phi[1:-1, 1:-1, 1:-1] = phat / (2.0 * (m + 1)) ** 3
    ax = np.zeros_like(phi)
    ax[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * h)

    def interp(field, p):
        def one(axg, val):
            k = int(np.searchsorted(axg, val) - 1)
            k = max(0, min(k, len(axg) - 2))
            return k, (val - axg[k]) / (axg[k + 1] - axg[k])

        i, tx = one(xs, p[0])
        jy, ty = one(xs, p[1])
        k, tz = one(xs, p[2])
        acc = 0.0
        for di, wi in ((0, 1 - tx), (1, tx)):
            for dj, wj in ((0, 1 - ty), (1, ty)):
                for dk, wk in ((0, 1 - tz), (1, tz)):
                    acc += wi * wj * wk * field[i + di, jy + dj, k + dk]
        return float(acc)

    accs = [interp(ax, (r, 0.0, 0.0)) for r in PROBES]
    lr = np.log(np.array(PROBES, float))
    la = np.log(np.abs(np.array(accs, float)))
    slope = float(np.linalg.lstsq(np.column_stack([lr, np.ones(3)]), la, rcond=None)[0][0])
    c_lin = bool(
        abs(slope - 1.0) < 0.40
        and abs(slope - 1.0) < abs(slope + 2.0)
        and all(a < 0.0 for a in accs)
    )
    c_fl = bool(abs(rho) > 0.95)
    payload = {
        "task": "m9.44_audit_pbc",
        "E_L": float(ev[i_l]),
        "E_R": float(ev[i_r]),
        "uniformity": unif,
        "rho_P": rho,
        "grow": grow,
        "a_r": accs,
        "slope": slope,
        "C_unif": c_unif,
        "C_fl": c_fl,
        "C_lin": c_lin,
        "verdicts": {
            "C_unif": "CONFIRMED" if c_unif else "REFUTED",
            "C_fl": "CONFIRMED" if c_fl else "REFUTED",
            "C_lin": "CONFIRMED" if c_lin else "REFUTED",
        },
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_44_audit_pbc.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    return 0 if (c_unif and c_lin) else 1


if __name__ == "__main__":
    raise SystemExit(main())
