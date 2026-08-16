#!/usr/bin/env python3
"""M9.62: pair first law (rank-4 S) and discrete div g.

1) Orthonormal two-source C. If La,Lb ⊥ Ra,Rb and
   orthonormal, C has moving eigenvalues
   α_A, 1-α_A, α_B, 1-α_B, so

       S_global = 2h(α_A)+2h(α_B).

   Verified from the 4×4 Gram and C in that basis.
   Then on R=3 balls:

       f_S = δS / S_global,   f_E = M_AB / M_tot

   Peschel of each ball block is mpmath eigsy.

2) Derived g = −(M/A) n, n = ∇M/|∇M|, at R=2.
   Lattice divergence of that g. Sources vs midpoint.

Exact open-hop basis. mpmath dps=80. No hop LAPACK.

PRE-REGISTERED:
  N=12. A=(3,6,6), B=(9,6,6), σ=1.
  α_A=0.02, α_B=0.04.
  C_gram  max |⟨ui|uj⟩−δij| < 1e-20
  C_c4    max |off-diag ⟨ui|C|uj⟩| < 1e-20
  C_sg    |S_4 − 2h(α_A)−2h(α_B)| / S_4 < 1e-20
  Enclosure on every two-sided R=3 centre (4³=64).
  C_rho   ρ(f_S, f_E) > 0.90
  C_rms   RMS(f_S−f_E) < 0.15
  C_both  balls containing both source sites:
          median |f_S−1| < 0.10 and |f_E−1| < 0.10
  Div on R=2 field, two-sided g, interior centres.
  C_src   max |div| at centres nearest A or B
          > 2 × |div| at (6,6,6)

Not claimed: derived Einstein, 8πG, de Sitter, MODELS.md.

Writes ../data/m9_62_pair_ent.json
"""

from __future__ import annotations

import json
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m9_60_exact as ex  # noqa: E402

DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
mp.mp.dps = 80
N = 12
SRC_A = (3, 6, 6)
SRC_B = (9, 6, 6)
MID = (6, 6, 6)
SIGMA = mp.mpf(1)
ALPHA_A = mp.mpf("0.02")
ALPHA_B = mp.mpf("0.04")
R_ENC = 3
R_DIV = 2
CLIP = mp.mpf("1e-30")
GRAD_FRAC = mp.mpf("1e-6")


def h_bin(a):
    if a <= 0 or a >= 1:
        return mp.mpf(0)
    return -a * mp.log(a) - (1 - a) * mp.log(1 - a)


def vec_at(field, sites):
    return [field[x][y][z] for (x, y, z) in sites]


def gram4(vecs):
    g = mp.zeros(4)
    for i in range(4):
        for j in range(4):
            g[i, j] = ex.dot3(vecs[i], vecs[j])
    return g


def c_in_basis(la, lb, ra, rb, aa, ab):
    # ⟨u|C|v⟩ = ⟨u|C0|v⟩ + aa(⟨u|Ra⟩⟨Ra|v⟩−⟨u|La⟩⟨La|v⟩)
    #          + ab(⟨u|Rb⟩⟨Rb|v⟩−⟨u|Lb⟩⟨Lb|v⟩)
    # C0 = occ projector: C0 La=La, C0 Lb=Lb, C0 Ra=0, C0 Rb=0
    basis = [la, lb, ra, rb]
    names_occ = [True, True, False, False]
    c = mp.zeros(4)
    for i in range(4):
        for j in range(4):
            val = mp.mpf(0)
            if names_occ[i] and names_occ[j]:
                val += ex.dot3(basis[i], basis[j])
            val += aa * (
                ex.dot3(basis[i], ra) * ex.dot3(ra, basis[j])
                - ex.dot3(basis[i], la) * ex.dot3(la, basis[j])
            )
            val += ab * (
                ex.dot3(basis[i], rb) * ex.dot3(rb, basis[j])
                - ex.dot3(basis[i], lb) * ex.dot3(lb, basis[j])
            )
            c[i, j] = val
    return c


def max_off_abs(mat):
    m = mp.mpf(0)
    n = mat.rows
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a = abs(mat[i, j])
            if a > m:
                m = a
    return m


def max_gram_err(g):
    m = mp.mpf(0)
    for i in range(4):
        for j in range(4):
            target = mp.mpf(1) if i == j else mp.mpf(0)
            a = abs(g[i, j] - target)
            if a > m:
                m = a
    return m


def peschel_s(block):
    ev, _ = mp.eigsy(block)
    acc = mp.mpf(0)
    for i in range(ev.rows):
        w = ev[i]
        if w < CLIP:
            w = CLIP
        if w > 1 - CLIP:
            w = 1 - CLIP
        acc -= w * mp.log(w) + (1 - w) * mp.log(1 - w)
    return acc


def occupied_list(eps):
    n = len(eps)
    out = []
    for a in range(n):
        for b in range(n):
            for c in range(n):
                if eps[a] + eps[b] + eps[c] < 0:
                    out.append((a, b, c))
    return out


def c0_block(psi, occ_modes, sites):
    n_b = len(sites)
    n_o = len(occ_modes)
    v = mp.matrix(n_b, n_o)
    for k, (nx, ny, nz) in enumerate(occ_modes):
        for i, (x, y, z) in enumerate(sites):
            v[i, k] = psi[x][nx] * psi[y][ny] * psi[z][nz]
    return v * v.T


def dC_block(la, ra, lb, rb, aa, ab, sites):
    va = vec_at(la, sites)
    wa = vec_at(ra, sites)
    vb = vec_at(lb, sites)
    wb = vec_at(rb, sites)
    n_b = len(sites)
    d = mp.zeros(n_b)
    for i in range(n_b):
        for j in range(n_b):
            d[i, j] = aa * (wa[i] * wa[j] - va[i] * va[j]) + ab * (
                wb[i] * wb[j] - vb[i] * vb[j]
            )
    return d


def ball_sites(center, radius, n):
    cx, cy, cz = center
    sites = []
    inside = [[[False] * n for _ in range(n)] for _ in range(n)]
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius * radius:
                    inside[x][y][z] = True
                    sites.append((x, y, z))
    return sites, inside


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = mp.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = mp.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def vnorm(v):
    return mp.sqrt(sum(x * x for x in v))


def field_g(c, masses, areas, m_tot):
    gm = ex.grad_at(c, masses)
    if gm is None:
        return None
    gn = vnorm(gm)
    if gn <= GRAD_FRAC * abs(m_tot):
        return None
    area = mp.mpf(areas[c])
    if area == 0:
        return None
    scale = -masses[c] / area / gn
    return [scale * gm[i] for i in range(3)]


def main() -> int:
    ex.N = N
    psi, eps = ex.open_1d(N)
    mask, n_occ = ex.occupy_mask(eps)
    occ_modes = occupied_list(eps)
    la, ra = ex.packet_lr(psi, eps, mask, SRC_A, SIGMA)
    lb, rb = ex.packet_lr(psi, eps, mask, SRC_B, SIGMA)
    la, lb = ex.orthonormalize(la, lb)
    ra, rb = ex.orthonormalize(ra, rb)
    de = ex.site_de([(la, ra, ALPHA_A), (lb, rb, ALPHA_B)])
    m_tot = sum(de, mp.mpf(0))

    basis = [la, lb, ra, rb]
    g4 = gram4(basis)
    c4 = c_in_basis(la, lb, ra, rb, ALPHA_A, ALPHA_B)
    gram_err = max_gram_err(g4)
    off_c = max_off_abs(c4)
    ev4, _ = mp.eigsy((c4 + c4.T) / 2)
    s4 = mp.mpf(0)
    for i in range(4):
        w = ev4[i]
        if w < CLIP:
            w = CLIP
        if w > 1 - CLIP:
            w = 1 - CLIP
        s4 -= w * mp.log(w) + (1 - w) * mp.log(1 - w)
    s_th = 2 * h_bin(ALPHA_A) + 2 * h_bin(ALPHA_B)
    sg_rel = abs(s4 - s_th) / s_th if s_th != 0 else abs(s4)

    lo, hi = R_ENC, N - R_ENC
    centres = [
        (x, y, z)
        for x in range(lo, hi)
        for y in range(lo, hi)
        for z in range(lo, hi)
    ]
    cset = set(centres)
    keep = []
    for c in centres:
        okc = True
        for ax in range(3):
            cp, cm = list(c), list(c)
            cp[ax] += 1
            cm[ax] -= 1
            if tuple(cp) not in cset or tuple(cm) not in cset:
                okc = False
        if okc:
            keep.append(c)

    rows = []
    f_s, f_e = [], []
    both_s, both_e = [], []
    for c in keep:
        sites, inside = ball_sites(c, R_ENC, N)
        sl = [ex.idx(x, y, z, N) for (x, y, z) in sites]
        mass = sum((de[i] for i in sl), mp.mpf(0))
        c0b = c0_block(psi, occ_modes, sites)
        dcb = dC_block(la, ra, lb, rb, ALPHA_A, ALPHA_B, sites)
        s0 = peschel_s(c0b)
        s1 = peschel_s(c0b + dcb)
        ds = s1 - s0
        fs = ds / s4
        fe = mass / m_tot
        f_s.append(fs)
        f_e.append(fe)
        sa = (c[0] - SRC_A[0]) ** 2 + (c[1] - SRC_A[1]) ** 2 + (c[2] - SRC_A[2]) ** 2
        sb = (c[0] - SRC_B[0]) ** 2 + (c[1] - SRC_B[1]) ** 2 + (c[2] - SRC_B[2]) ** 2
        both = sa <= R_ENC * R_ENC and sb <= R_ENC * R_ENC
        if both:
            both_s.append(abs(fs - 1))
            both_e.append(abs(fe - 1))
        rows.append(
            {
                "c": list(c),
                "n_sites": len(sites),
                "M": ex.fnum(mass),
                "dS": ex.fnum(ds),
                "f_S": ex.fnum(fs),
                "f_E": ex.fnum(fe),
                "both": both,
            }
        )

    rho = pearson(f_s, f_e)
    rms = mp.sqrt(sum((f_s[i] - f_e[i]) ** 2 for i in range(len(f_s))) / len(f_s))
    med_s = ex.mp_median(both_s)
    med_e = ex.mp_median(both_e)

    # --- part 2: div g at R=2 ---
    map_m, map_a = ex.mass_map(de, R_DIV, N)
    gs = {}
    for c in map_m:
        gv = field_g(c, map_m, map_a, m_tot)
        if gv is not None:
            gs[c] = gv

    def div_at(c):
        acc = mp.mpf(0)
        for ax in range(3):
            cp, cm = list(c), list(c)
            cp[ax] += 1
            cm[ax] -= 1
            tp, tm = tuple(cp), tuple(cm)
            if tp not in gs or tm not in gs:
                return None
            acc += (gs[tp][ax] - gs[tm][ax]) / 2
        return acc

    divs = {}
    for c in gs:
        d = div_at(c)
        if d is not None:
            divs[c] = d
    d_mid = divs.get(MID)
    # nearest axis centres to A and B that have div
    axis = [c for c in divs if c[1] == 6 and c[2] == 6]
    c_a = min(axis, key=lambda c: abs(c[0] - SRC_A[0])) if axis else None
    c_b = min(axis, key=lambda c: abs(c[0] - SRC_B[0])) if axis else None
    d_a = divs.get(c_a) if c_a else None
    d_b = divs.get(c_b) if c_b else None
    src_max = None
    if d_a is not None and d_b is not None:
        src_max = max(abs(d_a), abs(d_b))
    elif d_a is not None:
        src_max = abs(d_a)
    elif d_b is not None:
        src_max = abs(d_b)

    c_gram = bool(gram_err < mp.mpf("1e-20"))
    c_c4 = bool(off_c < mp.mpf("1e-20"))
    c_sg = bool(sg_rel < mp.mpf("1e-20"))
    c_rho = bool(rho is not None and abs(rho) > mp.mpf("0.90"))
    c_rms = bool(rms < mp.mpf("0.15"))
    c_both = bool(
        both_s
        and med_s is not None
        and med_e is not None
        and med_s < mp.mpf("0.10")
        and med_e < mp.mpf("0.10")
    )
    c_src = bool(
        src_max is not None and d_mid is not None and src_max > 2 * abs(d_mid)
    )
    ok = bool(c_gram and c_c4 and c_sg and c_rho and c_rms and c_both)
    payload = {
        "task": "m9.62_pair_ent",
        "precision": {
            "basis": "closed-form 1d open hop",
            "eigh_hop": "none",
            "peschel": "mpmath eigsy dps=80",
            "n_occ": n_occ,
        },
        "S_4": ex.fnum(s4),
        "S_theory": ex.fnum(s_th),
        "S_rel": ex.fnum(sg_rel),
        "gram_err": ex.fnum(gram_err),
        "C4_off": ex.fnum(off_c),
        "M_AB": ex.fnum(m_tot),
        "n_balls": len(rows),
        "n_both": len(both_s),
        "rho": ex.fnum(rho),
        "rms": ex.fnum(rms),
        "med_abs_fS_1": ex.fnum(med_s),
        "med_abs_fE_1": ex.fnum(med_e),
        "rows": rows,
        "div": {
            "n_g": len(gs),
            "n_div": len(divs),
            "c_near_A": list(c_a) if c_a else None,
            "c_near_B": list(c_b) if c_b else None,
            "div_near_A": ex.fnum(d_a),
            "div_near_B": ex.fnum(d_b),
            "div_mid": ex.fnum(d_mid),
            "src_over_mid": ex.fnum(src_max / abs(d_mid))
            if (src_max is not None and d_mid not in (None, 0))
            else None,
        },
        "C_gram": c_gram,
        "C_c4": c_c4,
        "C_sg": c_sg,
        "C_rho": c_rho,
        "C_rms": c_rms,
        "C_both": c_both,
        "C_src": c_src,
        "all_enc_gates": ok,
        "verdict": "PAIR_ENT_AND_DIV" if (ok and c_src) else (
            "PAIR_ENT_ONLY" if ok else "PAIR_ENT_FAIL"
        ),
        "not_claimed": ["derived Einstein", "8pi G", "de Sitter", "MODELS.md"],
    }
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "m9_62_pair_ent.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, indent=2, fp=fh)
        fh.write("\n")
    print(json.dumps({k: payload[k] for k in payload if k != "rows"}, indent=2))
    print("n_rows", len(rows))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
