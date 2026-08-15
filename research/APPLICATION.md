# New Model application: Emergent Gravity / NSM

> Posted as
> [discussion #442](https://github.com/openwave-labs/openwave/discussions/442)
> in the
> [New Model](https://github.com/openwave-labs/openwave/discussions/categories/new-model)
> category. This is the author's application, not a maintainer admission.
> First PR (proposed scaffold + Hehl-Datta gate, no `MODELS.md` edit):
> [openwave-labs/openwave#441](https://github.com/openwave-labs/openwave/pull/441).

## Model name

Emergent Gravity / New Standard Model (NSM). Proposed short column name: M9
(assignment is the maintainer's).

## Author

Dr. Robert W. McGwier, PhD. CTO, Cohere Technology Group. Sole author of the
14-15 August 2026 series.

## Author contact

GitHub: [@n4hy](https://github.com/n4hy)

## Lineage

Entanglement first-law gravity (Faulkner-Guica-Hartman-Myers-Van Raamsdonk,
JHEP 03 (2014) 051); Casini-Huerta-Myers modular Hamiltonians; Einstein-Cartan
(Kibble, Sciama, Hehl); Hehl-Datta 1971 four-fermion contact; Jacobson
entanglement equilibrium (nonlinear, tagged conjectural).

## Bedrock papers

Public record: [github.com/n4hy/New_Model_Emergent_Gravity](https://github.com/n4hy/New_Model_Emergent_Gravity)
(`research/`, CC-BY-4.0). Author-stated Zenodo DOIs are listed in
[`theory/_CITATIONS.md`](../theory/_CITATIONS.md). Specification of record for
the *effective* theory is Paper III, action (2). Papers I-II are the
entanglement / spin-current argument. Papers IV-VIII are the finite-ball and
\(I_B\) campaign. Paper IX is the domain-of-validity note. Document 10 is the
authoritative label table.

Which paper backs which claim:

| Claim | Paper |
| --- | --- |
| Linearized Einstein from ball first law (AdS) | I, citing FGHMV 2014 |
| Axial / spin-current selection | II |
| NSM action and HD term as the assembly | III |
| Condition NL, absorbability | IV |
| HD sign / magnitude from information | V, VII |
| \(I_B\) non-vanishing, digits open | VI, VIII |
| AdS closed, dS open | IX, Final Status |

## Substrate

Coframe \(e^a\) and independent Lorentz connection \(\omega^{ab}\) on a
4-manifold, plus the installed SM field content. "None, this is not a
dynamical lattice medium" is also accurate for OpenWave's particle rows.

## Dynamics

Tier 1: Palatini Einstein-Cartan plus SM, \(\kappa=8\pi G\). Torsion is
algebraic. The holographic certification of that gravitational sector is
claimed only in AdS, at linear order.

## Particle

An SM field excitation. Not a topological defect and not a time-periodic
soliton.

## Charge

Installed SM charge. Quantization is not derived.

## Free parameters ledger

| Category | Contents |
| --- | --- |
| Inputs | \(SU(3)_c\times SU(2)_L\times U(1)_Y\), three generations, Higgs sector, Yukawas, \(G\), \(\Lambda\). Neutrino Dirac vs Majorana empirically open |
| Calibration targets | None claimed for the HD coefficient |
| Predictions | One structural output beyond SM+GR: \(\mathcal{L}_{\mathrm{HD}}=-\frac{3\kappa}{16}J_5\cdot J_5\). Four named consequences, three tagged `[C]` (bounce, GME as a live test, IR/Milgrom if a dS extension exists) |

Zero free parameters is **not** claimed. The SM is not a prediction.

## Honest residuals

- Multi-digit coefficient of \(I_B\): open as a positive number, and now a
  documented in-column negative on three extraction routes
  (`m9_2_ib_hadamard.py` `FAILED_MULTI_DIGIT`; `m9_3_ib_analytic.py`
  `NOT_UNIVERSAL`; `m9_4_ib_hadamard_complete.py`
  `HADAMARD_COMPLETE_NOT_UNIVERSAL`). The Mittag-Leffler expansion of the
  kernel is complete; \(H(\tau)\) is not proportional to a local kernel, so
  there is no single multi-digit coefficient independent of the source.
- Second-order Einstein-Cartan from entanglement: metric Einstein through
  second order is cited (FHHPRV 2017), not re-derived. Axial / Cartan matching
  is obstructed (`m9_5_ec_symplectic.py` `STRUCTURE_ONLY`): CFT Fisher for a
  conserved current is nonlocal and algebraic torsion has no bulk kinetic
  term. Paper 14. No `[O]` moved to `[P]`.
- de Sitter / cosmology: FGHMV-standard copy is obstructed (Paper 17 /
  M9.6): \(S_{\mathrm{GH}}=3\pi/(G\Lambda)\), \(\partial S/\partial\Lambda<0\),
  opposite to CHM; \(\mathfrak{so}(1,4)\) too small for a CHM net of
  balls. Einstein+\(\Lambda\) from a cosmological CFT is not claimed.
  Jacobson is not a `[P]` substitute (Paper 18): 1995 gives Einstein
  with free \(\Lambda\) and no HD; 2016 conformal half does not apply
  to the SM.
- Nonlinear Einstein-Cartan as a positive holographic theorem: remains
  `[O]`. Closed only as the axial obstruction above.
- UV completion (Q4a): selection-uniqueness is answered in the
  negative (the SM is not a holographic CFT; the certified first law
  does not see \(G_{\mathrm{SM}}\)). Existence of some other pair
  remains open. No CFT was invented
  ([`findings/m9_5_q4a_pair_note.md`](findings/m9_5_q4a_pair_note.md)).
  The unique quadratic axial deformation (Q4b) recovers \(3/16\) in
  the infrared and is Yukawa at finite \(M\); that is a change of
  theory, not a selected completion
  ([`findings/m9_4_uv_deformation_note.md`](findings/m9_4_uv_deformation_note.md)).
- No dark-matter particle; no derivation of masses, mixings, generations,
  strong CP, or \(\Lambda\).
- The HD contact is a 1971 theorem of Einstein-Cartan + Dirac. The distinctive
  claim is the *selection* of Einstein-Cartan, which is not the first
  in-platform task.
- Torsion: algebraic, vacuum-vanishing, non-propagating (EC theorem).
  Spacetime HD is \(\sim G\) and is not a laboratory field. Spintronic
  Berry / SOC geometry is not the Palatini \(\omega\). A late-universe spin
  average is an estimate, not a cosmological no-go.

## Falsifiers

| Signature | Prediction | Current bound | Refutation |
| --- | --- | --- | --- |
| HD four-fermion | strength \(\sim G\), cross-species | collider contact bounds far above | a measured four-fermion coefficient inconsistent with \(3\kappa/16\) (impractical near term) |
| Quantum geometry | gravity carries coherence | Bose / Marletto-Vedral not yet decisive | confirmed null at sufficient sensitivity |
| High-density bounce | torsion bounce replaces the initial singularity | none | `[C]`; a derivation of nonlinear EC that forbids the bounce |
| IR / Milgrom | only if a dS extension exists | SPARC / galactic dynamics | `[C]` and domain-open; not to be scored as a prediction yet |

## Formal artifacts

- Author PDF series and this application:
  [github.com/n4hy/New_Model_Emergent_Gravity](https://github.com/n4hy/New_Model_Emergent_Gravity)
  (PDFs stay there; they are not committed to OpenWave).
- This proposed column: `openwave/xperiments/m9_emergent_gravity/`.
- First PR: [openwave-labs/openwave#441](https://github.com/openwave-labs/openwave/pull/441).
- M9.1 solver: `research/scripts/hehl_datta.py` +
  `m9_1_hehl_datta_elimination.py` (gate PASS \(r=3/16\); paper spin dual C2
  FAIL).
- \(I_B\) / EC campaign scripts (documented negatives, not official gravity
  cells): `m9_2_ib_hadamard.py`, `m9_3_ib_analytic.py`,
  `m9_4_ib_hadamard_complete.py`, `m9_5_ec_symplectic.py`.
- Newton-limit task run: C1 PASS, C2 FAIL. `scripts/m9_2_newton_limit.py`,
  [`findings/m9_2_newton_note.md`](findings/m9_2_newton_note.md).
- Metric-phenomena domain note (written; does not move `MODELS.md`):
  [`findings/m9_metric_phenomena_note.md`](findings/m9_metric_phenomena_note.md).
- Axial UV deformation (Q4b only): `scripts/m9_4_uv_axial.py`,
  [`findings/m9_4_uv_deformation_note.md`](findings/m9_4_uv_deformation_note.md).
- Q4a pair selection (uniqueness negative, existence open):
  `scripts/m9_5_q4a_pair.py`,
  [`findings/m9_5_q4a_pair_note.md`](findings/m9_5_q4a_pair_note.md).
- de Sitter FGHMV obstruction: `scripts/m9_6_ds_sign.py`,
  [`findings/m9_6_ds_closure_note.md`](findings/m9_6_ds_closure_note.md).
- A2 diamond waist (\(3+1\)D, two spacings): `scripts/m9_13_A2_diamond_4d.py`,
  [`findings/m9_13_A2_diamond_note.md`](findings/m9_13_A2_diamond_note.md).
- A1 diamond area law: `scripts/m9_14_A1_diamond_4d.py`,
  [`findings/m9_14_A1_diamond_note.md`](findings/m9_14_A1_diamond_note.md).
- CHM shape of the modular hop: `scripts/m9_15_chm_shape.py`,
  [`findings/m9_15_chm_shape_note.md`](findings/m9_15_chm_shape_note.md).
- First law, local not CHM-selected: `scripts/m9_16_first_law.py`,
  [`findings/m9_16_first_law_note.md`](findings/m9_16_first_law_note.md).
- Horizon vs bulk first law: `scripts/m9_18_surface_first_law.py`,
  [`findings/m9_17_18_horizon_first_law_note.md`](findings/m9_17_18_horizon_first_law_note.md).
- \(S^3\) curvature axis: `scripts/m9_19_s3_curvature_axis.py`,
  [`findings/m9_19_s3_curvature_note.md`](findings/m9_19_s3_curvature_note.md).
- Horizon shape (CHM vs linear): `scripts/m9_20_horizon_shape.py`,
  [`findings/m9_20_horizon_shape_note.md`](findings/m9_20_horizon_shape_note.md).
- Larger-R instrument halt: `scripts/m9_21_larger_horizon.py`,
  [`findings/m9_21_larger_horizon_note.md`](findings/m9_21_larger_horizon_note.md).
- C4 scored half-fill: `scripts/m9_22_halffill_horizon.py`,
  [`findings/m9_22_halffill_horizon_note.md`](findings/m9_22_halffill_horizon_note.md).
- Bloch vs CHM covering: `scripts/m9_23_bloch.py`,
  [`findings/m9_23_bloch_note.md`](findings/m9_23_bloch_note.md).
- Region shape at fixed \(H\): `scripts/m9_24_region_deform.py`,
  [`findings/m9_24_region_deform_note.md`](findings/m9_24_region_deform_note.md).
- Linear functional: `scripts/m9_25_linear_functional.py`,
  [`findings/m9_25_linear_functional_note.md`](findings/m9_25_linear_functional_note.md).
- Point source, flat wins: `scripts/m9_26_point_source.py`,
  [`findings/m9_26_point_source_note.md`](findings/m9_26_point_source_note.md).
- 1d CHM instrument calibration: `scripts/m9_27_1d_point_chm.py`,
  [`findings/m9_27_1d_point_chm_note.md`](findings/m9_27_1d_point_chm_note.md).
- Fixed-\(H\) first law: `scripts/m9_28_1d_state.py`,
  [`findings/m9_28_fixedh_state_note.md`](findings/m9_28_fixedh_state_note.md).
- Shape not sphere (guess, then probe measurement): `scripts/m9_29_shape.py`,
  [`findings/m9_29_shape_note.md`](findings/m9_29_shape_note.md).
- Area not Clausius: `scripts/m9_30_area.py`,
  [`findings/m9_30_area_note.md`](findings/m9_30_area_note.md).

## Which MODELS.md rows the model addresses

Native: Gravity: Newton limit; Gravity: metric phenomena. Possible later:
Lorentz covariance (action-level, not a boosted-defect measurement).

All particle, force-other-than-gravity, and wave-emergence rows: not derived.
They should stay 🚧. No `MODELS.md` cell is moved by this application or by
PR #441.

## Help wanted

- Maintainer scaffold and official ID; application is [discussion #442](https://github.com/openwave-labs/openwave/discussions/442).
- Review of PR #441 as the author's proposed first package (DCO signed;
  headless; no `MODELS.md`).
- Independent recompute of Papers IV-VII (Condition NL, \(I_B\),
  pure-information HD magnitude).
- A lattice or grid Newton-limit script that executes the locked M9.2 gates.
- A hostile parameter-count pass on the holographic half.

## What the first PR is

Not a `MODELS.md` edit. The first in-platform artifact is M9.1: extract the
HD coefficient from Palatini + Hermitian Dirac by stationarity, compare to
\(3/16\) only after extraction, mutate the Palatini factor to prove the check
can fail, and record an adversarial second-method audit. Holography is out of
scope for that task. The Newton cell is pre-registered only. The metric note
is a domain statement, not a cell.
