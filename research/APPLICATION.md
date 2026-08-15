# Draft: New Model discussion body

> Paste into
> [github.com/openwave-labs/openwave/discussions/categories/new-model](https://github.com/openwave-labs/openwave/discussions/categories/new-model).
> This is the author's application, not a maintainer admission.

## Model name

Emergent Gravity / New Standard Model (NSM). Proposed short column name: M9
(assignment is the maintainer's).

## Author

Dr. Robert W. McGwier, PhD. CTO, Cohere Technology Group. Sole author of the
14 August 2026 series.

## Author contact

GitHub: [@n4hy](https://github.com/n4hy)

ORCID: **\<if any\>**

## Lineage

Entanglement first-law gravity (Faulkner-Guica-Hartman-Myers-Van Raamsdonk,
JHEP 03 (2014) 051); Casini-Huerta-Myers modular Hamiltonians; Einstein-Cartan
(Kibble, Sciama, Hehl); Hehl-Datta 1971 four-fermion contact; Jacobson
entanglement equilibrium (nonlinear, tagged conjectural).

## Bedrock papers

Public record: the 14 August 2026 distribution (CC-BY-4.0). Author-stated
Zenodo DOIs are listed in
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

- Multi-digit coefficient of \(I_B\): open (order-unity claimed; analytic
  Hadamard expansion still required).
- de Sitter / cosmology: open (Paper IX).
- Nonlinear Einstein-Cartan from entanglement: open.
- UV completion: open by construction.
- No dark-matter particle; no derivation of masses, mixings, generations,
  strong CP, or \(\Lambda\).
- The HD contact is a 1971 theorem of Einstein-Cartan + Dirac. The distinctive
  claim is the *selection* of Einstein-Cartan, which is not the first
  in-platform task.

## Falsifiers

| Signature | Prediction | Current bound | Refutation |
| --- | --- | --- | --- |
| HD four-fermion | strength \(\sim G\), cross-species | collider contact bounds far above | a measured four-fermion coefficient inconsistent with \(3\kappa/16\) (impractical near term) |
| Quantum geometry | gravity carries coherence | Bose / Marletto-Vedral not yet decisive | confirmed null at sufficient sensitivity |
| High-density bounce | torsion bounce replaces the initial singularity | none | `[C]`; a derivation of nonlinear EC that forbids the bounce |
| IR / Milgrom | only if a dS extension exists | SPARC / galactic dynamics | `[C]` and domain-open; not to be scored as a prediction yet |

## Formal artifacts

- Author PDF series (local distribution zip; do not commit the PDFs).
- This draft column: `openwave/xperiments/m9_emergent_gravity/`.
- M9.1 solver: `research/scripts/hehl_datta.py` +
  `m9_1_hehl_datta_elimination.py`.
- No \(I_B\) numerical campaign code was included in the distribution zip.

## Which MODELS.md rows the model addresses

Native: Gravity: Newton limit; Gravity: metric phenomena. Possible later:
Lorentz covariance (action-level, not a boosted-defect measurement).

All particle, force-other-than-gravity, and wave-emergence rows: not derived.
They should stay 🚧.

## Help wanted

- Maintainer scaffold and official ID if the discussion is accepted.
- Independent recompute of Papers IV-VII (Condition NL, \(I_B\),
  pure-information HD magnitude).
- A lattice Newton-limit script for the first gravity cell.
- A hostile parameter-count pass on the holographic half.

## What the first PR is

Not a `MODELS.md` edit. The first in-platform artifact is M9.1: extract the
HD coefficient from Palatini + Hermitian Dirac by stationarity, compare to
\(3/16\) only after extraction, mutate the Palatini factor to prove the check
can fail, and record an adversarial second-method audit. Holography is out of
scope for that task.
