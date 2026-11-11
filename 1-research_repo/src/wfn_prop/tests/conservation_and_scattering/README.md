# Conservation & Scattering Verification

Purpose: demonstrate that the multi-band collision operator built by `MultiBandTransitionMatrix`
conserves exciton number, couples states with the correct orientation, and respects the
thermal-population asymmetry between emission and absorption.

> [!NOTE]
> For a detailed, intuitive line-by-line breakdown of every test result and plot panel, see **[EXPLANATION.md](EXPLANATION.md)**.

## Run
```
PYTHONPATH=<repo>/src python tests/conservation_and_scattering/test_conservation.py [config]
```
Default config: `config_naph_2state.txt`. Writes `test_results.md` (table) and
`report_conservation.png` (4-panel figure); exit code is nonzero on failure.

## Physics context
Boltzmann collision term for band `a`, momentum `Q`:

dn_a(Q)/dt = sum_{b,Q'} K_{(a,Q),(b,Q')} n_b(Q')

- Gain: `K[(a,Q+q),(b,Q)] += W_{b->a}(Q,q)`.
- Loss (diagonal): `K[(b,Q),(b,Q)] -= sum_{a,q} W_{b->a}(Q,q)`.
- Without radiative decay each column sums to zero, so total exciton number is conserved exactly;
  any transfer satisfies `dN_a/dt = -sum_{b!=a} dN_b/dt`.
- Radiative decay `gamma_a` is an extra diagonal sink at the `gamma_index` Q-point only.

### Energy ordering matters (not state index)
Rates are phonon-mediated. Which direction is "downhill" is set by the band energies E_a(Q),
**not** by the state label:
- Downhill (E_final < E_initial): phonon **emission**, ∝ (1 + n_qν). Finite at T→0.
- Uphill (E_final > E_initial): phonon **absorption**, ∝ n_qν = 1/(e^{ħω/kT} − 1). Vanishes as T→0.

In naphthalene here, mean E(S2) ≈ 4.121 eV > E(S1) ≈ 4.100 eV (Δ ≈ 20.7 meV), so S2→S1 is
emission and S1→S2 is thermally activated. The "thermal sink" interpretation of S1 is specific to
this ordering; Test 4 reads energies from the payloads and flips its expectation automatically if
the ordering differs.

## Tests
| # | Check | Pass criterion |
|---|---|---|
| 1 | Column sums of lossless K | max\|Σ_i K_ij\| / max\|K\| < 1e-12 |
| 2 | dN_tot/dt on random + Gaussian-like states | < 1e-12 · N |
| 3 | Inter-state flux balance (pure-state starts) | Σ_a dN_a/dt ≈ 0 |
| 4 | Higher-energy→lower-energy rate capacity > reverse > 0 | ratio > 1 |
| 5 | Open-system law dN/dt = −Σ_a γ_a N_a(γ-point) | residual < 1e-12 · N |

Test 4 compares rate capacities (sum of W), a necessary signature of detailed balance, not a full
detailed-balance check.
