# Conservation & Scattering Test Results

- Config: `config_naph_2state.txt`
- States (band order): ('S2', 'S1')
- N_Q: 256, matrix size: 512x512
- Mean energies: E(S2) = 4.1207 eV, E(S1) = 4.1000 eV
- Energy ordering (high->low): S2 > S1
- Radiative decay gamma: {'S2': 0.005, 'S1': 0.0} (applied at Q index 0)
- Rate capacities sum W: S1->S1: 54.709, S1->S2: 26.862, S2->S1: 32.363, S2->S2: 62.442

| Test | Criterion | Value | Tolerance | Result |
|---|---|---|---|---|
| 1. Column-sum conservation | max\|sum_i K_ij\| / max\|K\| | 1.431e-15 | 1e-12 | PASS |
| 2. Dynamic conservation | max\|dN_tot/dt\| / N | 1.930e-17 | 1e-12 | PASS |
| 3. State-to-state flux balance | \|sum_a dN_a/dt\| / N (pure-state starts) | 0.000e+00 | 1e-12 | PASS |
| 4. Emission > absorption capacity | E(S2)>E(S1) => sum W(S2->S1) > sum W(S1->S2) > 0 | 1.205e+00 | 1e+00 | PASS |
| 5. Open-system decay law | \|dN/dt + sum_a gamma_a N_a(Q=gamma)\| / N | 5.603e-18 | 1e-12 | PASS |

**Overall: ALL PASS**
