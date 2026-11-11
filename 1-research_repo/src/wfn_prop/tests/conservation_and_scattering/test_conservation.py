"""
Chunk 2: Collision-matrix and conservation verification for the S2 <-> S1 back-scattering model.

Run from anywhere:
    PYTHONPATH=<repo>/src python tests/conservation_and_scattering/test_conservation.py [config]

Outputs (next to this file): test_results.md, report_conservation.png.
Exit code is nonzero if any test fails. Fully config-driven (N states supported).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
WFN = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, WFN)
os.chdir(WFN)  # config paths are relative to wfn_prop/

from wfn_prop.io import parse_input_file
from wfn_prop.k_scat import MultiBandTransitionMatrix
import importlib.util
_spec = importlib.util.spec_from_file_location("wfn_main", os.path.join(WFN, "main.py"))
wfn_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wfn_main)

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "config_naph_2state.txt"
TOL_MATRIX = 1e-12   # relative to max |K| (float64 summation of O(1e1) entries)
TOL_FLUX = 1e-12


# ----------------------------------------------------------------- setup
cfg = parse_input_file(CONFIG)
phys = np.load(cfg["physics_file"])
Qpts = phys["Qpts"]
N_Q = len(Qpts)
states = cfg["state_names"]
gammas = tuple(float(g) for g in cfg["state_gamma_decay"])
nb = len(states)
gidx = int(phys["gamma_index"])

rates = {}
for o, so in enumerate(states):
    for i, si in enumerate(states):
        k = f"Rate_{si}_{so}"
        if k in phys:
            rates[(o, i)] = phys[k]


def build(gam):
    return MultiBandTransitionMatrix(n_bands=nb, rates_dict=rates, gamma_decay_rates=gam,
                                     map_Q_to_q=phys["Q_plus_q_map"], gamma_index=gidx)


# per-state mean energies (autodetect via main's search paths)
def state_energy(state):
    stem = os.path.basename(cfg["vel_file"][:-4])
    for c in (f"{stem}_{state}.npz", f"processing/xct_vel_extraction/data/{stem}_{state}.npz"):
        if os.path.exists(c):
            return np.load(c)["energy"]
    raise FileNotFoundError(f"No energy payload for {state}")


energies = {s: state_energy(s) for s in states}
Ebar = {s: float(energies[s].mean()) for s in states}

op0 = build(tuple(0.0 for _ in states))   # lossless operator
opg = build(gammas)                       # with radiative decay
K0 = op0.K_tensor.transpose(0, 2, 1, 3).reshape(nb * N_Q, nb * N_Q)  # [(a,i),(b,j)]
Kg = opg.K_tensor.transpose(0, 2, 1, 3).reshape(nb * N_Q, nb * N_Q)

results = []  # (name, criterion, value, tol, passed)


def record(name, criterion, value, tol, passed):
    results.append((name, criterion, value, tol, bool(passed)))


rng = np.random.default_rng(0)

# ---- Test 1: column sums of lossless matrix vanish
colsum = K0.sum(axis=0)
scale = np.abs(K0).max()
t1 = np.abs(colsum).max() / scale
record("1. Column-sum conservation", "max|sum_i K_ij| / max|K|", t1, TOL_MATRIX, t1 < TOL_MATRIX)

# ---- Test 2: dN_tot/dt = 0 for random and Gaussian-like occupations (lossless)
def total_rate(op, n4):
    d = op.calc_scattering(n4)
    return d.sum(), d.sum(axis=(1, 2, 3))

worst = 0.0
for trial in range(5):
    n4 = rng.random((nb, 3, 3, N_Q))
    dtot, _ = total_rate(op0, n4)
    worst = max(worst, abs(dtot) / n4.sum())
g = np.exp(-0.5 * (np.linalg.norm(Qpts - Qpts.mean(0), axis=1) / 0.3) ** 2)
n4 = np.zeros((nb, 3, 3, N_Q)); n4[0] = g
dtot, _ = total_rate(op0, n4)
worst = max(worst, abs(dtot) / n4.sum())
record("2. Dynamic conservation", "max|dN_tot/dt| / N", worst, TOL_FLUX, worst < TOL_FLUX)

# ---- Test 3: state-to-state flux balance (2-state specific; pairwise for N)
flux_rows = []
ok3 = True
worst3 = 0.0
for b in range(nb):
    n4 = np.zeros((nb, 3, 3, N_Q)); n4[b] = rng.random((3, 3, N_Q))
    _, per_state = total_rate(op0, n4)
    N = n4.sum()
    flux_rows.append((states[b], per_state / N))
    resid = abs(per_state.sum()) / N
    worst3 = max(worst3, resid)
record("3. State-to-state flux balance", "|sum_a dN_a/dt| / N (pure-state starts)", worst3,
       TOL_FLUX, worst3 < TOL_FLUX)

# ---- Test 4: energetic hierarchy vs. rate asymmetry (absorption vs emission)
order = sorted(states, key=lambda s: -Ebar[s])  # highest first
hi, lo = order[0], order[-1]
cap = {}
for (o, i), R in rates.items():
    cap[(states[i], states[o])] = float(R.sum())
down, up = cap.get((hi, lo), 0.0), cap.get((lo, hi), 0.0)
t4 = down > up > 0.0
record("4. Emission > absorption capacity",
       f"E({hi})>E({lo}) => sum W({hi}->{lo}) > sum W({lo}->{hi}) > 0",
       down / up if up else float("inf"), 1.0, t4)

# ---- Test 5: open-system decay: dN_tot/dt = -sum_a gamma_a N_a(gamma_index)
worst5 = 0.0
for trial in range(5):
    n4 = rng.random((nb, 3, 3, N_Q))
    dtot, _ = total_rate(opg, n4)
    expected = -sum(gammas[a] * n4[a, :, :, gidx].sum() for a in range(nb))
    worst5 = max(worst5, abs(dtot - expected) / n4.sum())
record("5. Open-system decay law", "|dN/dt + sum_a gamma_a N_a(Q=gamma)| / N", worst5,
       TOL_FLUX, worst5 < TOL_FLUX)

# ----------------------------------------------------------------- report
all_pass = all(r[-1] for r in results)
with open(os.path.join(HERE, "test_results.md"), "w") as f:
    f.write(f"# Conservation & Scattering Test Results\n\n")
    f.write(f"- Config: `{CONFIG}`\n- States (band order): {states}\n- N_Q: {N_Q}, matrix size: {nb*N_Q}x{nb*N_Q}\n")
    f.write("- Mean energies: " + ", ".join(f"E({s}) = {Ebar[s]:.4f} eV" for s in states) + "\n")
    f.write(f"- Energy ordering (high->low): {' > '.join(order)}\n")
    f.write(f"- Radiative decay gamma: {dict(zip(states, gammas))} (applied at Q index {gidx})\n")
    f.write(f"- Rate capacities sum W: " + ", ".join(f"{a}->{b}: {v:.3f}" for (a, b), v in sorted(cap.items())) + "\n\n")
    f.write("| Test | Criterion | Value | Tolerance | Result |\n|---|---|---|---|---|\n")
    for n, c, v, t, p in results:
        f.write(f"| {n} | {c.replace(chr(124), chr(92)+chr(124))} | {v:.3e} | {t:.0e} | {'PASS' if p else 'FAIL'} |\n")
    f.write(f"\n**Overall: {'ALL PASS' if all_pass else 'FAILURES PRESENT'}**\n")

fig, ax = plt.subplots(2, 2, figsize=(12, 9))
# (a) column sums
a = ax[0, 0]
a.semilogy(np.abs(colsum) + 1e-20, ".", ms=3, label="lossless K")
a.semilogy(np.abs(Kg.sum(axis=0)) + 1e-20, ".", ms=3, label="with radiative decay")
a.axhline(TOL_MATRIX * scale, color="k", ls="--", lw=1, label="tolerance")
a.set(xlabel="column index (state, Q)", ylabel=r"$|\sum_i K_{ij}|$", title="(a) Column sums (conservation)")
a.legend(fontsize=8)
# (b) flux balance
a = ax[0, 1]
w = 0.8 / nb
for b, (sname, v) in enumerate(flux_rows):
    a.bar(np.arange(nb) + b * w, v, w, label=f"start in {sname}")
a.set_xticks(np.arange(nb) + 0.4 - w / 2); a.set_xticklabels([f"dN({s})/dt/N" for s in states])
a.axhline(0, color="k", lw=0.8); a.legend(fontsize=8)
a.set(title="(b) Instantaneous inter-state flux (sums to zero)")
# (c) energies
a = ax[1, 0]
for s in states:
    a.hist(energies[s], bins=30, alpha=0.5, label=f"{s}  mean {Ebar[s]:.4f} eV")
a.set(xlabel="E (eV)", ylabel="count (Q-points)",
      title=f"(c) Band energies; down: {hi}->{lo} = {down:.1f}, up: {lo}->{hi} = {up:.1f}")
a.legend(fontsize=8)
# (d) decay law
a = ax[1, 1]
n4 = rng.random((nb, 3, 3, N_Q))
lhs = total_rate(opg, n4)[0]
rhs = -sum(gammas[k] * n4[k, :, :, gidx].sum() for k in range(nb))
a.bar(["numerical dN/dt", "-sum gamma N"], [lhs, rhs], color=["C0", "C1"])
a.set(title="(d) Open-system decay law (random state)")
fig.suptitle(f"Conservation report - {'ALL PASS' if all_pass else 'FAIL'}", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "report_conservation.png"), dpi=150)

print(open(os.path.join(HERE, "test_results.md")).read())
sys.exit(0 if all_pass else 1)
