# figure2_qubit_qubit.py
import numpy as np
import matplotlib.pyplot as plt
from qutip import (basis, tensor, qeye, sigmax, sigmay, sigmaz, sigmap, sigmam,
                   mesolve, expect, ptrace)
import krotov

# -------------------------
# Parameters (from Fig.2)
# -------------------------
omega = 1.0                 # set ω = 1 (units)
g = 0.2 * omega             # coupling between charger and battery
gamma = 0.05 * omega        # dissipation rate on the charger (local)
mu = 0.5 * omega            # coupling to the classical field (control)
F_amp = 0.5 * omega         # amplitude used for the sinusoidal drive (paper uses F = µ = 0.5ω)
Nb_T = 0.0                  # temperature (avg bath occupancy)
kappa = 0.5                 # initial guess amplitude (figure 2 used κ = 0.5)

# choose final time as in the paper conventions (often pi/g for plots)
tau = np.pi / g
N_t = 1001
tlist = np.linspace(0.0, tau, N_t)

# -------------------------
# Operators & Hamiltonian
# -------------------------
# single-qubit operators
sx = sigmax()
sz = sigmaz()
sp = sigmap()
sm = sigmam()
I = qeye(2)

# system: A = charger (qubit 0), B = battery (qubit 1)
sx_A = tensor(sx, I)
sz_A = tensor(sz, I)
sp_A = tensor(sp, I)
sm_A = tensor(sm, I)

sx_B = tensor(I, sx)
sz_B = tensor(I, sz)
sp_B = tensor(I, sp)
sm_B = tensor(I, sm)

# Free Hamiltonians (paper uses HA = ω/2 (-σz + I) etc.; we use simple σz forms with ω=1)
H_A = 0.5 * omega * sz_A
H_B = 0.5 * omega * sz_B

# Interaction HAB = g (σ+_A σ-_B + h.c.)
H_int = g * (sp_A * sm_B + sm_A * sp_B)

# Control Hamiltonian (couples to charger A): - mu * eps(t) * σx_A
H_control = - mu * sx_A  # time-dependent prefactor eps(t) will be provided separately

# Total drift Hamiltonian (time-independent part)
H0 = H_A + H_B + H_int

# -------------------------
# Dissipation (local on charger A)
# -------------------------
# Lindblad collapse operators for the charger (A) coupling to a zero-T bath
c_ops = []
if Nb_T == 0:
    c_ops.append(np.sqrt(gamma * (Nb_T + 1.0)) * sm_A)
else:
    # finite temperature (not used in Fig.2), include both up/down jumps
    c_ops.append(np.sqrt(gamma * (Nb_T + 1.0)) * sm_A)
    c_ops.append(np.sqrt(gamma * Nb_T) * sp_A)

# -------------------------
# initial and target states
# -------------------------
psi0 = tensor(basis(2, 0), basis(2, 0))   # |0>_A ⊗ |0>_B  (both ground)
# target: IA ⊗ |1><1| (i.e., we want battery excited; using a pure projection target on full system)
# simplest: target state = any charger state ⊗ battery excited. We follow the paper: use IA ⊗ |1>
# here choose charger left unprescribed by picking |0>_A -> but optimization can still drive battery population
psitgt = tensor(basis(2, 0), basis(2, 1))  # IA ⊗ |1>_B (one choice)

# -------------------------
# helper: sin^2 ramp S(t)
# -------------------------
def S_of_t(t, t_final=tau, ton_frac=0.005, toff_frac=0.005):
    ton = ton_frac * t_final
    toff = toff_frac * t_final
    if t < ton:
        return np.sin(np.pi * t / (2 * ton)) ** 2
    if t > (t_final - toff):
        return np.sin(np.pi * (t_final - t) / (2 * toff)) ** 2
    return 1.0

S_array = np.array([S_of_t(t) for t in tlist])

# -------------------------
# initial guess pulse (S(t) * kappa)
# -------------------------
def eps_guess(t, args=None):
    return float(S_of_t(t) * kappa)

# sinusoidal non-optimized pulse (for comparison)
def eps_sin(t, args=None):
    return float(S_of_t(t) * F_amp * np.cos(omega * t))

# -------------------------
# Create krotov objectives
# -------------------------
# Time-dependent Hamiltonian format for krotov: [H0, [H_control, pulse]]
H_td = [H0, [H_control, eps_guess]]

# build objective using density matrices (krotov supports state vectors too)
obj = krotov.Objective(initial_state=psi0, target=psitgt, H=H_td)

# -------------------------
# Krotov pulse optimization (very small/fast iteration for example)
# -------------------------
# choose the shape function S(t) for updates (as array interpolated by krotov)
def S_update(t):
    return S_of_t(t)

pulse_options = {
    eps_guess: dict(lambda_a=1.0, update_shape=S_update)
}

# run a few iterations (for a minimal example). For production, increase iter_count.
opt_result = krotov.optimize_pulses(
    objectives=[obj],
    pulse_options=pulse_options,
    tlist=tlist,
    propagator=krotov.propagators.expm,
    chi_constructor=krotov.functionals.chis_re,
    iter_stop=10,
)

# Retrieve optimized pulse as a function on tlist
opt_pulses = opt_result.optimized_controls
eps_opt_array = np.array(opt_pulses[0])

# If krotov didn't run (or for fallback), ensure eps_opt_array exists:
if eps_opt_array.size != tlist.size:
    eps_opt_array = np.array([eps_guess(t) for t in tlist])

# -------------------------
# Evolve with both pulses and compute energy & ergotropy of battery vs time
# -------------------------
# Time-dependent Hamiltonian for mesolve: [H0, [H_control, eps_func]]
H_with_opt = [H0, [H_control, lambda t, args: float(np.interp(t, tlist, eps_opt_array))]]
H_with_sin = [H0, [H_control, eps_sin]]

# evolve density matrices
out_opt = mesolve(H_with_opt, psi0, tlist, c_ops, [])
out_sin = mesolve(H_with_sin, psi0, tlist, c_ops, [])

# reduce to battery and compute energy and ergotropy
# battery Hamiltonian (single qubit) in the paper's convention; using 0.5*omega*sz_B
H_battery_local = 0.5 * omega * (-sigmaz() + qeye(2))

def ergotropy_rho(rho_sys, H_b_local):
    # trace out charger → battery density matrix (2x2)
    rho_b = ptrace(rho_sys, 1)
    rho_mat = rho_b.full()
    H_mat = H_b_local.full()

    # eigenvalues & eigenvectors
    er, vr = np.linalg.eigh(rho_mat)
    eH, vH = np.linalg.eigh(H_mat)

    # sort rho eigenvalues descending
    idx_r = np.argsort(er)[::-1]

    # sort Hamiltonian eigenvalues ascending (ground first)
    idx_H = np.argsort(eH)

    # construct passive state
    rho_passive = np.zeros_like(rho_mat, dtype=complex)
    for i, p_idx in enumerate(idx_r):
        vec = vH[:, idx_H[i]]
        rho_passive += er[p_idx] * np.outer(vec, vec.conj())

    # energies
    E = np.real(np.trace(rho_mat @ H_mat))
    E_pass = np.real(np.trace(rho_passive @ H_mat))

    return E, max(0.0, E - E_pass)

# compute arrays
E_opt = np.zeros_like(tlist)
erg_opt = np.zeros_like(tlist)
E_sin = np.zeros_like(tlist)
erg_sin = np.zeros_like(tlist)

for i, t in enumerate(tlist):
    rho_opt = out_opt.states[i]
    rho_sin = out_sin.states[i]
    E_opt[i], erg_opt[i] = ergotropy_rho(rho_opt, H_battery_local)
    E_sin[i], erg_sin[i] = ergotropy_rho(rho_sin, H_battery_local)

# -------------------------
# Plot & save figures
# -------------------------
plt.figure(figsize=(6, 4))
plt.plot(tlist * g, E_opt / omega, label='Energy (opt)', color='green')
plt.plot(tlist * g, erg_opt / omega, label='Ergotropy (opt)', linestyle='-.', color='green', alpha=0.8)
plt.plot(tlist * g, E_sin / omega, label='Energy (sin)', color='black')
plt.plot(tlist * g, erg_sin / omega, label='Ergotropy (sin)', linestyle='-.', color='black', alpha=0.8)
plt.xlabel(r'$g t$')
plt.ylabel('E, Ergotropy (units ω)')
plt.legend()
plt.tight_layout()
plt.savefig('./graphs/energy_ergotropy.png', dpi=200)
print("Saved energy/ergotropy figure: energy_ergotropy.png")

plt.figure(figsize=(6, 3))
plt.plot(tlist * g, eps_opt_array, label='optimized pulse', color='green')
plt.plot(tlist * g, [eps_sin(t) for t in tlist], label='sinusoidal pulse', color='black', linestyle='--')
plt.xlabel(r'$g t$')
plt.ylabel(r'$\epsilon(t)$')
plt.legend()
plt.tight_layout()
plt.savefig('./graphs/pulses.png', dpi=200)
print("Saved pulses figure: pulses.png")
