import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm as scipy_expm
from qutip import Qobj, ket2dm, mesolve, Options
import os

os.makedirs('./graphs', exist_ok=True)

# ============================================================
# Dissipative Lambda System — Non-Hermitian Hamiltonian
#
# Same 3-level Lambda system as example 02, but with an
# imaginary decay term on level |2>:
#
#   H0[1,1] = -i*gamma   (instead of 0)
#
# This non-Hermitian term causes the state norm to decay
# whenever population passes through |2>. The optimizer
# is therefore penalized for routing population via |2>.
# This is the STIRAP physics — transfer |1> -> |3> without
# populating the lossy intermediate level.
#
# Functional: J_T_re = 1 - Re{<psi(T)|psi_tgt>}  (chis_re)
# chi(T) = psi_tgt  (no tau scaling)
#
# Reference:
#   https://qucontrol.github.io/krotov/v1.3.0/notebooks/
#   03_example_lambda_system_rwa_non_hermitian.html
# ============================================================

# -------------------------
# Parameters (exactly from ARGS dict in the example)
# -------------------------
Omega0  = 5.0    # amplitude of both pump and Stokes pulses
DeltaTP = 3.0    # duration of pump pulse
DeltaTS = 3.0    # duration of Stokes pulse
t0P     = 2.0    # start time of pump pulse
t0S     = 0.0    # start time of Stokes pulse
t_rise  = 0.3    # switch-on/off time for update shape
E1      = 0.0
E2      = 10.0
E3      = 5.0
omega_P = 9.5
omega_S = 4.5
gamma   = 0.5    # decay rate on |2> — the key new parameter
T       = 5.0
N_t     = 500
lambda_a = 2.0   # larger than example 2 (was 0.5) to handle non-Hermitian
N_iter  = 40     # example runs 40, then continues to 2000

tlist = np.linspace(0.0, T, N_t)
dt    = tlist[1] - tlist[0]

# -------------------------
# Detunings
# -------------------------
Delta_P = E1 + omega_P - E2   # = -0.5
Delta_S = E3 + omega_S - E2   # = -0.5

# -------------------------
# Hamiltonians (dense 3x3 numpy matrices)
#
# Only change from example 02: H0[1,1] = -i*gamma instead of 0
# This adds the non-Hermitian decay on level |2>
# -------------------------
H0_mat = np.array([
    [Delta_P,      0.0,      0.0    ],
    [0.0,     -1j*gamma,     0.0    ],   # <-- non-Hermitian decay
    [0.0,          0.0,      Delta_S]
], dtype=complex)

HP_re_mat = -0.5 * np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
], dtype=complex)

HP_im_mat = -0.5 * np.array([
    [0.0,  1.0j, 0.0],
    [-1.0j, 0.0, 0.0],
    [0.0,  0.0,  0.0]
], dtype=complex)

HS_re_mat = -0.5 * np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0]
], dtype=complex)

HS_im_mat = -0.5 * np.array([
    [0.0, 0.0,    0.0],
    [0.0, 0.0,   1.0j],
    [0.0, -1.0j, 0.0 ]
], dtype=complex)

H_ctrl_mats = [HP_re_mat, HP_im_mat, HS_re_mat, HS_im_mat]

# -------------------------
# States
# -------------------------
ket1 = np.array([1.0, 0.0, 0.0], dtype=complex)
ket2 = np.array([0.0, 1.0, 0.0], dtype=complex)
ket3 = np.array([0.0, 0.0, 1.0], dtype=complex)

psi_init = ket1.copy()

# Phase-sensitive target in rotating frame
psi_tgt = np.exp(1j * (E2 - omega_S) * T) * ket3

# -------------------------
# Shape function S(t) — sinsq flattop, t_rise = 0.3
# -------------------------
def S_of_t(t):
    if t <= 0.0 or t >= T:
        return 0.0
    if t < t_rise:
        return float(np.sin(np.pi * t / (2.0 * t_rise)) ** 2)
    if t > T - t_rise:
        return float(np.sin(np.pi * (T - t) / (2.0 * t_rise)) ** 2)
    return 1.0

S_array = np.array([S_of_t(t) for t in tlist])

# -------------------------
# Blackman shape for guess pulses
# -------------------------
def blackman(t, t_start, t_stop):
    if t <= t_start or t >= t_stop:
        return 0.0
    t_w  = t_stop - t_start
    x    = (t - t_start) / t_w
    val  = 0.42 - 0.5 * np.cos(2 * np.pi * x) + 0.08 * np.cos(4 * np.pi * x)
    peak = 0.42 - 0.5 * np.cos(np.pi) + 0.08 * np.cos(2 * np.pi)   # = 1.0
    return float(val / peak)

# Guess controls (same as example 02)
eps_P1_guess = np.array([Omega0 * blackman(t, t0P, t0P + DeltaTP) for t in tlist])
eps_P2_guess = np.zeros(N_t)
eps_S1_guess = np.array([Omega0 * blackman(t, t0S, t0S + DeltaTS) for t in tlist])
eps_S2_guess = np.zeros(N_t)

eps_arrays = np.stack([eps_P1_guess, eps_P2_guess,
                       eps_S1_guess, eps_S2_guess])

# ============================================================
# Manual Krotov for non-Hermitian system
#
# Forward propagator — same as before, non-Hermiticity
# is already encoded in H0_mat:
#   psi(t+dt) = expm(-i * H_total * dt) psi(t)
#
# Backward propagator — for non-Hermitian H, the adjoint
# equation uses H†, so:
#   chi(t-dt) = expm(+i * H_total† * dt) chi(t)
#
# This differs from the Hermitian case only when H ≠ H†,
# i.e. H_total.conj().T ≠ H_total. Since H0_mat has
# -i*gamma on diagonal, H0†[1,1] = +i*gamma.
#
# Functional: J_T_re = 1 - Re{<psi_tgt|psi(T)>}  (chis_re)
# chi(T) = psi_tgt  (no tau scaling)
#
# Update rule (same as before):
#   delta_eps_l(t) = (S(t)/lambda_a) * Im{ <chi(t)| H_ctrl_l |psi(t)> }
# ============================================================

def build_H_total(eps_vals):
    H = H0_mat.copy()
    for l, e in enumerate(eps_vals):
        H += e * H_ctrl_mats[l]
    return H

def backward_propagate(eps_arrs):
    """
    Backward propagation — chis_re boundary condition.
    chi(T) = psi_tgt (no tau scaling needed for chis_re).

    For non-Hermitian H, backward uses H†:
        chi(t-dt) = expm(+i * H_total† * dt) chi(t)
    This naturally handles the non-Hermitian case.
    """
    states     = np.zeros((N_t, 3), dtype=complex)
    states[-1] = psi_tgt.copy()
    for k in range(N_t - 1, 0, -1):
        eps_mid = 0.5 * (eps_arrs[:, k] + eps_arrs[:, k - 1])
        H_tot   = build_H_total(eps_mid)
        prop    = scipy_expm(1j * H_tot.conj().T * dt)   # H† for non-Hermitian
        states[k - 1] = prop @ states[k]
    return states


print("Running Krotov optimisation (Dissipative Lambda system) ...")
print(f"  gamma = {gamma},  lambda_a = {lambda_a},  N_iter = {N_iter}\n")

F_prev           = -999.0
no_improve_count = 0
tol              = 1e-8
patience         = 5

for iteration in range(N_iter):

    bwd     = backward_propagate(eps_arrays)
    new_eps = eps_arrays.copy()
    psi     = psi_init.copy()

    for k in range(N_t - 1):
        for l in range(4):
            Hl_psi        = H_ctrl_mats[l] @ psi
            delta         = (S_array[k] / lambda_a) * np.imag(bwd[k].conj() @ Hl_psi)
            new_eps[l, k] = eps_arrays[l, k] + delta

        eps_mid = 0.5 * (new_eps[:, k] + eps_arrays[:, k + 1])
        H_tot   = build_H_total(eps_mid)
        psi     = scipy_expm(-1j * H_tot * dt) @ psi

    # Last point
    for l in range(4):
        Hl_psi         = H_ctrl_mats[l] @ psi
        delta          = (S_array[-1] / lambda_a) * np.imag(bwd[-1].conj() @ Hl_psi)
        new_eps[l, -1] = eps_arrays[l, -1] + delta

    new_eps = new_eps * S_array

    # J_T_re = 1 - Re{<psi_tgt|psi(T)>}
    F   = np.real(psi_tgt.conj() @ psi)
    J_T = 1.0 - F

    eps_arrays = new_eps

    print(f"  iter {iteration + 1:3d}   F = {F:.6f}   J_T = {J_T:.4e}   "
          f"norm = {np.linalg.norm(psi):.4f}")

    if abs(F - F_prev) < tol:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"\n  Stopped: no improvement for {patience} iterations")
            break
    else:
        no_improve_count = 0
    F_prev = F

print("\nOptimisation finished.")
eps_opt = eps_arrays.copy()

# -------------------------
# Simulate with mesolve for plotting
# (mesolve handles non-Hermitian H correctly)
# -------------------------
opts = Options(nsteps=10000)

H0_qutip    = Qobj(H0_mat)
HP_re_qutip = Qobj(HP_re_mat)
HP_im_qutip = Qobj(HP_im_mat)
HS_re_qutip = Qobj(HS_re_mat)
HS_im_qutip = Qobj(HS_im_mat)

proj1 = ket2dm(Qobj(ket1))
proj2 = ket2dm(Qobj(ket2))
proj3 = ket2dm(Qobj(ket3))

def make_pulse(arr):
    def _f(t, args=None):
        return float(np.interp(t, tlist, arr))
    return _f

H_guess_q = [
    H0_qutip,
    [HP_re_qutip, make_pulse(eps_P1_guess)],
    [HP_im_qutip, make_pulse(eps_P2_guess)],
    [HS_re_qutip, make_pulse(eps_S1_guess)],
    [HS_im_qutip, make_pulse(eps_S2_guess)],
]
H_opt_q = [
    H0_qutip,
    [HP_re_qutip, make_pulse(eps_opt[0])],
    [HP_im_qutip, make_pulse(eps_opt[1])],
    [HS_re_qutip, make_pulse(eps_opt[2])],
    [HS_im_qutip, make_pulse(eps_opt[3])],
]

psi_init_qutip = Qobj(psi_init)

print("Propagating guess pulses ...")
out_guess = mesolve(H_guess_q, psi_init_qutip, tlist, [],
                    [proj1, proj2, proj3], options=opts)
out_guess_states = mesolve(H_guess_q, psi_init_qutip, tlist, [], [],
                           options=opts)

print("Propagating optimised pulses ...")
out_opt   = mesolve(H_opt_q, psi_init_qutip, tlist, [],
                    [proj1, proj2, proj3], options=opts)
out_opt_states = mesolve(H_opt_q, psi_init_qutip, tlist, [], [],
                         options=opts)

# State norms over time
norm_guess = np.array([s.norm() for s in out_guess_states.states])
norm_opt   = np.array([s.norm() for s in out_opt_states.states])

# -------------------------
# Figure 1: Population dynamics — guess vs optimised
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, out, title in zip(
    axes,
    [out_guess, out_opt],
    ['Population dynamics — guess', f'Population dynamics — optimised  (F={F:.4f})']
):
    ax.axhline(y=1.0, color='black', lw=0.5, ls='dashed')
    ax.axhline(y=0.0, color='black', lw=0.5, ls='dashed')
    ax.plot(tlist, out.expect[0], label='|1⟩', color='#1f77b4')
    ax.plot(tlist, out.expect[1], label='|2⟩', color='#ff7f0e')
    ax.plot(tlist, out.expect[2], label='|3⟩', color='#2ca02c')
    ax.set_title(title)
    ax.set_xlabel('time')
    ax.set_ylabel('population')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('./graphs/dissipative_population.png', dpi=200)
plt.close()
print("Saved: ./graphs/dissipative_population.png")

# -------------------------
# Figure 2: State norm vs time — shows effect of non-Hermitian decay
# -------------------------
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(tlist, norm_guess, label='guess',     color='black', linestyle='--')
ax.plot(tlist, norm_opt,   label='optimised', color='green', linestyle='-')
ax.axhline(y=1.0, color='grey', lw=0.5, ls=':')
ax.set_xlabel('time')
ax.set_ylabel('state norm ‖ψ(t)‖')
ax.set_title('State norm — effect of non-Hermitian decay on |2⟩')
ax.legend()
plt.tight_layout()
plt.savefig('./graphs/dissipative_norm.png', dpi=200)
plt.close()
print("Saved: ./graphs/dissipative_norm.png")

# -------------------------
# Figure 3: Optimised pulse amplitude and phase
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 6))

pulse_pairs = [(eps_opt[0], eps_opt[1], 'Pump'),
               (eps_opt[2], eps_opt[3], 'Stokes')]

for col, (re, im, name) in enumerate(pulse_pairs):
    amp   = np.sqrt(re**2 + im**2)
    phase = np.arctan2(im, re) / np.pi

    axes[0, col].plot(tlist, amp, color='green')
    axes[0, col].set_title(f'{name} pulse amplitude')
    axes[0, col].set_xlabel('time')
    axes[0, col].set_ylabel('amplitude')

    axes[1, col].plot(tlist, phase, color='purple')
    axes[1, col].set_title(f'{name} pulse phase (π)')
    axes[1, col].set_xlabel('time')
    axes[1, col].set_ylabel('phase / π')

plt.tight_layout()
plt.savefig('./graphs/dissipative_pulse_amp_phase.png', dpi=200)
plt.close()
print("Saved: ./graphs/dissipative_pulse_amp_phase.png")

# -------------------------
# Summary
# -------------------------
print(f"\nResults after {N_iter} iterations:")
print(f"  Guess     — final |3> pop: {out_guess.expect[2][-1]:.4f}  "
      f"norm: {norm_guess[-1]:.4f}")
print(f"  Optimised — final |3> pop: {out_opt.expect[2][-1]:.4f}  "
      f"norm: {norm_opt[-1]:.4f}")
print(f"  F_re = {F:.6f}   J_T = {J_T:.4e}")
print(f"  (example reports F ~ 0.90 after 40 iter, ~0.98 after 2000 iter)")