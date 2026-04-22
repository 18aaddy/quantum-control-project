import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm as scipy_expm
from qutip import Qobj, ket2dm, mesolve, Options
import os

os.makedirs('./graphs', exist_ok=True)

# -------------------------
# Parameters (matching the example exactly)
# -------------------------
E1      = 0.0
E2      = 10.0
E3      = 5.0
omega_P = 9.5
omega_S = 4.5
T       = 5.0
N_t     = 500
lambda_a = 0.5    # same for all 4 controls
N_iter  = 15      # example runs 15 iterations

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
# H_RWA = H0
#       + Omega_P1(t) * HP_re
#       + Omega_P2(t) * HP_im
#       + Omega_S1(t) * HS_re
#       + Omega_S2(t) * HS_im
# -------------------------
H0_mat = np.array([
    [Delta_P, 0.0,     0.0    ],
    [0.0,     0.0,     0.0    ],
    [0.0,     0.0,     Delta_S]
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
    [0.0, 0.0,   0.0 ],
    [0.0, 0.0,   1.0j],
    [0.0, -1.0j, 0.0 ]
], dtype=complex)

# List of control Hamiltonians (same order as controls below)
H_ctrl_mats = [HP_re_mat, HP_im_mat, HS_re_mat, HS_im_mat]

# -------------------------
# States (numpy arrays, shape (3,))
# -------------------------
ket1 = np.array([1.0, 0.0, 0.0], dtype=complex)
ket2 = np.array([0.0, 1.0, 0.0], dtype=complex)
ket3 = np.array([0.0, 0.0, 1.0], dtype=complex)

psi_init = ket1.copy()

# Phase-sensitive target in the rotating frame:
#   psi_tgt = exp(i*(E2 - omega_S)*T) * ket3
psi_tgt = np.exp(1j * (E2 - omega_S) * T) * ket3

# -------------------------
# Shape function S(t) — sinsq flattop with t_rise = 0.3
#
# Matches krotov.shapes.flattop(..., func='sinsq'):
#   sin^2(pi*t / (2*t_rise))  for the ramp-on region
#   sin^2(pi*(T-t)/(2*t_fall)) for the ramp-off region
#   1 in the flat middle
# -------------------------
t_rise = 0.3
t_fall = 0.3

def S_of_t(t):
    if t <= 0.0 or t >= T:
        return 0.0
    if t < t_rise:
        return float(np.sin(np.pi * t / (2.0 * t_rise)) ** 2)
    if t > T - t_fall:
        return float(np.sin(np.pi * (T - t) / (2.0 * t_fall)) ** 2)
    return 1.0

S_array = np.array([S_of_t(t) for t in tlist])

# -------------------------
# Blackman shape for guess pulses
#
# Matches krotov.shapes.blackman(t, t_start, t_stop):
#   0.42 - 0.5*cos(2*pi*(t-t_start)/(t_stop-t_start))
#         + 0.08*cos(4*pi*(t-t_start)/(t_stop-t_start))
#   normalized so peak = 1
# -------------------------
def blackman(t, t_start, t_stop):
    if t <= t_start or t >= t_stop:
        return 0.0
    t_w = t_stop - t_start
    x   = (t - t_start) / t_w
    val = 0.42 - 0.5 * np.cos(2 * np.pi * x) + 0.08 * np.cos(4 * np.pi * x)
    # normalize so peak (at x=0.5) = 1
    peak = 0.42 - 0.5 * np.cos(np.pi) + 0.08 * np.cos(2 * np.pi)  # = 1.0
    return float(val / peak)

# -------------------------
# Guess controls (4 real arrays)
# Pump: real part is Blackman on [2, 5], imaginary part = 0
# Stokes: real part is Blackman on [0, 3], imaginary part = 0
# -------------------------
Omega0 = 5.0

eps_P1_guess = np.array([Omega0 * blackman(t, 2.0, 5.0) for t in tlist])
eps_P2_guess = np.zeros(N_t)
eps_S1_guess = np.array([Omega0 * blackman(t, 0.0, 3.0) for t in tlist])
eps_S2_guess = np.zeros(N_t)

# Stack as (4, N_t) array: rows = [P1, P2, S1, S2]
eps_arrays = np.stack([eps_P1_guess, eps_P2_guess,
                       eps_S1_guess, eps_S2_guess])

# ============================================================
# Manual Krotov for closed system with 4 controls
#
# Forward propagator (one step):
#   H_total = H0 + sum_l eps_l(t) * H_ctrl_l
#   |psi(t+dt)> = expm(-i * H_total * dt) |psi>
#
# Backward (co-state):
#   |chi(t-dt)> = expm(+i * H_total * dt) |chi>
#
# Functional: J_T_re = 1 - Re{<psi_tgt|psi(T)>}
# chi(T) = psi_tgt  (chis_re boundary condition, no tau scaling)
#
# Update rule for each control l:
#   delta_eps_l(t) = (S(t)/lambda_a) * Im{ <chi(t)| H_ctrl_l |psi(t)> }
# ============================================================

def build_H_total(eps_vals):
    """Build total Hamiltonian from array of control values."""
    H = H0_mat.copy()
    for l, e in enumerate(eps_vals):
        H += e * H_ctrl_mats[l]
    return H

def backward_propagate(eps_arrs):
    """
    Backward propagation for chis_re functional.
    chi(T) = psi_tgt  (no tau scaling — this is chis_re not chis_ss)
    Returns states array shape (N_t, 3).
    """
    states     = np.zeros((N_t, 3), dtype=complex)
    states[-1] = psi_tgt.copy()
    for k in range(N_t - 1, 0, -1):
        # midpoint control values
        eps_mid = 0.5 * (eps_arrs[:, k] + eps_arrs[:, k - 1])
        H_tot   = build_H_total(eps_mid)
        prop    = scipy_expm(1j * H_tot * dt)   # backward: +i
        states[k - 1] = prop @ states[k]
    return states


print("Running Krotov optimisation (Lambda system) ...")
print(f"  3-level system,  T={T},  N_t={N_t},  lambda_a={lambda_a}")
print(f"  |1> -> exp(i*phi)*|3>  (phase-sensitive, chis_re)\n")

F_prev           = 0.0
no_improve_count = 0
tol              = 1e-8
patience         = 5

for iteration in range(N_iter):

    bwd      = backward_propagate(eps_arrays)
    new_eps  = eps_arrays.copy()
    psi      = psi_init.copy()

    for k in range(N_t - 1):
        # Update each control independently
        for l in range(4):
            Hl_psi     = H_ctrl_mats[l] @ psi
            delta      = (S_array[k] / lambda_a) * np.imag(bwd[k].conj() @ Hl_psi)
            new_eps[l, k] = eps_arrays[l, k] + delta

        # Propagate forward under updated eps at k, old eps at k+1
        eps_mid = 0.5 * (new_eps[:, k] + eps_arrays[:, k + 1])
        H_tot   = build_H_total(eps_mid)
        psi     = scipy_expm(-1j * H_tot * dt) @ psi

    # Last time point
    for l in range(4):
        Hl_psi        = H_ctrl_mats[l] @ psi
        delta         = (S_array[-1] / lambda_a) * np.imag(bwd[-1].conj() @ Hl_psi)
        new_eps[l, -1] = eps_arrays[l, -1] + delta

    # Enforce S(t) boundary shape on all controls
    new_eps = new_eps * S_array   # broadcasts (4, N_t) * (N_t,)

    # J_T_re = 1 - Re{<psi_tgt|psi(T)>}
    F   = np.real(psi_tgt.conj() @ psi)
    J_T = 1.0 - F

    eps_arrays = new_eps

    print(f"  iter {iteration + 1:3d}   J_T = {J_T:.4e}   F = {F:.6f}")

    if J_T < 1e-3:
        print(f"\n  Converged: J_T < 1e-3 at iteration {iteration + 1}")
        break

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
# Simulate dynamics with mesolve for plotting
# -------------------------
opts = Options(nsteps=10000)

# Build QuTiP Hamiltonians for mesolve
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

# Guess Hamiltonian
H_guess = [
    H0_qutip,
    [HP_re_qutip, make_pulse(eps_P1_guess)],
    [HP_im_qutip, make_pulse(eps_P2_guess)],
    [HS_re_qutip, make_pulse(eps_S1_guess)],
    [HS_im_qutip, make_pulse(eps_S2_guess)],
]

# Optimised Hamiltonian
H_opt = [
    H0_qutip,
    [HP_re_qutip, make_pulse(eps_opt[0])],
    [HP_im_qutip, make_pulse(eps_opt[1])],
    [HS_re_qutip, make_pulse(eps_opt[2])],
    [HS_im_qutip, make_pulse(eps_opt[3])],
]

psi_init_qutip = Qobj(psi_init)

print("Propagating guess pulses ...")
out_guess = mesolve(H_guess, psi_init_qutip, tlist, [],
                    [proj1, proj2, proj3], options=opts)
print("Propagating optimised pulses ...")
out_opt   = mesolve(H_opt,   psi_init_qutip, tlist, [],
                    [proj1, proj2, proj3], options=opts)

# -------------------------
# Figure 1: Population dynamics — guess vs optimised
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, out, title in zip(
    axes,
    [out_guess, out_opt],
    ['Population dynamics — guess', f'Population dynamics — optimised  (J_T={J_T:.2e})']
):
    ax.plot(tlist, out.expect[0], label='|1⟩', color='#1f77b4')
    ax.plot(tlist, out.expect[1], label='|2⟩', color='#ff7f0e')
    ax.plot(tlist, out.expect[2], label='|3⟩', color='#2ca02c')
    ax.set_title(title)
    ax.set_xlabel('time')
    ax.set_ylabel('population')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('./graphs/lambda_population.png', dpi=200)
plt.close()
print("Saved: ./graphs/lambda_population.png")

# -------------------------
# Figure 2: Control pulses — guess vs optimised
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 6))

labels   = ['Ωₚ real (guess)', 'Ωₚ imag (guess)', 'Ωₛ real (guess)', 'Ωₛ imag (guess)']
guesses  = [eps_P1_guess, eps_P2_guess, eps_S1_guess, eps_S2_guess]
opt_list = [eps_opt[0], eps_opt[1], eps_opt[2], eps_opt[3]]
opt_lbl  = ['Ωₚ real (opt)', 'Ωₚ imag (opt)', 'Ωₛ real (opt)', 'Ωₛ imag (opt)']

for i, ax in enumerate(axes.flat):
    ax.plot(tlist, guesses[i],  color='black', linestyle='--', linewidth=1.2, label='guess')
    ax.plot(tlist, opt_list[i], color='green', linestyle='-',  linewidth=1.2, label='opt')
    ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
    ax.set_xlabel('time')
    ax.set_ylabel('amplitude')
    ax.set_title(opt_lbl[i])
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('./graphs/lambda_controls.png', dpi=200)
plt.close()
print("Saved: ./graphs/lambda_controls.png")

# -------------------------
# Summary
# -------------------------
print(f"\nResults:")
print(f"  Guess   — final |3> population: {out_guess.expect[2][-1]:.6f}")
print(f"  Optimised — final |3> population: {out_opt.expect[2][-1]:.6f}  (target: 1.0)")
print(f"  J_T (opt) = {J_T:.4e}  (target: < 1e-3)")