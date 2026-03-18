import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm as scipy_expm
from qutip import (basis, tensor, qeye, sigmax, sigmaz, sigmap, sigmam,
                   mesolve, ptrace, Options, ket2dm, liouvillian,
                   operator_to_vector)
import os

os.makedirs('./graphs', exist_ok=True)

# -------------------------
# Parameters (exactly from Fig.2 caption)
# -------------------------
omega    = 1.0
g        = 0.2  * omega   # charger-battery coupling
gamma    = 0.05 * omega   # dissipation rate (charger only)
mu       = 0.5  * omega   # coupling to laser field
Nb_T     = 0.0            # zero temperature
kappa    = 0.5            # initial guess amplitude (eq. 9)
lambda_a = 10.0           # Appendix D Fig.5: lambda~9.9 gives W_opt~6 with ~1300 steps
N_iter   = 1500           # Appendix D: ~1300-1500 steps needed at lambda~10

tau   = np.pi / g         # final time: maximises sinusoidal charging
N_t   = 1001
tlist = np.linspace(0.0, tau, N_t)
dt    = tlist[1] - tlist[0]

# -------------------------
# Operators
# -------------------------
sx = sigmax(); sz = sigmaz(); sp = sigmap(); sm = sigmam(); I2 = qeye(2)

sx_A = tensor(sx, I2);  sz_A = tensor(sz, I2)
sp_A = tensor(sp, I2);  sm_A = tensor(sm, I2)
sx_B = tensor(I2, sx);  sz_B = tensor(I2, sz)
sp_B = tensor(I2, sp);  sm_B = tensor(I2, sm)

# -------------------------
# Hamiltonian — equation (15)
# H_A = (w/2)(-sz+I): ground state energy=0, excited energy=omega
# -------------------------
H_A       = (omega / 2.0) * ((-sz_A) + tensor(I2, I2))
H_B       = (omega / 2.0) * ((-sz_B) + tensor(I2, I2))
H_int     = g * (sp_A * sm_B + sm_A * sp_B)
H_control = -mu * sx_A
H0        = H_A + H_B + H_int

# -------------------------
# Collapse operators — equation (17), zero temperature
# -------------------------
c_ops = [np.sqrt(gamma * (Nb_T + 1.0)) * sm_A]

# -------------------------
# Liouvillian superoperators as dense 16x16 numpy matrices
#
#   L0_mat  = -i[H0, .] + D_T[.]    (drift + dissipation)
#   Lc_mat  = -i[H_control, .]      (control, no dissipation)
#
# These act on the vectorized density matrix vec(rho) (shape 16):
#   d vec(rho)/dt = (L0_mat + eps(t)*Lc_mat) @ vec(rho)
# -------------------------
L0_mat = liouvillian(H0, c_ops).data.toarray()   # 16x16
Lc_mat = liouvillian(H_control).data.toarray()   # 16x16

# -------------------------
# States: vectorized density matrices (numpy arrays, shape (16,))
# -------------------------
rho0    = ket2dm(tensor(basis(2, 0), basis(2, 0)))  # |00><00|
rho_tgt = ket2dm(tensor(basis(2, 0), basis(2, 1)))  # |01><01|

rho0_np  = operator_to_vector(rho0).full().flatten()    # (16,)
rtgt_np  = operator_to_vector(rho_tgt).full().flatten() # (16,)

# -------------------------
# Shape function S(t) — equation (8)
# -------------------------
ton = toff = 0.005 * tau

def S_of_t(t):
    if t < ton:
        return float(np.sin(np.pi * t / (2.0 * ton)) ** 2)
    if t > (tau - toff):
        return float(np.sin(np.pi * (tau - t) / (2.0 * toff)) ** 2)
    return 1.0

S_array = np.array([S_of_t(t) for t in tlist])

# -------------------------
# Sinusoidal reference pulse — Appendix A, equation (A4)
# eps(t) = E0*cos(wt) with E0=2, giving F=mu=0.5w.
# Pure cosine, no S(t) envelope — the paper compares against an unmodified
# sinusoidal drive. W_osc = 2*tau + sin(2*w*tau)/w ~ 31.42  [eq. 20]
# -------------------------
def eps_sin_func(t, args=None):
    return 2.0 * np.cos(omega * t)

# -------------------------
# Initial guess — equation (9): eps^(0)(t) = S(t) * kappa
# The flat shaped guess is what the paper specifies. Starting from the
# sinusoidal adds Krotov corrections ON TOP of ±2, producing a pulse
# larger than sinusoidal. Starting from near-zero lets the algorithm
# build the optimal pulse from scratch, which the paper shows uses
# far less energy (W_opt ~ 6.38) than the sinusoidal (W_osc ~ 31.42).
# -------------------------
eps_array = S_array * kappa   # shape (N_t,), max amplitude = kappa = 0.5

# ============================================================
# Manual Krotov algorithm for open quantum systems
# Reference: Section 3, equations (5), (11), (13), (14)
#
# FORWARD (eq. 5):
#   d vec(rho)/dt = L vec(rho),  L = L0 + eps*Lc
#   vec(rho(t+dt)) = expm(L*dt) @ vec(rho(t))
#
# BACKWARD (eq. 13):
#   d vec(sigma)/dt = -L† vec(sigma),  sigma(tau) = rho_tgt
#   vec(sigma(t-dt)) = expm(L†*dt) @ vec(sigma(t))
#
# UPDATE (eq. 14):
#   Δeps(t) = (S(t)/λ) Im{ vec(sigma)^H · [i*Lc · vec(rho)] }
#
# Trace identity:
#   Tr[sigma · X] = vec(sigma)^H · vec(X)   (sigma Hermitian,
#   column-stack convention: vec(A)^H·vec(B) = Tr[A†B] = Tr[sigma·X])
# ============================================================

def backward_propagate(eps_arr):
    """
    Backward-propagate sigma from tau to 0.
    sigma(tau) = rho_tgt.
    d sigma/dt = -L† sigma  =>  sigma(t-dt) = expm(L†*dt) sigma(t).
    Returns array shape (N_t, 16).
    """
    states = np.zeros((N_t, 16), dtype=complex)
    states[-1] = rtgt_np
    for k in range(N_t - 1, 0, -1):
        eps_mid = 0.5 * (eps_arr[k] + eps_arr[k - 1])
        L_tot   = L0_mat + eps_mid * Lc_mat
        prop    = scipy_expm(L_tot.conj().T * dt)   # expm(L† dt)
        states[k - 1] = prop @ states[k]
    return states


print("Running manual Krotov optimisation ...")

F_prev          = 0.0
no_improve_count = 0
tol             = 1e-6   # fidelity change tolerance
patience        = 40     # stop after this many non-improving iterations

for iteration in range(N_iter):

    # Step 1: backward pass under previous pulse
    bwd = backward_propagate(eps_array)

    # Step 2: forward pass with sequential pulse update
    new_eps = eps_array.copy()
    rho     = rho0_np.copy()

    for k in range(N_t - 1):
        # Pulse update at t_k using current rho and old sigma (eq. 14)
        X_vec      = 1j * Lc_mat @ rho
        delta      = (S_array[k] / lambda_a) * np.imag(bwd[k].conj() @ X_vec)
        new_eps[k] = eps_array[k] + delta

        # Propagate rho: use updated eps[k], old eps[k+1]
        eps_mid = 0.5 * (new_eps[k] + eps_array[k + 1])
        prop    = scipy_expm((L0_mat + eps_mid * Lc_mat) * dt)
        rho     = prop @ rho

    # Update last point
    X_vec       = 1j * Lc_mat @ rho
    delta       = (S_array[-1] / lambda_a) * np.imag(bwd[-1].conj() @ X_vec)
    new_eps[-1] = eps_array[-1] + delta

    # Re-enforce S(t) shape constraint: pulse must be zero at boundaries.
    # The update rule preserves this in theory (delta ~ S(t)/lambda), but
    # accumulated floating-point drift can violate it over many iterations.
    # Multiplying by S_array restores the shape without changing the interior.
    new_eps = new_eps * S_array

    # Fidelity: F = Re{ Tr[rho_tgt† rho(T)] } = Re{ vec(rho_tgt)^H @ vec(rho(T)) }
    F = np.real(rtgt_np.conj() @ rho)

    eps_array = new_eps

    if (iteration + 1) % 20 == 0 or iteration == 0:
        W_cur = np.trapz(eps_array**2, tlist)
        print(f"  iter {iteration + 1:4d}   fidelity = {F:.6f}   W = {W_cur:.3f}")

    # Early stopping: halt when fidelity stops improving
    if abs(F - F_prev) < tol:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"  Converged at iteration {iteration + 1} (fidelity change < {tol})")
            break
    else:
        no_improve_count = 0
    F_prev = F

print("Optimisation finished.")
eps_opt_array = eps_array.copy()

# -------------------------
# Final time evolution using QuTiP mesolve
# -------------------------
def make_interp_pulse(arr):
    def _pulse(t, args=None):
        return float(np.interp(t, tlist, arr))
    return _pulse

H_with_opt = [H0, [H_control, make_interp_pulse(eps_opt_array)]]
H_with_sin = [H0, [H_control, eps_sin_func]]

opts = Options(nsteps=10000)

print("Propagating optimised pulse ...")
out_opt = mesolve(H_with_opt, rho0, tlist, c_ops, [], options=opts)
print("Propagating sinusoidal pulse ...")
out_sin = mesolve(H_with_sin, rho0, tlist, c_ops, [], options=opts)

# -------------------------
# Battery Hamiltonian — ground=0, excited=omega
# -------------------------
H_battery_local = (omega / 2.0) * (-sigmaz() + qeye(2))

def ergotropy_rho(rho_sys, H_b_local):
    """
    Compute battery energy E and ergotropy E(rho) = E - E(rho_passive).
    Equation (1) of the paper.
    """
    rho_b   = ptrace(rho_sys, 1)
    rho_mat = rho_b.full()
    H_mat   = H_b_local.full()

    er, _  = np.linalg.eigh(rho_mat)
    eH, vH = np.linalg.eigh(H_mat)

    idx_r = np.argsort(er)[::-1]   # descending population
    idx_H = np.argsort(eH)         # ascending energy

    rho_passive = np.zeros_like(rho_mat, dtype=complex)
    for i, p_idx in enumerate(idx_r):
        vec = vH[:, idx_H[i]]
        rho_passive += er[p_idx] * np.outer(vec, vec.conj())

    E      = np.real(np.trace(rho_mat    @ H_mat))
    E_pass = np.real(np.trace(rho_passive @ H_mat))

    return E, max(0.0, E - E_pass)

# -------------------------
# Observables at every time step
# -------------------------
E_opt   = np.zeros(N_t)
erg_opt = np.zeros(N_t)
E_sin   = np.zeros(N_t)
erg_sin = np.zeros(N_t)

for i in range(N_t):
    E_opt[i],   erg_opt[i] = ergotropy_rho(out_opt.states[i], H_battery_local)
    E_sin[i],   erg_sin[i] = ergotropy_rho(out_sin.states[i], H_battery_local)

E_opt_n   = E_opt   / omega
erg_opt_n = erg_opt / omega
E_sin_n   = E_sin   / omega
erg_sin_n = erg_sin / omega

gt = tlist * g   # x-axis: g*t in [0, pi]

# -------------------------
# Quality factors — equations (3) and (4)
# -------------------------
aE    = (erg_opt[-1] / erg_sin[-1] - 1.0) * 100.0 if erg_sin[-1] > 0 else float('nan')
W_osc = 2.0 * tau + np.sin(2.0 * omega * tau) / omega   # eq. 20: ~31.42
W_opt = np.trapz(eps_opt_array ** 2, tlist)
aW    = (W_osc / W_opt - 1.0) * 100.0

print(f"\nQuality factors at t = tau:")
print(f"  E_opt (ergotropy) = {erg_opt[-1]:.4f} omega")
print(f"  E_sin (ergotropy) = {erg_sin[-1]:.4f} omega")
print(f"  alpha_E = {aE:.1f}%    (paper: ~9.7%)")
print(f"  W_opt   = {W_opt:.2f}  (paper: ~6.38)")
print(f"  W_osc   = {W_osc:.2f}  (paper: ~31.42)")
print(f"  alpha_W = {aW:.1f}%   (paper: ~392%)")

# -------------------------
# Figure 2(a): Energy & Ergotropy vs g*t
# -------------------------
fig, ax = plt.subplots(figsize=(5, 4))

ax.plot(gt, E_opt_n,   color='green', linestyle='-',  linewidth=1.5, label='Energy (opt)')
ax.plot(gt, erg_opt_n, color='green', linestyle='-.', linewidth=1.5, label='Ergotropy (opt)', alpha=0.9)
ax.plot(gt, E_sin_n,   color='black', linestyle='-',  linewidth=1.5, label='Energy (sin)')
ax.plot(gt, erg_sin_n, color='black', linestyle='-.', linewidth=1.5, label='Ergotropy (sin)', alpha=0.9)

ax.set_xlabel(r'$g\tau$', fontsize=12)
ax.set_ylabel(r'$E_B/\omega$,  $\mathcal{E}_B/\omega$', fontsize=11)
ax.set_xlim(0, np.pi)
ax.set_ylim(bottom=0)
ax.legend(fontsize=9)
ax.set_title('Fig. 2(a) — Battery energy & ergotropy', fontsize=10)
plt.tight_layout()
plt.savefig('./graphs/fig2a_energy_ergotropy.png', dpi=200)
plt.close()
print("Saved: ./graphs/fig2a_energy_ergotropy.png")

# -------------------------
# Figure 2(b): Pulse shapes
# -------------------------
eps_sin_array = np.array([eps_sin_func(t) for t in tlist])

fig, ax = plt.subplots(figsize=(5, 3))

ax.plot(gt, eps_opt_array, color='green', linestyle='-',  linewidth=1.5, label='Optimised pulse')
ax.plot(gt, eps_sin_array, color='black', linestyle='--', linewidth=1.5, label='Sinusoidal pulse')

ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
ax.set_xlabel(r'$g\tau$', fontsize=12)
ax.set_ylabel(r'$\epsilon(t)$', fontsize=12)
ax.set_xlim(0, np.pi)
ax.legend(fontsize=9)
ax.set_title('Fig. 2(b) — Field pulses', fontsize=10)
plt.tight_layout()
plt.savefig('./graphs/fig2b_pulses.png', dpi=200)
plt.close()
print("Saved: ./graphs/fig2b_pulses.png")