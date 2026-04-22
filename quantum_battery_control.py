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

# Parameters
omega = 1.0
g     = 0.2  * omega
gamma = 0.05 * omega
mu    = 0.5  * omega
Nb_T  = 0.0
kappa = 0.5

LAMBDA_VALUES = [15]

N_iter = 2000

tau   = np.pi / g
N_t   = 10001
tlist = np.linspace(0.0, tau, N_t)
dt    = tlist[1] - tlist[0]

# Operators
sx = sigmax(); sz = sigmaz(); sp = sigmap(); sm = sigmam(); I2 = qeye(2)

sx_A = tensor(sx, I2);  sz_A = tensor(sz, I2)
sp_A = tensor(sp, I2);  sm_A = tensor(sm, I2)
sx_B = tensor(I2, sx);  sz_B = tensor(I2, sz)
sp_B = tensor(I2, sp);  sm_B = tensor(I2, sm)

# Hamiltonian
H_A       = (omega / 2.0) * ((-sz_A) + tensor(I2, I2))
H_B       = (omega / 2.0) * ((-sz_B) + tensor(I2, I2))
H_int     = g * (sp_A * sm_B + sm_A * sp_B)
H_control = -mu * sx_A
H0        = H_A + H_B + H_int

# Collapse operators
c_ops = [np.sqrt(gamma * (Nb_T + 1.0)) * sm_A]

# Liouvillian operators
L0_mat = liouvillian(H0, c_ops).data.toarray()
Lc_mat = liouvillian(H_control).data.toarray()

# States
rho0    = ket2dm(tensor(basis(2, 0), basis(2, 0)))
rho_tgt = (
    ket2dm(tensor(basis(2, 0), basis(2, 1))) +
    ket2dm(tensor(basis(2, 1), basis(2, 1)))
) / 2.0

rho0_np  = operator_to_vector(rho0).full().flatten()
rtgt_np  = operator_to_vector(rho_tgt).full().flatten()

# Hilbert-Schmidt norm for normalization
rtgt_norm_sq       = np.real(rtgt_np.conj() @ rtgt_np)
rtgt_hs_norm       = np.sqrt(rtgt_norm_sq)
rtgt_normalized_np = rtgt_np / rtgt_hs_norm

# Shape function S(t)
ton = toff = 0.005 * tau

def S_of_t(t):
    if t < ton:
        return float(np.sin(np.pi * t / (2.0 * ton)) ** 2)
    if t > (tau - toff):
        return float(np.sin(np.pi * (tau - t) / (2.0 * toff)) ** 2)
    return 1.0

S_array = np.array([S_of_t(t) for t in tlist])

# Sinusoidal reference pulse
def eps_sin_func(t, args=None):
    return 2.0 * np.cos(omega * t)

# Krotov backward propagation
def backward_propagate(eps_arr, tau_val):
    chi_T  = (tau_val / rtgt_norm_sq) * (rtgt_np / rtgt_hs_norm)
    states = np.zeros((N_t, 16), dtype=complex)
    states[-1] = chi_T
    for k in range(N_t - 1, 0, -1):
        eps_mid = 0.5 * (eps_arr[k] + eps_arr[k - 1])
        L_tot   = L0_mat + eps_mid * Lc_mat
        prop    = scipy_expm(L_tot.conj().T * dt)
        states[k - 1] = prop @ states[k]
    return states

# Ergotropy computation
H_battery_local = (omega / 2.0) * (-sigmaz() + qeye(2))

def ergotropy_rho(rho_sys, H_b_local):
    rho_b   = ptrace(rho_sys, 1)
    rho_mat = rho_b.full()
    H_mat   = H_b_local.full()

    er, _  = np.linalg.eigh(rho_mat)
    eH, vH = np.linalg.eigh(H_mat)

    idx_r = np.argsort(er)[::-1]
    idx_H = np.argsort(eH)

    rho_passive = np.zeros_like(rho_mat, dtype=complex)
    for i, p_idx in enumerate(idx_r):
        vec = vH[:, idx_H[i]]
        rho_passive += er[p_idx] * np.outer(vec, vec.conj())

    E      = np.real(np.trace(rho_mat    @ H_mat))
    E_pass = np.real(np.trace(rho_passive @ H_mat))

    return E, max(0.0, E - E_pass)

# Compute sinusoidal reference curves once
opts = Options(nsteps=10000)
H_with_sin = [H0, [H_control, eps_sin_func]]

print("Propagating sinusoidal reference pulse ...")
out_sin = mesolve(H_with_sin, rho0, tlist, c_ops, [], options=opts)

E_sin   = np.zeros(N_t)
erg_sin = np.zeros(N_t)
for i in range(N_t):
    E_sin[i], erg_sin[i] = ergotropy_rho(out_sin.states[i], H_battery_local)

E_sin_n   = E_sin   / omega
erg_sin_n = erg_sin / omega
gt        = tlist * g

W_osc          = 2.0 * tau + np.sin(2.0 * omega * tau) / omega
eps_sin_array  = np.array([eps_sin_func(t) for t in tlist])

def forward_propagate_full(eps_arr):
    rho = rho0_np.copy()
    for k in range(N_t - 1):
        eps_mid = 0.5 * (eps_arr[k] + eps_arr[k + 1])
        prop    = scipy_expm((L0_mat + eps_mid * Lc_mat) * dt)
        rho     = prop @ rho
    return rho

for lambda_a in LAMBDA_VALUES:

    print(f"\n{'='*55}")
    print(f"  Running Krotov optimisation   lambda_a = {lambda_a}")
    print(f"{'='*55}")

    # Fresh initial guess for each lambda
    eps_array = S_array * kappa

    rho_T   = forward_propagate_full(eps_array)
    tau_val = rtgt_np.conj() @ rho_T

    tol              = 1e-6
    patience         = 50
    F_prev           = 0.0
    no_improve_count = 0
    converged_at     = N_iter

    for iteration in range(N_iter):

        bwd     = backward_propagate(eps_array, tau_val)
        new_eps = eps_array.copy()
        rho     = rho0_np.copy()

        for k in range(N_t - 1):
            X_vec      = 1j * Lc_mat @ rho
            delta      = (S_array[k] / lambda_a) * np.imag(bwd[k].conj() @ X_vec)
            new_eps[k] = eps_array[k] + delta

            eps_mid = 0.5 * (new_eps[k] + eps_array[k + 1])
            prop    = scipy_expm((L0_mat + eps_mid * Lc_mat) * dt)
            rho     = prop @ rho

        X_vec       = 1j * Lc_mat @ rho
        delta       = (S_array[-1] / lambda_a) * np.imag(bwd[-1].conj() @ X_vec)
        new_eps[-1] = eps_array[-1] + delta

        new_eps = new_eps * S_array   # enforce boundary condition

        F      = np.real(rtgt_np.conj() @ rho)
        F_norm = F / rtgt_norm_sq

        eps_array = new_eps

        if (iteration + 1) % 100 == 0 or iteration == 0:
            W_cur = np.trapz(eps_array**2, tlist)
            print(f"  iter {iteration + 1:4d}   fidelity = {F_norm:.6f}   W = {W_cur:.3f}")

        if abs(F_norm - F_prev) < tol:
            no_improve_count += 1
            if no_improve_count >= patience:
                converged_at = iteration + 1
                print(f"  Converged at iteration {converged_at}")
                break
        else:
            no_improve_count = 0
        F_prev = F_norm

    eps_opt_array = eps_array.copy()

    def make_interp_pulse(arr):
        def _pulse(t, args=None):
            return float(np.interp(t, tlist, arr))
        return _pulse

    H_with_opt = [H0, [H_control, make_interp_pulse(eps_opt_array)]]
    out_opt    = mesolve(H_with_opt, rho0, tlist, c_ops, [], options=opts)

    E_opt   = np.zeros(N_t)
    erg_opt = np.zeros(N_t)
    for i in range(N_t):
        E_opt[i], erg_opt[i] = ergotropy_rho(out_opt.states[i], H_battery_local)

    E_opt_n   = E_opt   / omega
    erg_opt_n = erg_opt / omega

    # Quality factors
    aE    = (erg_opt[-1] / erg_sin[-1] - 1.0) * 100.0 if erg_sin[-1] > 0 else float('nan')
    W_opt = np.trapz(eps_opt_array**2, tlist)
    aW    = (W_osc / W_opt - 1.0) * 100.0

    print(f"\n  Quality factors (lambda={lambda_a}, iter={converged_at}):")
    print(f"    alpha_E = {aE:.1f}%   (paper: ~9.7%)")
    print(f"    W_opt   = {W_opt:.2f}  (paper: ~6.38)")
    print(f"    W_osc   = {W_osc:.2f}  (paper: ~31.42)")
    print(f"    alpha_W = {aW:.1f}%  (paper: ~392%)")

    tag = str(lambda_a).replace('.', 'p')   # e.g. 9.9 -> "9p9"

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
    ax.set_title(
        f'Battery energy & ergotropy\n'
        f'λ={lambda_a}  |  αE={aE:.1f}%  αW={aW:.1f}%  W_opt={W_opt:.2f}',
        fontsize=9
    )
    plt.tight_layout()
    path_a = f'./graphs/energy_ergotropy_lambda{tag}.png'
    plt.savefig(path_a, dpi=200)
    plt.close()
    print(f"  Saved: {path_a}")

    fig, ax = plt.subplots(figsize=(5, 3))

    ax.plot(gt, eps_opt_array, color='green', linestyle='-',  linewidth=1.5, label='Optimised pulse')
    ax.plot(gt, eps_sin_array, color='black', linestyle='--', linewidth=1.5, label='Sinusoidal pulse')

    ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
    ax.set_xlabel(r'$g\tau$', fontsize=12)
    ax.set_ylabel(r'$\epsilon(t)$', fontsize=12)
    ax.set_xlim(0, np.pi)
    ax.legend(fontsize=9)
    ax.set_title(f'Field pulses   λ={lambda_a}  W_opt={W_opt:.2f}', fontsize=9)
    plt.tight_layout()
    path_b = f'./graphs/pulses_lambda{tag}.png'
    plt.savefig(path_b, dpi=200)
    plt.close()
    print(f"  Saved: {path_b}")

print("\nAll lambda sweeps complete.")