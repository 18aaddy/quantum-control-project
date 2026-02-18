import numpy as np
import matplotlib.pyplot as plt
from qutip import (basis, tensor, qeye, sigmax, sigmay, sigmaz, sigmap, sigmam,
                   mesolve, expect, ptrace)
import krotov

# Parameters
omega = 1.0
g = 0.2 * omega
gamma = 0.05 * omega
mu = 0.5 * omega
F_amp = 0.5 * omega
Nb_T = 0.0
kappa = 0.5
tau = np.pi / g
N_t = 1001
tlist = np.linspace(0.0, tau, N_t)

# Operators & Hamiltonian
sx = sigmax()
sz = sigmaz()
sp = sigmap()
sm = sigmam()
I = qeye(2)

# system A = charger, B battery
sx_A = tensor(sx, I)
sz_A = tensor(sz, I)
sp_A = tensor(sp, I)
sm_A = tensor(sm, I)

sx_B = tensor(I, sx)
sz_B = tensor(I, sz)
sp_B = tensor(I, sp)
sm_B = tensor(I, sm)

H_A = 0.5 * omega * sz_A
H_B = 0.5 * omega * sz_B

H_int = g * (sp_A * sm_B + sm_A * sp_B)

H_control = - mu * sx_A

# drift Hamiltonian
H0 = H_A + H_B + H_int

# Dissipation (local on charger A)
c_ops = []
c_ops.append(np.sqrt(gamma * (Nb_T + 1.0)) * sm_A)

psi0 = tensor(basis(2, 0), basis(2, 0))   # both ground
psitgt = tensor(basis(2, 0), basis(2, 1))  # system in excited, charger ground

# shape function
def S_of_t(t, t_final=tau, ton_frac=0.005, toff_frac=0.005):
    ton = ton_frac * t_final
    toff = toff_frac * t_final
    if t < ton:
        return np.sin(np.pi * t / (2 * ton)) ** 2
    if t > (t_final - toff):
        return np.sin(np.pi * (t_final - t) / (2 * toff)) ** 2
    return 1.0

S_array = np.array([S_of_t(t) for t in tlist])

# initial guess pulse
def eps_guess(t, args=None):
    return float(S_of_t(t) * kappa)

def eps_sin(t, args=None):
    return float(S_of_t(t) * F_amp * np.cos(omega * t))

H_td = [H0, [H_control, eps_guess]]

obj = krotov.Objective(initial_state=psi0, target=psitgt, H=H_td)

def S_update(t):
    return S_of_t(t)

pulse_options = {
    eps_guess: dict(lambda_a=1.0, update_shape=S_update)
}

opt_result = krotov.optimize_pulses(
    objectives=[obj],
    pulse_options=pulse_options,
    tlist=tlist,
    propagator=krotov.propagators.expm,
    chi_constructor=krotov.functionals.chis_re,
    iter_stop=10,
)

opt_pulses = opt_result.optimized_controls
eps_opt_array = np.array(opt_pulses[0])

if eps_opt_array.size != tlist.size:
    eps_opt_array = np.array([eps_guess(t) for t in tlist])

H_with_opt = [H0, [H_control, lambda t, args: float(np.interp(t, tlist, eps_opt_array))]]
H_with_sin = [H0, [H_control, eps_sin]]

# evolve density matrices
out_opt = mesolve(H_with_opt, psi0, tlist, c_ops, [])
out_sin = mesolve(H_with_sin, psi0, tlist, c_ops, [])

# reduce to battery and compute energy and ergotropy
H_battery_local = 0.5 * omega * (-sigmaz() + qeye(2))

def ergotropy_rho(rho_sys, H_b_local):
    rho_b = ptrace(rho_sys, 1)
    rho_mat = rho_b.full()
    H_mat = H_b_local.full()

    er, vr = np.linalg.eigh(rho_mat)
    eH, vH = np.linalg.eigh(H_mat)

    idx_r = np.argsort(er)[::-1]
    idx_H = np.argsort(eH)

    rho_passive = np.zeros_like(rho_mat, dtype=complex)
    for i, p_idx in enumerate(idx_r):
        vec = vH[:, idx_H[i]]
        rho_passive += er[p_idx] * np.outer(vec, vec.conj())

    E = np.real(np.trace(rho_mat @ H_mat))
    E_pass = np.real(np.trace(rho_passive @ H_mat))

    return E, max(0.0, E - E_pass)

E_opt = np.zeros_like(tlist)
erg_opt = np.zeros_like(tlist)
E_sin = np.zeros_like(tlist)
erg_sin = np.zeros_like(tlist)

for i, t in enumerate(tlist):
    rho_opt = out_opt.states[i]
    rho_sin = out_sin.states[i]
    E_opt[i], erg_opt[i] = ergotropy_rho(rho_opt, H_battery_local)
    E_sin[i], erg_sin[i] = ergotropy_rho(rho_sin, H_battery_local)

# Plot & save figures
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
