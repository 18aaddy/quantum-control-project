import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm as scipy_expm
from qutip import (sigmaz, sigmax, ket, ket2dm, mesolve, Options)
import os

os.makedirs('./graphs', exist_ok=True)

omega    = 1.0
ampl0    = 0.2    # guess field amplitude
T        = 5.0
N_t      = 500    # number of time steps
lambda_a = 5.0    # Krotov step-size penalty
N_iter   = 50

tlist = np.linspace(0.0, T, N_t)
dt    = tlist[1] - tlist[0]

# Operators
H0     = -0.5 * omega * sigmaz()
H1     = sigmax()
H0_mat = H0.full()
H1_mat = H1.full()

# States
psi_init    = ket("0")
psi_tgt     = ket("1")
psi_init_np = psi_init.full().flatten()   # shape (2,)
psi_tgt_np  = psi_tgt.full().flatten()    # shape (2,)

t_rise = 0.3
t_fall = 0.3
_bmax  = 0.42 - 0.5 * np.cos(np.pi) + 0.08 * np.cos(2 * np.pi)   # = 1.0

def S_of_t(t):
    if t <= 0 or t >= T:
        return 0.0
    if t < t_rise:
        x = t / t_rise
        return (0.42 - 0.5 * np.cos(np.pi * x) + 0.08 * np.cos(2 * np.pi * x)) / _bmax
    if t > T - t_fall:
        x = (T - t) / t_fall
        return (0.42 - 0.5 * np.cos(np.pi * x) + 0.08 * np.cos(2 * np.pi * x)) / _bmax
    return 1.0

S_array = np.array([S_of_t(t) for t in tlist])

# Guess field: ampl0 * S(t)
eps_guess_array = ampl0 * S_array
eps_array       = eps_guess_array.copy()


def backward_propagate(eps_arr, tau):
    states     = np.zeros((N_t, 2), dtype=complex)
    states[-1] = tau * psi_tgt_np          # scaled by complex overlap
    for k in range(N_t - 1, 0, -1):
        eps_mid    = 0.5 * (eps_arr[k] + eps_arr[k - 1])
        H_tot      = H0_mat + eps_mid * H1_mat
        prop       = scipy_expm(1j * H_tot * dt)
        states[k-1] = prop @ states[k]
    return states


print("Running Krotov optimisation ...")
print(f"  |0> -> |1>,  T={T},  N_t={N_t},  lambda_a={lambda_a}\n")

F_prev           = 0.0
no_improve_count = 0
tol              = 1e-8
patience         = 10

def forward_propagate(eps_arr):
    psi = psi_init_np.copy()
    for k in range(N_t - 1):
        eps_mid = 0.5 * (eps_arr[k] + eps_arr[k + 1])
        H_tot   = H0_mat + eps_mid * H1_mat
        psi     = scipy_expm(-1j * H_tot * dt) @ psi
    return psi

psi_T = forward_propagate(eps_array)
tau   = psi_tgt_np.conj() @ psi_T    # <psi_tgt|psi(T)>

for iteration in range(N_iter):

    bwd     = backward_propagate(eps_array, tau)
    new_eps = eps_array.copy()
    psi     = psi_init_np.copy()

    for k in range(N_t - 1):
        delta      = (S_array[k] / lambda_a) * np.imag(bwd[k].conj() @ (H1_mat @ psi))
        new_eps[k] = eps_array[k] + delta

        eps_mid = 0.5 * (new_eps[k] + eps_array[k + 1])
        H_tot   = H0_mat + eps_mid * H1_mat
        psi     = scipy_expm(-1j * H_tot * dt) @ psi

    # Last point
    delta       = (S_array[-1] / lambda_a) * np.imag(bwd[-1].conj() @ (H1_mat @ psi))
    new_eps[-1] = eps_array[-1] + delta

    # Enforce boundary shape
    new_eps = new_eps * S_array

    # Compute tau and fidelity from new psi(T)
    tau  = psi_tgt_np.conj() @ psi    # <psi_tgt|psi(T)>, used next iteration
    F    = np.abs(tau) ** 2
    J_T  = 1.0 - F

    eps_array = new_eps

    print(f"  iter {iteration + 1:3d}   J_T = {J_T:.4e}   F = {F:.6f}")

    if J_T < 1e-3:
        print(f"\n  Converged: J_T < 1e-3 at iteration {iteration + 1}")
        break

    if abs(F - F_prev) < tol:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"\n  Stopped: no fidelity improvement for {patience} iterations")
            break
    else:
        no_improve_count = 0
    F_prev = F

print("\nOptimisation finished.")
eps_opt_array = eps_array.copy()

proj0 = ket2dm(ket("0"))
proj1 = ket2dm(ket("1"))
opts  = Options(nsteps=10000)

def make_pulse(arr):
    def _f(t, args=None):
        return float(np.interp(t, tlist, arr))
    return _f

H_guess = [H0, [H1, make_pulse(eps_guess_array)]]
H_opt   = [H0, [H1, make_pulse(eps_opt_array)]]

print("Propagating guess pulse ...")
out_guess = mesolve(H_guess, psi_init, tlist, [], [proj0, proj1], options=opts)
print("Propagating optimised pulse ...")
out_opt   = mesolve(H_opt,   psi_init, tlist, [], [proj0, proj1], options=opts)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(tlist, out_guess.expect[0], label='|0⟩', color='#1f77b4')
axes[0].plot(tlist, out_guess.expect[1], label='|1⟩', color='#ff7f0e')
axes[0].set_title('Population dynamics — guess field')
axes[0].set_xlabel('time')
axes[0].set_ylabel('population')
axes[0].legend()
axes[0].set_ylim(-0.05, 1.05)

axes[1].plot(tlist, out_opt.expect[0], label='|0⟩', color='#1f77b4')
axes[1].plot(tlist, out_opt.expect[1], label='|1⟩', color='#ff7f0e')
axes[1].set_title(f'Population dynamics — optimised field  (J_T={J_T:.2e})')
axes[1].set_xlabel('time')
axes[1].set_ylabel('population')
axes[1].legend()
axes[1].set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('./graphs/population_dynamics.png', dpi=200)
plt.close()
print("Saved: ./graphs/population_dynamics.png")

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(tlist, eps_guess_array, label='guess',     color='black', linestyle='--', linewidth=1.5)
ax.plot(tlist, eps_opt_array,   label='optimised', color='green', linestyle='-',  linewidth=1.5)
ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
ax.set_xlabel('time')
ax.set_ylabel('ε(t)')
ax.set_title('Control fields')
ax.legend()
plt.tight_layout()
plt.savefig('./graphs/control_fields.png', dpi=200)
plt.close()
print("Saved: ./graphs/control_fields.png")

print(f"\nResults:")
print(f"  Guess field   — final |1> population: {out_guess.expect[1][-1]:.6f}")
print(f"  Optimised     — final |1> population: {out_opt.expect[1][-1]:.6f}  (target: 1.0)")
print(f"  J_T (opt)     = {J_T:.4e}  (target: < 1e-3)")