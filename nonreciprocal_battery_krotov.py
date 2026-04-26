import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

os.makedirs('./graphs', exist_ok=True)

# Parameters
omega     = 1.0
omega_L   = omega           # resonant drive: omega_L - omega = 0
kappa_c   = 0.003 * omega
kappa_b   = 0.003 * omega
Gamma     = 0.04  * omega
epsilon_0 = 0.1   * omega   # guess pulse amplitude

mu = -1.0 + 0j              # mu =-p_b·p_c*=-1

Lambda_c = Gamma + kappa_c
Lambda_b = Gamma + kappa_b
J        = Gamma / 2.0

# Target time Jt = 7
T_target = 7.0 / J
N_t      = 5000
tlist    = np.linspace(0.0, T_target, N_t)
dt       = tlist[1] - tlist[0]

N_iter = 200

# Krotov step sizes
LAMBDA_VALUES = [5]

print(f"Parameters:")
print(f"  ω={omega}, κ={kappa_c:.4f}, Γ={Gamma:.4f}, ε₀={epsilon_0:.4f}")
print(f"  J=Γ/2={J:.4f},  T={T_target:.1f}  (Jt=7)")
print(f"  N_t={N_t}, dt={dt:.5f}")
print(f"  Oscillation period 2π/ω ≈ {2*np.pi/omega:.2f},  steps/period ≈ {2*np.pi/omega/dt:.0f}")

# Shape function
ton = toff = 0.005 * T_target

def S_of_t(t):
    if t < ton:
        return float(np.sin(np.pi * t / (2.0 * ton))**2)
    if t > T_target - toff:
        return float(np.sin(np.pi * (T_target - t) / (2.0 * toff))**2)
    return 1.0

S_array = np.array([S_of_t(t) for t in tlist])

#RK4 stepper
def rk4(z, t, step, rhs):
    k1 = rhs(z, t)
    k2 = rhs(z + (step/2)*k1, t + step/2)
    k3 = rhs(z + (step/2)*k2, t + step/2)
    k4 = rhs(z + step*k3,      t + step)
    return z + (step/6)*(k1 + 2*k2 + 2*k3 + k4)

# Forward RHS first moment equations
# z = [<c>, <b>]
def make_forward_rhs(A, B, Lc, Lb):
    def rhs(z, t, eps=0.0):
        c_m, b_m = z
        drive = np.exp(-1j * omega_L * t) * eps
        dc = -(Lc/2 + 1j*omega)*c_m - 1j*A*b_m - 1j*drive
        db = -(Lb/2 + 1j*omega)*b_m - 1j*B*c_m
        return np.array([dc, db])
    return rhs

# Adjoint RHS costate equations
# λ = [λ_c, λ_b], propagated backward
def make_adjoint_rhs(A, B, Lc, Lb):
    def rhs(lam, t):
        lam_c, lam_b = lam
        # dλ_c/dt = (Lc/2 - iω)λ_c - iB*λ_b
        # dλ_b/dt = -iA*λ_c + (Lb/2 - iω)λ_b
        dlam_c = (Lc/2 - 1j*omega)*lam_c - 1j*np.conj(B)*lam_b
        dlam_b = -1j*np.conj(A)*lam_c + (Lb/2 - 1j*omega)*lam_b
        return np.array([dlam_c, dlam_b])
    return rhs

# Full forward propagation store all states
def forward_propagate_all(eps_arr, fwd_rhs):
    states    = np.zeros((N_t, 2), dtype=complex)
    z         = np.array([0+0j, 0+0j])
    states[0] = z
    for k in range(N_t - 1):
        eps_mid = 0.5 * (eps_arr[k] + eps_arr[k+1])
        e_mid   = eps_mid   # capture value for lambda
        z       = rk4(z, tlist[k], dt,
                      lambda z, t: fwd_rhs(z, t, eps=e_mid))
        states[k+1] = z
    return states

# Backward propagation adjoint co-state
def backward_propagate(b_T, adj_rhs):
    b_T_norm    = abs(b_T)
    if b_T_norm < 1e-15:
        lam_bT = 0+0j
    else:
        lam_bT = omega * b_T / b_T_norm
    lam_states      = np.zeros((N_t, 2), dtype=complex)
    lam             = np.array([0+0j, lam_bT])
    lam_states[-1]  = lam
    for k in range(N_t - 1, 0, -1):
        lam             = rk4(lam, tlist[k], -dt, adj_rhs)
        lam_states[k-1] = lam
    return lam_states

# Full 5-moment ODE for plotting
def make_rhs_5moment(g, Lc, Lb, Gamma_val, mu_val, eps_func):
    A = g           + 1j*mu_val         *Gamma_val/2
    B = np.conj(g)  + 1j*np.conj(mu_val)*Gamma_val/2
    C = g           - 1j*mu_val         *Gamma_val/2
    D = np.conj(g)  - 1j*np.conj(mu_val)*Gamma_val/2

    def rhs(t, y):
        c_m, b_m, nc, nb, cb = y
        eps       = eps_func(t)
        drive_fwd = np.exp(-1j*omega_L*t)*eps
        drive_bwd = np.exp(+1j*omega_L*t)*eps

        dc  = -(Lc/2 + 1j*omega)*c_m - 1j*A*b_m - 1j*drive_fwd
        db  = -(Lb/2 + 1j*omega)*b_m - 1j*B*c_m
        dnc = -Lc*nc - 2*np.real(1j*A*cb) - 2*np.imag(drive_bwd*c_m)
        dnb = -Lb*nb + 2*np.real(1j*C*cb)
        dcb = -((Lc+Lb)/2)*cb - 1j*B*nc + 1j*D*nb + 1j*drive_bwd*b_m

        return [dc, db, dnc, dnb, dcb]
    return rhs

# Case definitions
cases = [
    {
        'name':      'Case 1: Γ=0, g real (no reservoir)',
        'label':     'case1',
        'g':         J,
        'Lambda_c':  kappa_c,
        'Lambda_b':  kappa_b,
        'Gamma_val': 0.0,
        'color':     'black',
        'lambda_values': [200],
    },
    {
        'name':      'Case 2: Γ≠0, g real (reciprocal)',
        'label':     'case2',
        'g':         J,
        'Lambda_c':  Lambda_c,
        'Lambda_b':  Lambda_b,
        'Gamma_val': Gamma,
        'color':     'blue',
        'lambda_values': LAMBDA_VALUES,
    },
    {
        'name':      'Case 3: g=+iΓ/2 (nonreciprocal)',
        'label':     'case3',
        'g':         +1j*Gamma/2,
        'Lambda_c':  Lambda_c,
        'Lambda_b':  Lambda_b,
        'Gamma_val': Gamma,
        'color':     'green',
        'lambda_values': LAMBDA_VALUES,
    },
]

all_results = {}

for case in cases:

    g_case = case['g']
    Lc     = case['Lambda_c']
    Lb     = case['Lambda_b']
    Gval   = case['Gamma_val']

    # Coupling coefficients
    A = g_case + 1j*mu*Gval/2
    B = np.conj(g_case) + 1j*np.conj(mu)*Gval/2

    fwd_rhs = make_forward_rhs(A, B, Lc, Lb)
    adj_rhs = make_adjoint_rhs(A, B, Lc, Lb)

    print(f"\n{'='*60}")
    print(f"  {case['name']}")
    print(f"  A = {A:.4f},  B = {B:.4f}")
    print(f"{'='*60}")

    # Guess pulse and its performance
    eps_guess = S_array * epsilon_0
    fwd_guess = forward_propagate_all(eps_guess, fwd_rhs)
    F_guess   = omega * abs(fwd_guess[-1, 1])**2
    print(f"  Guess pulse  E_B(T)/ω = {F_guess:.4f}")

    for lambda_a in case['lambda_values']:

        print(f"\n  Krotov optimisation   lambda_a = {lambda_a}")
        print(f"  {'─'*45}")

        eps_array = eps_guess.copy()

        # Initial forward pass to get b_T for first backward boundary
        fwd_states = forward_propagate_all(eps_array, fwd_rhs)
        b_T        = fwd_states[-1, 1]   # <b(T)>

        F_prev           = 0.0
        no_improve_count = 0
        tol              = 1e-8
        patience         = 30
        converged_at     = N_iter

        for iteration in range(N_iter):

            # Step 1: backward pass under current eps 
            bwd = backward_propagate(b_T, adj_rhs)

            #  Step 2: sequential forward pass with Krotov updates 
            new_eps = eps_array.copy()
            z       = np.array([0+0j, 0+0j])

            for k in range(N_t - 1):
                lam_c = bwd[k, 0]

                # Gradient of E_B(T) w.r.t. eps(t_k)
                grad = np.imag(
                    np.conj(lam_c) * np.exp(-1j * omega_L * tlist[k])
                )

                new_eps[k] = eps_guess[k] + (S_array[k] / lambda_a) * grad

                eps_mid = 0.5 * (new_eps[k] + eps_array[k+1])
                e_mid   = eps_mid
                z       = rk4(z, tlist[k], dt,
                              lambda z, t: fwd_rhs(z, t, eps=e_mid))

            # Last time point
            lam_c_last = bwd[-1, 0]
            grad_last  = np.imag(
                np.conj(lam_c_last) * np.exp(-1j * omega_L * tlist[-1])
            )
            new_eps[-1] = eps_guess[-1] + (S_array[-1] / lambda_a) * grad_last

            # Enforce S(t) boundary shape
            new_eps = new_eps * S_array

            # Fluence normalisation
            # Since the system is linaer, so Krotov without a
            # constraint will just increase amplitude indefinitely.
            fluence_guess = np.trapz(eps_guess**2, tlist)
            fluence_new   = np.trapz(new_eps**2,   tlist)
            if fluence_new > 1e-15:
                new_eps = new_eps * np.sqrt(fluence_guess / fluence_new)

            # Fidelity from sequential forward pass final state
            b_T_new = z[1]
            F       = omega * abs(b_T_new)**2

            eps_array = new_eps
            b_T       = b_T_new   # update boundary condition for next iteration

            if (iteration + 1) % 20 == 0 or iteration == 0:
                print(f"  iter {iteration+1:4d}   E_B(T)/ω = {F:.4f}")

            if abs(F - F_prev) < tol:
                no_improve_count += 1
                if no_improve_count >= patience:
                    converged_at = iteration + 1
                    print(f"  Converged at iteration {converged_at}")
                    break
            else:
                no_improve_count = 0
            F_prev = F

        eps_opt = eps_array.copy()
        print(f"\n  Final E_B(T)/ω:  guess = {F_guess:.4f}  →  optimised = {F:.4f}"
              f"  (improvement: {(F/F_guess - 1)*100:.1f}%)")

        # Full 5-moment ODE for plotting
        def make_interp_pulse(arr):
            def _f(t):
                return float(np.interp(t, tlist, arr))
            return _f

        def make_shaped_const(val):
            def _f(t):
                return val * S_of_t(t)
            return _f

        t_plot = np.linspace(0, T_target, 2000)
        y0_5   = [0+0j, 0+0j, 0+0j, 0+0j, 0+0j]

        sol_guess = solve_ivp(
            make_rhs_5moment(g_case, Lc, Lb, Gval, mu,
                             make_shaped_const(epsilon_0)),
            (0, T_target), y0_5, t_eval=t_plot,
            method='RK45', rtol=1e-9, atol=1e-12
        )
        sol_opt = solve_ivp(
            make_rhs_5moment(g_case, Lc, Lb, Gval, mu,
                             make_interp_pulse(eps_opt)),
            (0, T_target), y0_5, t_eval=t_plot,
            method='RK45', rtol=1e-9, atol=1e-12
        )

        # total: coherent + quantum fluctuations
        E_B_guess = omega * np.real(sol_guess.y[3])
        E_B_opt   = omega * np.real(sol_opt.y[3])
        Jt_plot   = J * t_plot

        # Store best result per case (highest E_B_opt at T)
        if case['label'] not in all_results or F > all_results[case['label']]['F_opt']:
            all_results[case['label']] = {
                'name':      case['name'],
                'color':     case['color'],
                'eps_guess': eps_guess,
                'eps_opt':   eps_opt,
                'E_B_guess': E_B_guess,
                'E_B_opt':   E_B_opt,
                'Jt_plot':   Jt_plot,
                'F_guess':   F_guess,
                'F_opt':     F,
                'lambda_a':  lambda_a,
            }

        # Plot 1: Pulse shapes
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(J*tlist, eps_guess, color='black',       linestyle='--',
                linewidth=1.5, label=f'Guess  (ε₀={epsilon_0:.2f}ω)')
        ax.plot(J*tlist, eps_opt,   color=case['color'], linestyle='-',
                linewidth=1.5, label='Krotov optimised')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.set_xlabel(r'$Jt$', fontsize=12)
        ax.set_ylabel(r'$\varepsilon(t)$', fontsize=12)
        ax.set_title(f'{case["name"]}\nλ_a={lambda_a}', fontsize=9)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 7)
        plt.tight_layout()
        plt.savefig(f'./graphs/pulse_{case["label"]}_lambda{lambda_a}.png', dpi=200)
        plt.close()

        # Plot 2: Battery energy vs time
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(Jt_plot, E_B_guess, color='black',       linestyle='--',
                linewidth=1.5, label=f'Guess   E_B(T)={F_guess:.2f}ω')
        ax.plot(Jt_plot, E_B_opt,   color=case['color'], linestyle='-',
                linewidth=1.5, label=f'Optimised  E_B(T)={F:.2f}ω')
        ax.axvline(7.0, color='grey', linestyle=':', linewidth=1.0,
                   label='Target  Jt=7')
        ax.set_xlabel(r'$Jt$', fontsize=12)
        ax.set_ylabel(r'$E_B\,/\,\omega$', fontsize=12)
        ax.set_title(f'Battery energy {case["name"]}\nλ_a={lambda_a}',
                     fontsize=9)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 7)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(f'./graphs/energy_{case["label"]}_lambda{lambda_a}.png', dpi=200)
        plt.close()

        print(f"  Saved: pulse_{case['label']}_lambda{lambda_a}.png  |  "
              f"energy_{case['label']}_lambda{lambda_a}.png")

# Comparison plots all three cases on one figure

# Battery energy comparison (guess vs optimised)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

for label, res in all_results.items():
    axes[0].plot(res['Jt_plot'], res['E_B_guess'],
                 color=res['color'], linestyle='--', linewidth=1.8,
                 label=res['name'].split(':')[0])
    axes[1].plot(res['Jt_plot'], res['E_B_opt'],
                 color=res['color'], linestyle='-',  linewidth=1.8,
                 label=f"{res['name'].split(':')[0]}  E_B={res['F_opt']:.1f}ω  (λ={res['lambda_a']})")

for ax, title in zip(axes, ['Guess pulses (constant ε₀)',
                              'Krotov optimised pulses']):
    ax.set_xlabel(r'$Jt$', fontsize=12)
    ax.set_ylabel(r'$E_B\,/\,\omega$', fontsize=12)
    ax.set_title(f'Battery energy {title}', fontsize=10)
    ax.axvline(7.0, color='grey', linestyle=':', linewidth=1.0)
    ax.set_xlim(0, 7)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('./graphs/comparison_energy_all_cases.png', dpi=200)
plt.close()
print("\nSaved: ./graphs/comparison_energy_all_cases.png")

# Pulse shape comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))

for label, res in all_results.items():
    name_short = res['name'].split(':')[0]
    axes[0].plot(J*tlist, res['eps_guess'],
                 color=res['color'], linestyle='--', linewidth=1.5,
                 label=name_short)
    axes[1].plot(J*tlist, res['eps_opt'],
                 color=res['color'], linestyle='-',  linewidth=1.5,
                 label=name_short)

for ax, title in zip(axes, ['Guess pulses', 'Krotov optimised pulses']):
    ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
    ax.set_xlabel(r'$Jt$', fontsize=12)
    ax.set_ylabel(r'$\varepsilon(t)$', fontsize=12)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 7)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('./graphs/comparison_pulses_all_cases.png', dpi=200)
plt.close()
print("Saved: ./graphs/comparison_pulses_all_cases.png")

print("\nAll cases complete.")