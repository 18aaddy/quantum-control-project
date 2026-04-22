import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
omega   = 1.0
omega_L = omega             # resonant drive
kappa_c = 0.003 * omega
kappa_b = 0.003 * omega
Gamma   = 0.04  * omega
epsilon = 0.1   * omega

# Symmetric reservoir: p_c = p_b = 1
# mu = -p_b * p_a*
mu = -1.0 + 0j

# Derived: Lambda_i = Gamma_i + kappa_i
# Gamma_i = Gamma * |p_i|^2 = Gamma (since |p_i|=1)
Lambda_c = Gamma + kappa_c
Lambda_b = Gamma + kappa_b

# J = |g| = Gamma/2 for time axis scaling
J = Gamma / 2.0

# Time axis
t_end = 15.0 / J
N_t   = 3000
tlist = np.linspace(0.0, t_end, N_t)
Jt    = J * tlist

print(f"Parameters:")
print(f"  omega   = {omega},  omega_L = {omega_L}")
print(f"  kappa   = {kappa_c:.4f}")
print(f"  Gamma   = {Gamma:.4f}")
print(f"  epsilon = {epsilon:.4f}")
print(f"  mu      = {mu}")
print(f"  Lambda  = {Lambda_c:.4f}")
print(f"  J = Gamma/2 = {J:.4f}")


def make_rhs(g, Lambda_c, Lambda_b, Gamma_val, mu_val, omega_val, omega_L_val, eps):
    def rhs(t, y):
        c_m  = y[0]   # <c>
        b_m  = y[1]   # <b>
        nc   = y[2]   # <c_dagc>
        nb   = y[3]   # <b_dagb>
        cb   = y[4]   # <c_dagb>

        # Time-dependent drive phase factors
        drive_fwd = np.exp(-1j * omega_L_val * t) * eps
        drive_bwd = np.exp(+1j * omega_L_val * t) * eps

        # Coupling combinations appearing in equations
        # charger:  (g + i*mu*Gamma/2)
        A = g + 1j * mu_val * Gamma_val / 2.0
        # battery:  (g* + i*mu**Gamma/2)  [mu conjugated]
        B = np.conj(g) + 1j * np.conj(mu_val) * Gamma_val / 2.0
        # battery n and battery term:  (g - i*mu*Gamma/2)
        C = g - 1j * mu_val * Gamma_val / 2.0
        # cross-correlator battery: (g* - i*mu**Gamma/2)
        D = np.conj(g) - 1j * np.conj(mu_val) * Gamma_val / 2.0

        # first moments 
        dc = -(Lambda_c / 2.0 + 1j * omega_val) * c_m \
             - 1j * A * b_m \
             - 1j * drive_fwd

        db = -(Lambda_b / 2.0 + 1j * omega_val) * b_m \
             - 1j * B * c_m

        # second moments (diagonal)
        dnc = -Lambda_c * nc \
              - 2.0 * np.real(1j * A * cb) \
              - 2.0 * np.imag(drive_bwd * c_m)

        dnb = -Lambda_b * nb \
              + 2.0 * np.real(1j * C * cb)

        # cross-correlator
        dcb = -((Lambda_c + Lambda_b) / 2.0) * cb \
              - 1j * B * nc \
              + 1j * D * nb \
              + 1j * drive_bwd * b_m

        return [dc, db, dnc, dnb, dcb]
    return rhs

# Initial conditions: all moments zero
y0 = [0+0j, 0+0j, 0+0j, 0+0j, 0+0j]

# Case 1: Gamma=0, g real — no shared reservoir
print("\nSolving Case 1: Gamma=0, g=J real (no reservoir) ...")
sol1 = solve_ivp(
    make_rhs(J, kappa_c, kappa_b, 0.0, mu, omega, omega_L, epsilon),
    (0, t_end), y0, t_eval=tlist,
    method='RK45', rtol=1e-9, atol=1e-12
)
E_B1 = omega * np.real(sol1.y[3])
E_A1 = omega * np.real(sol1.y[2])

# Case 2: Gamma!=0, g real — shared reservoir, reciprocal
print("Solving Case 2: Gamma!=0, g=J real (reciprocal) ...")
sol2 = solve_ivp(
    make_rhs(J, Lambda_c, Lambda_b, Gamma, mu, omega, omega_L, epsilon),
    (0, t_end), y0, t_eval=tlist,
    method='RK45', rtol=1e-9, atol=1e-12
)
E_B2 = omega * np.real(sol2.y[3])
E_A2 = omega * np.real(sol2.y[2])

# Case 3: Nonreciprocal condition
print("Solving Case 3: g=+i*Gamma/2 (nonreciprocal) ...")
g_nr = +1j * Gamma / 2.0
sol3 = solve_ivp(
    make_rhs(g_nr, Lambda_c, Lambda_b, Gamma, mu, omega, omega_L, epsilon),
    (0, t_end), y0, t_eval=tlist,
    method='RK45', rtol=1e-9, atol=1e-12
)
E_B3 = omega * np.real(sol3.y[3])
E_A3 = omega * np.real(sol3.y[2])

# Analytical steady-state values
Lambda   = Lambda_c   # symmetric case
E_B3_ss  = 16 * omega * Gamma**2 * epsilon**2 / Lambda**4
E_A3_ss  =  4 * omega * epsilon**2 / Lambda**2
Cd       = 4 * Gamma**2 / Lambda**2

print(f"\nAnalytical steady states (Case 3):")
print(f"  E_B^nr(inf) = {E_B3_ss:.4f}  |  numerical: {E_B3[-1]:.4f}")
print(f"  E_A^nr(inf) = {E_A3_ss:.4f}  |  numerical: {E_A3[-1]:.4f}")
print(f"  C_d = {Cd:.4f}  (should be > 1 for battery > charger)")
print(f"\nFig 3 comparison (Case2 reciprocal vs Case3 nonreciprocal):")
print(f"  E_B reciprocal ss    = {E_B2[-1]:.4f}")
print(f"  E_B nonreciprocal ss = {E_B3[-1]:.4f}")
print(f"  eta_BB(inf) = {E_B3[-1]/E_B2[-1]:.3f}  (paper: ~4)")

# Plots

# E_A^nr and E_B^nr vs Jt (Case 3
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(Jt, E_A3, color='steelblue',  linewidth=2.0, label=r'$E_A^{nr}$ (charger)')
ax.plot(Jt, E_B3, color='darkorange', linewidth=2.0, label=r'$E_B^{nr}$ (battery)')
ax.axhline(E_A3_ss, color='steelblue',  linestyle='--', linewidth=1.0, alpha=0.5)
ax.axhline(E_B3_ss, color='darkorange', linestyle='--', linewidth=1.0, alpha=0.5,
           label='analytical $\infty$')
ax.set_xlabel(r'$Jt$', fontsize=13)
ax.set_ylabel(r'Energy / $\omega$', fontsize=12)
ax.set_title('Fig. 2(a) — Nonreciprocal regime', fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(0, Jt[-1])
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('./graphs/fig2a_nonreciprocal.png', dpi=200)
plt.close()
print("\nSaved: ./graphs/fig2a_nonreciprocal.png")

# Fig 2(b): eta_AB^nr = E_B^nr / E_A^nr
eta_AB = np.where(E_A3 > 1e-15, E_B3 / E_A3, 0.0)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(Jt, eta_AB, color='black', linewidth=2.0, label=r'$\eta_{AB}^{nr}(t)$')
ax.axhline(Cd, color='red', linestyle='--', linewidth=1.5,
           label=r'$C_d = 4\Gamma^2/\Lambda^2 = $' + f'{Cd:.2f}')
ax.axhline(1.0, color='grey', linestyle=':', linewidth=1.0)
ax.set_xlabel(r'$Jt$', fontsize=13)
ax.set_ylabel(r'$\eta_{AB}^{nr} = E_B^{nr}/E_A^{nr}$', fontsize=12)
ax.set_title('Fig. 2(b) — Battery/charger ratio (nonreciprocal)', fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(0, Jt[-1])
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('./graphs/fig2b_eta_AB.png', dpi=200)
plt.close()
print("Saved: ./graphs/fig2b_eta_AB.png")

# Fig 3(a): Reciprocal (Case 2) vs Nonreciprocal (Case 3)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(Jt, E_B2, color='black', linewidth=2.0,
        label=r'$E_B$ (reciprocal: $\Gamma\neq0$, $g$ real)')
ax.plot(Jt, E_B3, color='green', linewidth=2.0,
        label=r'$E_B^{nr}$ (nonreciprocal: $g=+i\Gamma/2$)')
ax.set_xlabel(r'$Jt$', fontsize=13)
ax.set_ylabel(r'Battery energy / $\omega$', fontsize=12)
ax.set_title('Fig. 3(a) — Reciprocal vs Nonreciprocal', fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(0, Jt[-1])
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('./graphs/fig3a_comparison.png', dpi=200)
plt.close()
print("Saved: ./graphs/fig3a_comparison.png")

# Fig 3(b): eta_BB = E_B^nr / E_B 
eta_BB = np.where(E_B2 > 1e-15, E_B3 / E_B2, 0.0)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(Jt, eta_BB, color='black', linewidth=2.0,
        label=r'$\eta_{BB}(t) = E_B^{nr}/E_B$')
ax.axhline(E_B3[-1]/E_B2[-1], color='red', linestyle='--', linewidth=1.5,
           label=r'$\eta_{BB}(\infty) = $' + f'{E_B3[-1]/E_B2[-1]:.2f}')
ax.axhline(1.0, color='grey', linestyle=':', linewidth=1.0)
ax.set_xlabel(r'$Jt$', fontsize=13)
ax.set_ylabel(r'$\eta_{BB}(t)$', fontsize=12)
ax.set_title('Fig. 3(b) — Nonreciprocal advantage', fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(0, Jt[-1])
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('./graphs/fig3b_eta_BB.png', dpi=200)
plt.close()
print("Saved: ./graphs/fig3b_eta_BB.png")

#  All three cases 
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(Jt, E_B1, color='black', linewidth=2.0, linestyle='-',
        label=r'Case 1: $\Gamma=0$, $g$ real (no reservoir)')
ax.plot(Jt, E_B2, color='blue',  linewidth=2.0, linestyle='-',
        label=r'Case 2: $\Gamma\neq0$, $g$ real (reciprocal)')
ax.plot(Jt, E_B3, color='green', linewidth=2.0, linestyle='-',
        label=r'Case 3: $g=+i\Gamma/2$ (nonreciprocal)')
ax.set_xlabel(r'$Jt$', fontsize=13)
ax.set_ylabel(r'Battery energy $E_B$ / $\omega$', fontsize=12)
ax.set_title('Battery energy — all three cases', fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(0, Jt[-1])
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('./graphs/all_cases.png', dpi=200)
plt.close()
print("Saved: ./graphs/all_cases.png")