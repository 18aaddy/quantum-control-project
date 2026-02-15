import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import krotov

# 1. Physical Parameters from the Paper [cite: 311]
omega = 1.0          # Qubit frequency
g = 0.2 * omega      # Coupling strength
gamma = 0.05 * omega # Dissipation rate 
mu = 0.5 * omega     # Control coupling strength
F = 0.5 * omega      # Oscillatory field amplitude
tau = np.pi / g      # Final time [cite: 323]
nt = 500             # Time steps
tlist = np.linspace(0, tau, nt)

# 2. Operators & Hamiltonian Construction [cite: 244, 246]
# Basis: |charger, battery>
sx_A = qt.tensor(qt.sigmax(), qt.qeye(2))
sz_A = qt.tensor(qt.sigmaz(), qt.qeye(2))
sz_B = qt.tensor(qt.qeye(2), qt.sigmaz())
sp_A = qt.tensor(qt.sigmap(), qt.qeye(2))
sm_A = qt.tensor(qt.sigmam(), qt.qeye(2))
sp_B = qt.tensor(qt.qeye(2), qt.sigmap())
sm_B = qt.tensor(qt.qeye(2), qt.sigmam())

# Free Hamiltonians and Interaction
H_A = 0.5 * omega * (-sz_A + qt.tensor(qt.qeye(2), qt.qeye(2)))
H_B = 0.5 * omega * (-sz_B + qt.tensor(qt.qeye(2), qt.qeye(2)))
H_AB = g * (sp_A * sm_B + sm_A * sp_B)
H0 = H_A + H_B + H_AB
Hc = -mu * sx_A  # Control field couples to charger [cite: 244]

# Dissipator (Local approach: acts only on charger) [cite: 90, 250]
L = [np.sqrt(gamma) * sm_A] 

# 3. Define Metric Functions (Energy & Ergotropy) [cite: 145, 146]
def get_metrics(states):
    energies = []
    ergotropies = []
    h_b_local = 0.5 * omega * (-qt.sigmaz() + qt.qeye(2))
    for state in states:
        rho_B = state.ptrace(1) # Focus only on the battery [cite: 87]
        E = qt.expect(h_b_local, rho_B)
        # Ergotropy for single qubit [cite: 146]
        evals = np.sort(rho_B.eigenenergies())[::-1] 
        E_passive = evals[1] * omega # Passive state energy
        energies.append(E / omega)
        ergotropies.append(max(0, (E - E_passive) / omega))
    return energies, ergotropies

# 4. Scenario A: Non-Optimized Oscillatory Drive [cite: 263]
def osc_pulse(t, args):
    return (F/mu) * np.cos(omega * t)

res_osc = qt.mesolve([H0, [Hc, osc_pulse]], qt.tensor(qt.basis(2,0), qt.basis(2,0)), tlist, c_ops=L)
e_osc, erg_osc = get_metrics(res_osc.states)

# 5. Scenario B: Krotov Optimization [cite: 53, 163]
initial_state = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
target_state = qt.tensor(qt.qeye(2), qt.basis(2, 1)) # Excited battery [cite: 149]

def guess_pulse(t, args):
    return 0.5 # κ = 0.5 [cite: 308]

def shape_func(t):
    return krotov.shapes.flattop(t, t_start=0, t_stop=tau, t_rise=0.005*tau, t_fall=0.005*tau)

objectives = [krotov.Objective(initial_state=initial_state, target_state=target_state, H=[H0, [Hc, guess_pulse]], c_ops=L)]
pulse_options = {guess_pulse: dict(lambda_a=10.0, shape=shape_func)}

opt_result = krotov.optimize_pulses(objectives, pulse_options, tlist, iter_stop=15, propagator=krotov.propagators.expm)
e_opt, erg_opt = get_metrics(opt_result.optimized_objectives[0].mesolve(tlist, c_ops=L).states)

# 6. Plotting Results [cite: 306, 307]
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot (a): Energy and Ergotropy
ax[0].plot(g*tlist, e_opt, 'g-', label='E (Optimized)')
ax[0].plot(g*tlist, erg_opt, 'g-.', label='Erg (Optimized)')
ax[0].plot(g*tlist, e_osc, 'k-', label='E (Sinusoidal)')
ax[0].plot(g*tlist, erg_osc, 'k-.', label='Erg (Sinusoidal)')
ax[0].set_xlabel('gt'); ax[0].set_ylabel('E/w, Erg/w'); ax[0].legend()
ax[0].set_title('Energy & Ergotropy Evolution')

# Plot (b): Field Pulses
ax[1].plot(g*tlist, opt_result.optimized_controls[0], 'g-', label='Optimized Pulse')
ax[1].plot(g*tlist, [osc_pulse(t, None) for t in tlist], 'k--', label='Sinusoidal Pulse')
ax[1].set_xlabel('gt'); ax[1].set_ylabel('epsilon(t)'); ax[1].legend()
ax[1].set_title('Field Pulses')

plt.tight_layout()
plt.savefig('quantum_battery_simulation.png')
print("Simulation complete. Image saved as 'quantum_battery_simulation.png'.")