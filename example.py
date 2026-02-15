import numpy as np
import qutip
import krotov

omega = 1.0  # Qubit frequency
T = 5.0      # Total pulse duration
nt = 500     # Number of time steps
tlist = np.linspace(0, T, nt)

# Drift Hamiltonian (H0): Energy of the qubit
H0 = -0.5 * omega * qutip.sigmaz()

# Control Hamiltonian (H1): How the laser/microwave interacts
H1 = qutip.sigmax()

# The initial guess pulse (a simple constant or sine wave)
def guess_pulse(t, args):
    return 0.1


# The Hamiltonian list format required by QuTiP/Krotov
objectives = [
    krotov.Objective(
        initial_state=qutip.basis(2, 0),  # Start at |0>
        target=qutip.basis(2, 1),         # Aim for |1>
        H=[H0, [H1, guess_pulse]]         # H = H0 + epsilon(t)*H1
    )
]

# S(t) ensures the pulse ramps up and down smoothly


def shape_function(t):
    return krotov.shapes.flattop(t, t_start=0, t_stop=T, t_rise=0.5, t_fall=0.5, func='blackman')


# Pulse options: 'lambda_a' is the step size. Smaller = more stable but slower.
pulse_options = {
    guess_pulse: dict(lambda_a=1.0, update_shape=shape_function)
}


opt_result = krotov.optimize_pulses(
    objectives,
    pulse_options=pulse_options,
    tlist=tlist,
    iter_stop=10,         # Number of iterations (forward/backward passes)
    propagator=krotov.propagators.expm,  # How to solve the Schrodinger eq.
    chi_constructor=krotov.functionals.chis_re,  # Constructor for chi functional
)


def _maybe_get_final_JT(result):
    # Different krotov versions/logging setups expose different fields
    for key in ("J_T", "J_T_vals"):
        if hasattr(result, key):
            val = getattr(result, key)
            if key == "J_T":
                return val
            if isinstance(val, (list, tuple)) and len(val) > 0:
                return val[-1]
    return None


def _format_number_or_array(val):
    arr = np.asarray(val)
    if arr.shape == ():
        x = arr.item()
        if isinstance(x, complex) or np.iscomplexobj(arr):
            return f"{x.real:.6g}{x.imag:+.6g}j"
        return f"{float(x):.6g}"
    if arr.size == 1:
        x = arr.reshape(()).item()
        if isinstance(x, complex) or np.iscomplexobj(arr):
            return f"{x.real:.6g}{x.imag:+.6g}j"
        return f"{float(x):.6g}"
    preview_n = min(5, arr.size)
    flat = arr.ravel()
    if np.iscomplexobj(arr):
        abs_flat = np.abs(flat)
        return (
            f"array(shape={arr.shape}, |min|={float(abs_flat.min()):.6g}, |max|={float(abs_flat.max()):.6g}, "
            f"first={flat[:preview_n]})"
        )
    return (
        f"array(shape={arr.shape}, min={float(flat.min()):.6g}, max={float(flat.max()):.6g}, "
        f"first={flat[:preview_n]})"
    )


print("\n=== Krotov optimization result ===")
print(opt_result.message)
print(
    f"iterations recorded: {len(opt_result.iters) - 1} (including guess as iter 0)")
final_tau = opt_result.tau_vals[-1] if getattr(
    opt_result, "tau_vals", None) else None
if final_tau is not None:
    print(f"final tau: {_format_number_or_array(final_tau)}")
final_JT = _maybe_get_final_JT(opt_result)
if final_JT is not None and isinstance(final_JT, (float, int, np.floating, np.integer)):
    print(f"final J_T (if available): {final_JT:.6g}")

# Show the optimized control (and guess) in a useful way
guess_eps = np.array(opt_result.guess_controls[0])
opt_eps = np.array(opt_result.optimized_controls[0])
print(f"control array shape: {opt_eps.shape}")
print(
    f"guess control:     min={guess_eps.min():.6g}, max={guess_eps.max():.6g}")
print(f"optimized control: min={opt_eps.min():.6g}, max={opt_eps.max():.6g}")
preview_n = min(10, opt_eps.size)
print(f"optimized control first {preview_n} points:")
print(opt_eps[:preview_n])

# Plot if matplotlib is available; otherwise the printed summary above is the result.
try:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(tlist, guess_eps, label="guess")
    plt.plot(tlist, opt_eps, label="optimized")
    plt.xlabel("t")
    plt.ylabel("control amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()
except Exception:
    pass
