"""
common.py -- shared ground truth + data generation for all three demos.

Everything in this project revolves around ONE equation, the 1D viscous Burgers
equation on a periodic domain x in [-1, 1):

    du/dt + u * du/dx = nu * d2u/dx2

It is the simplest PDE that is genuinely interesting: the nonlinear term
u*du/dx steepens the wave into a shock, the viscous term nu*d2u/dx2 fights back
and keeps it smooth. Everything you learn here transfers to Navier-Stokes.

This file has no torch in it. It is pure numpy "classical numerics", and it
plays two very different roles:

  * For the PINN (script 01) it is the REFERENCE SOLUTION -- the thing we grade
    the network against. The PINN never sees it during training.
  * For the operator learning demos (scripts 02, 03) it is the DATA GENERATOR.
    We solve the PDE a thousand times with a thousand different initial
    conditions, and the network learns the input -> output map from those pairs.

That difference is the whole point of this project. Read the README.
"""

import os
import numpy as np

# ----------------------------------------------------------------------------
# Problem constants -- change these in one place and every script follows.
# ----------------------------------------------------------------------------

NU = 0.02          # viscosity. The classic Raissi et al. PINN paper uses
                   # 0.01/pi ~= 0.0032, which makes a much sharper shock. It is
                   # a harder and slower demo (you need N=1024 for the reference
                   # solution and a longer PINN training run). Try it once
                   # everything below works.
L = 2.0            # domain length, x in [-1, 1)
T_FINAL = 1.0      # we always integrate to t = 1
NX = 256           # grid points used for the operator-learning datasets

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "data")
FIG_DIR = os.path.join(_HERE, "figures")


def grid(n=NX):
    """The periodic spatial grid. Note the endpoint x=+1 is EXCLUDED: on a
    periodic domain x=+1 is the same point as x=-1, and including both would
    make the FFT wrong."""
    return -1.0 + L * np.arange(n) / n


# ----------------------------------------------------------------------------
# 1. The classical solver: pseudo-spectral in space, integrating-factor RK4 in
#    time.
# ----------------------------------------------------------------------------
#
# Why spectral? On a periodic domain, derivatives are exact and free in Fourier
# space: if u(x) = sum_k u_hat_k exp(i k x), then du/dx has coefficients
# i*k*u_hat_k. So we hop to Fourier space, multiply by i*k, hop back. The error
# decays faster than any power of 1/N for smooth u ("spectral accuracy").
#
# In Fourier space the PDE becomes an ODE for each mode:
#
#     d(u_hat)/dt = -nu*k^2 * u_hat   +   Nl(u_hat)
#                   \-----------/         \-------/
#                    stiff, linear         nonlinear
#                                          = -0.5 * i*k * FFT(u^2)
#
# The stiff linear part is the problem: -nu*k^2 is huge for large k, so a plain
# explicit RK4 would need a microscopic dt. The integrating-factor trick solves
# the linear part EXACTLY (it is just multiplication by exp(-nu*k^2*dt)) and
# runs RK4 only on the nonlinear part. Then dt is limited only by the mild
# advective CFL condition dt < dx / max|u|.
#
# We also write u*du/dx as 0.5*d(u^2)/dx, which conserves mass exactly and is
# cheaper (one FFT instead of two).


def burgers_solve(u0, nu=NU, T=T_FINAL, dt=1e-3, n_save=2):
    """Solve Burgers forward in time. Fully batched over initial conditions.

    Args:
        u0: (N,) or (B, N) real array of initial conditions on grid(N).
        n_save: how many time snapshots to return, including t=0 and t=T.
    Returns:
        t: (n_save,) times
        u: (B, n_save, N) solution snapshots
    """
    u0 = np.atleast_2d(np.asarray(u0, dtype=np.float64))
    B, N = u0.shape

    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)      # wavenumbers
    # 2/3 dealiasing: squaring u doubles its bandwidth, and the top third of
    # modes would otherwise fold back as garbage ("aliasing"). Zero them.
    mask = np.abs(k) < (2.0 / 3.0) * np.abs(k).max()

    n_steps = int(round(T / dt))
    dt = T / n_steps
    save_every = max(1, n_steps // (n_save - 1)) if n_save > 1 else n_steps

    E = np.exp(-nu * k**2 * dt / 2.0)                  # half-step of diffusion
    E2 = E**2                                          # full step
    g = -0.5j * dt * k * mask                          # dt * (nonlinear op)

    def Nl(v):
        """dt * nonlinear term, computed pseudo-spectrally: square in physical
        space, differentiate in Fourier space."""
        u = np.real(np.fft.ifft(v, axis=-1))
        return g * np.fft.fft(u * u, axis=-1)

    v = np.fft.fft(u0, axis=-1)
    out = [u0.copy()]
    ts = [0.0]

    for step in range(1, n_steps + 1):
        # Integrating-factor RK4 (the classic Trefethen "p27" scheme).
        a = Nl(v)
        b = Nl(E * (v + a / 2))
        c = Nl(E * v + b / 2)
        d = Nl(E2 * v + E * c)
        v = E2 * v + (E2 * a + 2 * E * (b + c) + d) / 6.0
        if step % save_every == 0 or step == n_steps:
            out.append(np.real(np.fft.ifft(v, axis=-1)))
            ts.append(step * dt)

    u = np.stack(out, axis=1)                          # (B, n_save, N)
    return np.array(ts), u


# ----------------------------------------------------------------------------
# 2. Random initial conditions: a Gaussian random field.
# ----------------------------------------------------------------------------
#
# To LEARN an operator we need a probability distribution over inputs -- an
# operator is only defined relative to the space of functions you feed it. The
# standard choice is a Gaussian random field with a decaying power spectrum:
#
#     u0_hat_k ~ N(0, 1) * (k^2 + tau^2)^(-alpha/2)
#
# Bigger alpha => faster spectral decay => smoother functions. This is exactly
# the sampler used in the FNO and DeepONet papers.


def sample_initial_conditions(n, N=NX, alpha=3.0, tau=4.0, seed=0):
    """Draw n smooth, periodic, zero-mean random functions on grid(N)."""
    rng = np.random.default_rng(seed)
    kk = np.fft.fftfreq(N, d=1.0 / N)                  # integer wavenumbers
    spectrum = (np.abs(kk) ** 2 + tau**2) ** (-alpha / 2.0)
    spectrum[0] = 0.0                                  # kill the mean
    xi = rng.normal(size=(n, N)) + 1j * rng.normal(size=(n, N))
    f = np.fft.ifft(xi * spectrum, axis=-1).real
    f /= np.abs(f).max(axis=-1, keepdims=True)         # normalise to max|u|=1
    f *= rng.uniform(0.6, 1.2, size=(n, 1))            # then vary the amplitude
    return f


# ----------------------------------------------------------------------------
# 3. The operator-learning dataset: u0  ->  u(., T=1)
# ----------------------------------------------------------------------------


def make_operator_dataset(n_train=1000, n_test=200, N=NX, dt=1e-3, seed=0,
                          cache=True):
    """Generate (or load) input/output pairs for the solution operator

        G : u0(.)  |-->  u(., t=1)

    This is the object scripts 02 and 03 try to approximate with a network.
    Generating it takes ~30s the first time and is then cached to data/.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(
        CACHE_DIR, f"burgers_op_n{n_train}_{n_test}_N{N}_nu{NU}_s{seed}.npz")
    if cache and os.path.exists(path):
        d = np.load(path)
        return d["a_train"], d["u_train"], d["a_test"], d["u_test"], d["x"]

    n = n_train + n_test
    print(f"[data] solving Burgers for {n} initial conditions "
          f"(N={N}, nu={NU}, T={T_FINAL})...")
    a = sample_initial_conditions(n, N=N, seed=seed)
    _, sol = burgers_solve(a, dt=dt, n_save=2)
    u = sol[:, -1, :]                                  # keep only t = T

    a_train, u_train = a[:n_train], u[:n_train]
    a_test, u_test = a[n_train:], u[n_train:]
    x = grid(N)
    if cache:
        np.savez_compressed(path, a_train=a_train, u_train=u_train,
                            a_test=a_test, u_test=u_test, x=x)
        print(f"[data] cached -> {os.path.relpath(path, _HERE)}")
    return a_train, u_train, a_test, u_test, x


# ----------------------------------------------------------------------------
# 4. Small helpers
# ----------------------------------------------------------------------------


def rel_l2(pred, true, axis=-1):
    """Relative L2 error, the standard metric in this literature.
    0.03 means '3% error', and is roughly the level a good FNO reaches here."""
    num = np.linalg.norm(pred - true, axis=axis)
    den = np.linalg.norm(true, axis=axis)
    return float(np.mean(num / den))


def savefig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"[plot] saved -> {os.path.relpath(path, _HERE)}")
    return path


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# Self-test: run `python common.py` to sanity-check the solver.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    # Check 1: mass conservation. Both terms of Burgers are in divergence form,
    # so the spatial mean of u must be constant in time, to machine precision.
    x = grid(256)
    u0 = -np.sin(np.pi * x)[None, :]
    t, u = burgers_solve(u0, dt=5e-4, n_save=11)
    drift = np.abs(u.mean(-1) - u0.mean()).max()
    print(f"mass drift over t=[0,1]: {drift:.2e}   (should be ~1e-16)")

    # Check 2: self-convergence. Halving dt should change the answer very
    # little if the time integrator is doing its job.
    _, u_a = burgers_solve(u0, dt=1e-3, n_save=2)
    _, u_b = burgers_solve(u0, dt=5e-4, n_save=2)
    print(f"dt refinement rel-L2 diff: {rel_l2(u_a[:, -1], u_b[:, -1]):.2e}")

    # Check 3: grid refinement.
    x2 = grid(512)
    _, u_c = burgers_solve(-np.sin(np.pi * x2)[None, :], dt=5e-4, n_save=2)
    print(f"N refinement rel-L2 diff:  "
          f"{rel_l2(u_b[0, -1], u_c[0, -1, ::2]):.2e}")

    t0 = time.time()
    make_operator_dataset()
    print(f"dataset ready in {time.time() - t0:.1f}s")
