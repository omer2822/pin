"""
01 -- Physics-Informed Neural Network (PINN), from scratch.

    Run:  python 01_pinn_burgers.py        (~1-2 min on a laptop CPU)

THE ONE IDEA
------------
A PINN does not learn from data. It learns from the equation itself.

We declare that the solution is a neural network,

    u(x, t) ~= u_theta(x, t)          # a plain MLP, 2 inputs -> 1 output

and then we ask: what would make this network BE the solution? Three things:

    1. it satisfies the PDE everywhere inside the domain
    2. it matches the initial condition at t = 0
    3. it matches the boundary conditions at x = +-1

So we turn each of those into a squared-error term and minimise their sum:

    Loss = mean( r(x,t)^2 ) + mean( (u-u_0)^2 ) + mean( u_boundary^2 )

where r is the PDE residual

    r(x, t) = du/dt + u * du/dx - nu * d2u/dx2       ( == 0 for the true u )

THE TRICK THAT MAKES IT WORK
----------------------------
How do we get du/dt, du/dx, d2u/dx2? Not with finite differences -- we take
them with the SAME autograd that computes the gradients for backprop. The
network is a closed-form differentiable function of (x, t), so its derivatives
are exact, mesh-free, and available at any point you like. That is the entire
trick behind PINNs. `torch.autograd.grad(..., create_graph=True)` is the line
that matters.

WHAT A PINN IS AND IS NOT
-------------------------
This solves ONE instance of the PDE. Change the initial condition and you throw
the network away and retrain from scratch, which takes about a minute -- while
the classical spectral solver in common.py does the same job in 140
milliseconds and is more accurate. For a plain forward problem like this one, a
PINN LOSES to classical numerics, by roughly 500x. That is not a bug in this
script; it is the honest state of the field.

PINNs earn their keep when the classical solver cannot be written down:
inverse problems (nu unknown, infer it from scattered measurements), problems
where you have sparse noisy data AND physics, irregular geometries, and
high-dimensional PDEs where meshes are impossible.

Scripts 02 and 03 attack the complementary problem: learning the map from ANY
initial condition to its solution, all at once.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common

torch.manual_seed(0)
np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

NU = common.NU
N_COLLOCATION = 5_000      # random (x,t) points where we enforce the PDE
N_INITIAL = 300            # points where we enforce u(x,0) = -sin(pi x)
N_BOUNDARY = 300           # points where we enforce u(+-1,t) = 0
ADAM_STEPS = 4_000         # bump these two if you want a lower error;
LBFGS_STEPS = 600          # they are set for a ~90s laptop run


# ----------------------------------------------------------------------------
# The network. Nothing exotic: an MLP with tanh activations.
# ----------------------------------------------------------------------------
#
# tanh, not ReLU -- and this is not a style choice. The loss contains d2u/dx2,
# and the second derivative of a ReLU network is zero almost everywhere, so the
# viscous term would vanish and training would collapse. Any activation you use
# in a PINN must be smooth to at least the order of the PDE. tanh, sin, and
# gelu all work; relu does not.

class MLP(nn.Module):
    def __init__(self, layers=(2, 64, 64, 64, 64, 1)):
        super().__init__()
        mods = []
        for i in range(len(layers) - 1):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                mods.append(nn.Tanh())
        self.net = nn.Sequential(*mods)
        # Xavier init matters more here than in normal deep learning: the loss
        # involves derivatives of the network, so a bad scale at init makes the
        # residual term explode.
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))


# ----------------------------------------------------------------------------
# The physics-informed part.
# ----------------------------------------------------------------------------

def pde_residual(model, x, t):
    """r = u_t + u*u_x - nu*u_xx, evaluated by automatic differentiation."""
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u = model(x, t)

    # create_graph=True is essential: we need to backprop THROUGH these
    # derivatives when we later call loss.backward(). Without it the optimiser
    # sees no gradient from the residual term at all.
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                               create_graph=True)[0]
    return u_t + u * u_x - NU * u_xx


def sample_points(device):
    """Collocation points are just random draws from the domain -- there is no
    mesh anywhere in this script. Resampling them every so often acts like data
    augmentation and reduces overfitting to a fixed point cloud."""
    x_f = torch.rand(N_COLLOCATION, 1, device=device) * 2 - 1
    t_f = torch.rand(N_COLLOCATION, 1, device=device) * common.T_FINAL
    return x_f, t_f


def main():
    t_start = time.time()
    model = MLP().to(DEV)
    print(f"device={DEV}  params={common.count_params(model):,}")

    # --- fixed IC / BC point sets -------------------------------------------
    x_i = (torch.rand(N_INITIAL, 1, device=DEV) * 2 - 1)
    t_i = torch.zeros_like(x_i)
    u_i = -torch.sin(np.pi * x_i)                       # the initial condition

    t_b = torch.rand(N_BOUNDARY, 1, device=DEV) * common.T_FINAL
    x_b = torch.where(torch.rand_like(t_b) < 0.5,
                      -torch.ones_like(t_b), torch.ones_like(t_b))
    u_b = torch.zeros_like(t_b)                         # u(+-1, t) = 0

    x_f, t_f = sample_points(DEV)
    mse = nn.MSELoss()

    def total_loss():
        l_pde = mse(pde_residual(model, x_f, t_f), torch.zeros_like(t_f))
        l_ic = mse(model(x_i, t_i), u_i)
        l_bc = mse(model(x_b, t_b), u_b)
        # Loss weighting is THE practical pain point of PINNs. Equal weights
        # work for this problem; for stiffer ones people use adaptive schemes
        # (NTK-based, gradient-norm balancing, self-adaptive weights).
        return l_pde + l_ic + l_bc, l_pde, l_ic, l_bc

    # --- stage 1: Adam ------------------------------------------------------
    # Adam gets you into the right basin quickly but plateaus around 1e-4.
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)
    for it in range(1, ADAM_STEPS + 1):
        if it % 500 == 0:
            x_f, t_f = sample_points(DEV)                # resample collocation
        opt.zero_grad()
        loss, l_pde, l_ic, l_bc = total_loss()
        loss.backward()
        opt.step()
        sched.step()
        if it % 1000 == 0 or it == 1:
            print(f"  adam {it:5d}  loss={loss.item():.3e}  "
                  f"pde={l_pde.item():.2e} ic={l_ic.item():.2e} "
                  f"bc={l_bc.item():.2e}")

    # --- stage 2: L-BFGS ----------------------------------------------------
    # This second-order pass is what actually makes PINNs accurate, and it is
    # the step most tutorials skip. Expect the loss to drop another 1-2 orders
    # of magnitude. L-BFGS is full-batch, hence the closure.
    print("  switching to L-BFGS...")
    opt2 = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=LBFGS_STEPS,
                             history_size=50, tolerance_grad=1e-9,
                             tolerance_change=1e-11,
                             line_search_fn="strong_wolfe")

    state = {"n": 0}

    def closure():
        opt2.zero_grad()
        loss, l_pde, l_ic, l_bc = total_loss()
        loss.backward()
        state["n"] += 1
        if state["n"] % 200 == 0:
            print(f"  lbfgs {state['n']:5d}  loss={loss.item():.3e}  "
                  f"pde={l_pde.item():.2e} ic={l_ic.item():.2e} "
                  f"bc={l_bc.item():.2e}")
        return loss

    opt2.step(closure)
    train_time = time.time() - t_start
    print(f"trained in {train_time:.1f}s")

    # --- grade it against the classical solver ------------------------------
    # The PINN has never seen this. It only ever saw the equation.
    x_ref = common.grid(512)
    t_ref, u_ref = common.burgers_solve(-np.sin(np.pi * x_ref)[None, :],
                                        dt=5e-4, n_save=101)
    u_ref = u_ref[0]                                    # (101, 512)

    XX, TT = np.meshgrid(x_ref, t_ref, indexing="xy")
    with torch.no_grad():
        xt = torch.tensor(XX.reshape(-1, 1), dtype=torch.float32, device=DEV)
        tt = torch.tensor(TT.reshape(-1, 1), dtype=torch.float32, device=DEV)
        u_pinn = model(xt, tt).cpu().numpy().reshape(XX.shape)

    err = common.rel_l2(u_pinn, u_ref)
    print(f"\nrelative L2 error vs spectral reference: {err:.3%}")
    print(f"PINN training time {train_time:.1f}s  vs  "
          f"spectral solver ~0.14s for the same answer")

    # --- figure -------------------------------------------------------------
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])

    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(u_pinn, extent=[-1, 1, 0, 1], origin="lower",
                   aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("x"); ax.set_ylabel("t")
    ax.set_title(f"PINN solution u(x,t)   |   rel-L2 vs reference = {err:.2%}")
    for ts in (0.25, 0.5, 0.9):
        ax.axhline(ts, color="k", ls="--", lw=0.8, alpha=0.6)
    fig.colorbar(im, ax=ax, pad=0.01)

    for j, ts in enumerate((0.25, 0.5, 0.9)):
        i = np.argmin(np.abs(t_ref - ts))
        ax = fig.add_subplot(gs[1, j])
        ax.plot(x_ref, u_ref[i], "k-", lw=2.5, alpha=0.35, label="spectral ref")
        ax.plot(x_ref, u_pinn[i], "C3--", lw=1.6, label="PINN")
        ax.set_title(f"t = {t_ref[i]:.2f}"); ax.set_xlabel("x")
        ax.set_ylim(-1.15, 1.15)
        if j == 0:
            ax.set_ylabel("u"); ax.legend(fontsize=8)

    fig.suptitle("01 -- PINN: one PDE instance, learned from the equation alone",
                 y=0.99)
    fig.tight_layout()
    common.savefig(fig, "01_pinn.png")


if __name__ == "__main__":
    main()
