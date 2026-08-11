"""
02 -- DeepONet: the first neural operator, from scratch.

    Run:  python 02_deeponet.py        (~60s on a laptop CPU; generates data
                                        on first run, ~11s)

THE SHIFT IN GOAL
-----------------
Script 01 learned ONE function u(x,t). This script learns a MAP BETWEEN
FUNCTION SPACES -- an operator:

    G : u0(.)  |-->  u(., t=1)

Feed it any initial condition, get back the Burgers solution one time unit
later. Same network, no retraining, ~1 millisecond per solve. That is the
promise of operator learning: pay a large one-time training cost, then get
solutions essentially for free forever.

THE PROBLEM AND THE TRICK
-------------------------
A function is an infinite-dimensional object, and a network wants finite
vectors. DeepONet (Lu, Jin, Karniadakis 2019) splits the job in two:

    BRANCH net:  reads the input FUNCTION, sampled at m fixed "sensor"
                 locations  ->  outputs p coefficients  b_1..b_p
    TRUNK  net:  reads a single output LOCATION y
                 ->  outputs p basis values          t_1(y)..t_p(y)

    G(u0)(y)  =  sum_k  b_k(u0) * t_k(y)  +  b_0

Look at that formula: it is a basis expansion where the TRUNK learns the basis
functions and the BRANCH learns the coefficients. Classical spectral methods
fix the basis in advance (sines, Chebyshev polynomials); DeepONet learns a
basis tailored to this particular operator. The design is backed by the
universal approximation theorem for operators (Chen & Chen, 1995).

WHAT THIS BUYS AND WHAT IT COSTS
--------------------------------
  + The output is a genuine continuous function: y is a real number, so you can
    evaluate the answer at any point, on any grid, at any resolution. We
    demonstrate that at the end.
  + The architecture is trivial -- two MLPs and a dot product.
  - The INPUT is locked to the m sensor locations chosen at training time. Feed
    it a u0 on a different grid and you must interpolate first.
  - Accuracy on sharp features is mediocre. Expect a few percent here, versus
    well under one percent for the FNO in script 03. The branch net has to
    compress the whole input function into p numbers through a dense layer,
    which throws away local structure.
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

M_SENSORS = 128       # input function sampled here (fixed forever after training)
P_BASIS = 200         # number of learned basis functions
WIDTH = 256
N_FOURIER = 16        # harmonic features fed to the trunk (see below)
EPOCHS = 600
BATCH = 64


# ----------------------------------------------------------------------------
# The architecture
# ----------------------------------------------------------------------------

def mlp(sizes, act=nn.GELU, last_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    def __init__(self, m=M_SENSORS, p=P_BASIS, width=WIDTH,
                 n_fourier=N_FOURIER):
        super().__init__()
        self.n_fourier = n_fourier
        # Branch: the discretised input function -> p coefficients.
        self.branch = mlp([m, width, width, width, p])
        # Trunk: output coordinate (plus harmonics) -> p basis values.
        # last_act=True is deliberate: the trunk's outputs are basis FUNCTIONS,
        # and leaving them unactivated tends to make training unstable.
        self.trunk = mlp([1 + 2 * n_fourier, width, width, width, p],
                         last_act=True)
        self.bias = nn.Parameter(torch.zeros(1))

    def features(self, y):
        """Hand the trunk [y, sin(pi y), cos(pi y), sin(2 pi y), ...] instead of
        just y. Plain MLPs are famously bad at representing oscillations
        (spectral bias -- they fit low frequencies fast and high ones almost
        never), and our solutions are periodic with sharp fronts. Handing the
        trunk a ready-made harmonic basis to mix cut the test error here from
        ~5% to ~3.5%. Same idea as Fourier features in NeRF, and the same idea
        the FNO in script 03 takes to its logical conclusion."""
        if self.n_fourier == 0:
            return y
        k = torch.arange(1, self.n_fourier + 1, device=y.device,
                         dtype=y.dtype) * np.pi
        return torch.cat([y, torch.sin(k * y), torch.cos(k * y)], dim=-1)

    def forward(self, a, y):
        """a: (B, m) input functions.   y: (Q, 1) query locations.
        returns (B, Q) -- every input evaluated at every query point."""
        b = self.branch(a)                 # (B, p)  coefficients
        t = self.trunk(self.features(y))   # (Q, p)  basis evaluated at y
        return b @ t.T + self.bias         # (B, Q)  the dot product


def rel_l2_loss(pred, true):
    """Relative L2, averaged over the batch. Training on RELATIVE error rather
    than MSE matters: it stops the loss from being dominated by the
    high-amplitude samples, and it is the metric everyone reports."""
    num = torch.linalg.norm(pred - true, dim=-1)
    den = torch.linalg.norm(true, dim=-1)
    return (num / den).mean()


def main():
    # --- data ---------------------------------------------------------------
    a_tr, u_tr, a_te, u_te, x_full = common.make_operator_dataset()
    # Sensors: subsample the 256-point grid down to m=128 input locations.
    step = len(x_full) // M_SENSORS
    x_sens = x_full[::step]
    a_tr_s, a_te_s = a_tr[:, ::step], a_te[:, ::step]

    # Output is evaluated on the full 256-point grid.
    A_tr = torch.tensor(a_tr_s, dtype=torch.float32, device=DEV)
    U_tr = torch.tensor(u_tr, dtype=torch.float32, device=DEV)
    A_te = torch.tensor(a_te_s, dtype=torch.float32, device=DEV)
    U_te = torch.tensor(u_te, dtype=torch.float32, device=DEV)
    Y = torch.tensor(x_full[:, None], dtype=torch.float32, device=DEV)

    print(f"device={DEV}  train={A_tr.shape}  ->  {U_tr.shape}   "
          f"sensors m={M_SENSORS}, basis p={P_BASIS}")

    model = DeepONet().to(DEV)
    print(f"params={common.count_params(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # --- train --------------------------------------------------------------
    t0 = time.time()
    n = A_tr.shape[0]
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=DEV)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = rel_l2_loss(model(A_tr[idx], Y), U_tr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if ep % 50 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                te = rel_l2_loss(model(A_te, Y), U_te).item()
            print(f"  epoch {ep:4d}  train={tot / n:.4f}  test={te:.4f}")
    train_time = time.time() - t0

    # --- evaluate -----------------------------------------------------------
    model.eval()
    with torch.no_grad():
        t1 = time.time()
        pred_te = model(A_te, Y)
        infer_time = time.time() - t1
    pred_te_np = pred_te.cpu().numpy()
    err_per_sample = (np.linalg.norm(pred_te_np - u_te, axis=-1)
                      / np.linalg.norm(u_te, axis=-1))
    print(f"\ntrained in {train_time:.1f}s")
    print(f"test relative L2: mean {err_per_sample.mean():.3%}  "
          f"median {np.median(err_per_sample):.3%}  "
          f"worst {err_per_sample.max():.3%}")
    print(f"inference: {len(u_te)} PDE solves in {infer_time * 1e3:.1f} ms "
          f"({infer_time / len(u_te) * 1e6:.0f} us each)")

    # --- the payoff: evaluate at arbitrary output locations -----------------
    # The trunk takes a real number, so we can query a 4x finer grid than the
    # one we trained on, at zero cost and with no interpolation. Note this is
    # about the OUTPUT only -- the input is still stuck at m sensors.
    x_fine = np.linspace(-1, 1, 4 * len(x_full), endpoint=False)
    with torch.no_grad():
        Yf = torch.tensor(x_fine[:, None], dtype=torch.float32, device=DEV)
        pred_fine = model(A_te[:1], Yf).cpu().numpy()[0]

    # --- figure -------------------------------------------------------------
    order = np.argsort(err_per_sample)
    picks = [("best", order[0]), ("median", order[len(order) // 2]),
             ("worst", order[-1])]

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for j, (label, i) in enumerate(picks):
        ax = axes[0, j]
        ax.plot(x_sens, a_te_s[i], "C0-", lw=1.4, label="input $u_0$")
        ax.plot(x_full, u_te[i], "k-", lw=2.5, alpha=0.35, label="true $u(.,1)$")
        ax.plot(x_full, pred_te_np[i], "C3--", lw=1.5, label="DeepONet")
        ax.set_title(f"{label} test case -- rel-L2 {err_per_sample[i]:.2%}")
        ax.set_xlabel("x")
        if j == 0:
            ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.hist(err_per_sample * 100, bins=30, color="C0", alpha=0.8)
    ax.axvline(err_per_sample.mean() * 100, color="C3", ls="--",
               label=f"mean {err_per_sample.mean():.2%}")
    ax.set_xlabel("relative L2 error (%)"); ax.set_ylabel("test samples")
    ax.set_title("error distribution"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(x_full, u_te[0], "ko", ms=3, alpha=0.4, label="truth (256 pts)")
    ax.plot(x_fine, pred_fine, "C3-", lw=1.2,
            label="DeepONet queried at 1024 pts")
    ax.set_title("output is a continuous function"); ax.set_xlabel("x")
    ax.legend(fontsize=8)

    # The learned basis: this is what the trunk net invented for itself.
    ax = axes[1, 2]
    with torch.no_grad():
        basis = model.trunk(model.features(Y)).cpu().numpy()
    for k in range(6):
        ax.plot(x_full, basis[:, k], lw=1.2)
    ax.set_title("6 of the $p$ learned trunk basis functions")
    ax.set_xlabel("x")

    fig.suptitle("02 -- DeepONet: branch/trunk decomposition of the solution "
                 "operator", y=1.0)
    fig.tight_layout()
    common.savefig(fig, "02_deeponet.png")


if __name__ == "__main__":
    main()
