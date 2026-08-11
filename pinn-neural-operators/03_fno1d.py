"""
03 -- Fourier Neural Operator (FNO), 1D, from scratch.

    Run:  python 03_fno1d.py        (~2 min on a laptop CPU)

Same task as script 02:  G : u0(.) |--> u(., t=1)  for viscous Burgers.
Different, and much better, architecture.

WHY A CONVOLUTION, AND WHY IN FOURIER SPACE
-------------------------------------------
Start from the linear case. Solving a linear PDE means applying a Green's
function -- an integral operator:

    (K u)(x) = integral  k(x, y) u(y) dy

If the problem is translation-invariant then k(x,y) = k(x-y) and that integral
is a CONVOLUTION. And the convolution theorem says a convolution is just
pointwise multiplication in Fourier space:

    K u  =  IFFT( k_hat . FFT(u) )

So: instead of learning a kernel k, learn its Fourier multiplier k_hat
DIRECTLY -- one complex weight per frequency. That is the whole idea. One FNO
layer is

    u  <-  sigma(  W u  +  IFFT( R . FFT(u) )  )
                   \-/     \------------------/
                  local     global convolution,
                  1x1 conv  R = learned complex weights, truncated to the
                            lowest `modes` frequencies

Stack four of those with a nonlinearity between them and you can represent
nonlinear operators, not just linear ones.

THE THREE THINGS THAT MAKE THIS GOOD
------------------------------------
1. GLOBAL RECEPTIVE FIELD IN ONE LAYER. An ordinary CNN needs many layers to
   propagate information across the domain. The FFT touches every point at
   once, which is exactly right for PDEs, where a solution at time t depends on
   the whole initial condition.

2. MODE TRUNCATION IS THE REGULARISER. We keep only the lowest `modes`
   frequencies (16 here) and throw the rest away. PDE solutions are smooth-ish,
   so the low modes carry the physics; discarding the tail prevents overfitting
   AND caps the parameter count.

3. DISCRETISATION INVARIANCE. The weights R index FREQUENCIES, not grid
   points. Frequency 5 means the same thing on a 64-point grid and a 4096-point
   grid. So an FNO trained at one resolution can be evaluated at another with
   no retraining. The bottom of this script trains at N=128 and evaluates at
   N=256 to prove it -- and that property is exactly what a DeepONet (whose
   branch net has a fixed input dimension) cannot do.

COST: O(N log N) per layer, from the FFT.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common

torch.manual_seed(0)
np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

MODES = 16          # Fourier modes kept per layer
WIDTH = 48          # channels in the lifted space
N_LAYERS = 4
EPOCHS = 120
BATCH = 20
TRAIN_SUB = 2       # train on every 2nd grid point (N=128), test at N=256


# ----------------------------------------------------------------------------
# The one new layer type
# ----------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """Global convolution done as a pointwise multiply in Fourier space."""

    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * out_ch)
        # One complex matrix per retained frequency. This is R.
        # Shape (in, out, modes); torch.cfloat = complex64.
        self.weight = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes, dtype=torch.cfloat))

    def forward(self, x):                       # x: (B, C, N)
        B, C, N = x.shape
        # rfft because x is real: only non-negative frequencies are stored.
        x_ft = torch.fft.rfft(x)                # (B, C, N//2+1) complex

        out_ft = torch.zeros(B, self.weight.shape[1], N // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        # Multiply the lowest `modes` frequencies by the learned weights and
        # leave everything above them at ZERO -- that is the truncation. Note
        # `modes` is capped by the grid: a 32-point grid cannot hold 16 modes.
        k = min(self.modes, N // 2 + 1)
        out_ft[:, :, :k] = torch.einsum(
            "bik,iok->bok", x_ft[:, :, :k], self.weight[:, :, :k])

        # Back to physical space. Passing n=N is what lets the SAME weights
        # produce an output on ANY grid size -- the source of discretisation
        # invariance.
        return torch.fft.irfft(out_ft, n=N)


class FNO1d(nn.Module):
    def __init__(self, modes=MODES, width=WIDTH, n_layers=N_LAYERS):
        super().__init__()
        # LIFT: (u0(x), x) -> width channels. Feeding the coordinate x is
        # standard in FNO and gives the network a sense of absolute position.
        self.lift = nn.Linear(2, width)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        # The 1x1 conv is the local ("W") path. It carries the high frequencies
        # that mode truncation deleted -- without it the FNO cannot represent
        # anything sharper than mode 16.
        self.local = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        # PROJECT back down to a scalar field.
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(),
                                  nn.Linear(128, 1))

    def forward(self, u0, x):
        """u0: (B, N) input function.  x: (N,) grid coordinates."""
        B, N = u0.shape
        xg = x.expand(B, N)
        h = torch.stack([u0, xg], dim=-1)        # (B, N, 2)
        h = self.lift(h).permute(0, 2, 1)        # (B, width, N)

        for i, (sp, lc) in enumerate(zip(self.spectral, self.local)):
            h_new = sp(h) + lc(h)
            h = F.gelu(h_new) if i < len(self.spectral) - 1 else h_new

        h = self.proj(h.permute(0, 2, 1))        # (B, N, 1)
        return h.squeeze(-1)                     # (B, N)


def rel_l2_loss(pred, true):
    num = torch.linalg.norm(pred - true, dim=-1)
    den = torch.linalg.norm(true, dim=-1)
    return (num / den).mean()


def main():
    # --- data ---------------------------------------------------------------
    a_tr, u_tr, a_te, u_te, x_full = common.make_operator_dataset()

    # Train on a COARSE grid on purpose, so the super-resolution test at the
    # end is a real zero-shot test.
    x_lo = x_full[::TRAIN_SUB]
    A_tr = torch.tensor(a_tr[:, ::TRAIN_SUB], dtype=torch.float32, device=DEV)
    U_tr = torch.tensor(u_tr[:, ::TRAIN_SUB], dtype=torch.float32, device=DEV)
    A_te_lo = torch.tensor(a_te[:, ::TRAIN_SUB], dtype=torch.float32, device=DEV)
    U_te_lo = torch.tensor(u_te[:, ::TRAIN_SUB], dtype=torch.float32, device=DEV)
    X_lo = torch.tensor(x_lo, dtype=torch.float32, device=DEV)

    A_te_hi = torch.tensor(a_te, dtype=torch.float32, device=DEV)
    X_hi = torch.tensor(x_full, dtype=torch.float32, device=DEV)

    print(f"device={DEV}  training grid N={len(x_lo)}  "
          f"(will also be tested zero-shot at N={len(x_full)})")

    model = FNO1d().to(DEV)
    print(f"params={common.count_params(model):,}  "
          f"modes={MODES} width={WIDTH} layers={N_LAYERS}")

    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
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
            loss = rel_l2_loss(model(A_tr[idx], X_lo), U_tr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if ep % 20 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                te = rel_l2_loss(model(A_te_lo, X_lo), U_te_lo).item()
            print(f"  epoch {ep:4d}  train={tot / n:.4f}  test={te:.4f}  "
                  f"[{time.time() - t0:.0f}s]")
    train_time = time.time() - t0

    # --- evaluate at the TRAINING resolution --------------------------------
    model.eval()
    with torch.no_grad():
        t1 = time.time()
        pred_lo = model(A_te_lo, X_lo)
        infer_time = time.time() - t1
    pred_lo = pred_lo.cpu().numpy()
    u_te_lo = u_te[:, ::TRAIN_SUB]
    err_lo = (np.linalg.norm(pred_lo - u_te_lo, axis=-1)
              / np.linalg.norm(u_te_lo, axis=-1))

    print(f"\ntrained in {train_time:.1f}s")
    print(f"test rel-L2 @ N={len(x_lo)} (train res): mean {err_lo.mean():.3%}  "
          f"median {np.median(err_lo):.3%}  worst {err_lo.max():.3%}")
    print(f"inference: {len(u_te)} PDE solves in {infer_time * 1e3:.1f} ms "
          f"({infer_time / len(u_te) * 1e6:.0f} us each)")

    # --- THE zero-shot super-resolution test --------------------------------
    # Exact same weights, twice the grid points. No retraining, no fine-tuning,
    # not even a config change. Try this with any ordinary CNN or MLP and the
    # shapes will not even line up.
    with torch.no_grad():
        pred_hi = model(A_te_hi, X_hi).cpu().numpy()
    err_hi = (np.linalg.norm(pred_hi - u_te, axis=-1)
              / np.linalg.norm(u_te, axis=-1))
    print(f"test rel-L2 @ N={len(x_full)} (never trained on): "
          f"mean {err_hi.mean():.3%}")
    print("  -> the two numbers being close IS the discretisation-invariance "
          "property.")

    # --- figure -------------------------------------------------------------
    order = np.argsort(err_hi)
    picks = [("best", order[0]), ("median", order[len(order) // 2]),
             ("worst", order[-1])]

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for j, (label, i) in enumerate(picks):
        ax = axes[0, j]
        ax.plot(x_full, a_te[i], "C0-", lw=1.2, alpha=0.7, label="input $u_0$")
        ax.plot(x_full, u_te[i], "k-", lw=2.5, alpha=0.35, label="true $u(.,1)$")
        ax.plot(x_full, pred_hi[i], "C3--", lw=1.5, label="FNO")
        ax.set_title(f"{label} test case -- rel-L2 {err_hi[i]:.2%}")
        ax.set_xlabel("x")
        if j == 0:
            ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.hist(err_hi * 100, bins=30, color="C2", alpha=0.85)
    ax.axvline(err_hi.mean() * 100, color="C3", ls="--",
               label=f"mean {err_hi.mean():.2%}")
    ax.set_xlabel("relative L2 error (%)"); ax.set_ylabel("test samples")
    ax.set_title("error distribution"); ax.legend(fontsize=8)

    # Resolution sweep: evaluate the SAME weights on several grids.
    ax = axes[1, 1]
    res_list, res_err = [], []
    for sub in (8, 4, 2, 1):
        N = len(x_full) // sub
        with torch.no_grad():
            p = model(torch.tensor(a_te[:, ::sub], dtype=torch.float32,
                                   device=DEV),
                      torch.tensor(x_full[::sub], dtype=torch.float32,
                                   device=DEV)).cpu().numpy()
        t = u_te[:, ::sub]
        res_list.append(N)
        res_err.append(float(np.mean(np.linalg.norm(p - t, axis=-1)
                                     / np.linalg.norm(t, axis=-1))) * 100)
    ax.plot(res_list, res_err, "o-", color="C2")
    ax.axvline(len(x_lo), color="k", ls="--", lw=1,
               label=f"trained at N={len(x_lo)}")
    ax.set_xscale("log", base=2); ax.set_xlabel("evaluation grid size N")
    ax.set_ylabel("rel-L2 (%)"); ax.set_ylim(bottom=0)
    ax.set_title("same weights, different grids"); ax.legend(fontsize=8)

    # What the network actually learned: magnitude of the spectral weights.
    ax = axes[1, 2]
    for li, sp in enumerate(model.spectral):
        mag = sp.weight.detach().abs().mean(dim=(0, 1)).cpu().numpy()
        ax.semilogy(np.arange(len(mag)), mag, lw=1.4, label=f"layer {li}")
    ax.set_xlabel("Fourier mode k"); ax.set_ylabel("mean |R|")
    ax.set_title("learned spectral weights"); ax.legend(fontsize=7)

    fig.suptitle("03 -- Fourier Neural Operator: learn the kernel's Fourier "
                 "multiplier", y=1.0)
    fig.tight_layout()
    common.savefig(fig, "03_fno.png")


if __name__ == "__main__":
    main()
