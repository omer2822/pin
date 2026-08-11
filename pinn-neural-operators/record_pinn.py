"""
record_pinn.py -- train the PINN from 01 and record what it looks like on the
way, so the animation in ani.py can show real footage instead of a cartoon.

    python record_pinn.py            (needs torch; ~1-2 min on a laptop CPU)

It imports the *actual* MLP and pde_residual from 01_pinn_burgers.py -- no
reimplementation -- and only adds a snapshot hook to the training loop. What
gets written to data/pinn_training.npz:

    x, t_snap     the grid snapshots live on
    frames        (n_frames, n_t, n_x)  u_theta(x,t) at each checkpoint
    resid         (n_frames, n_t, n_x)  |PDE residual| at each checkpoint
    loss, l_pde, l_ic, l_bc             the losses logged at each checkpoint
    step, stage                         iteration count, 0 = Adam, 1 = L-BFGS
    u_ref                               spectral reference on the same grid

Frame 0 is the network at initialisation -- an untrained tanh MLP -- so the
animation's starting curve is the real thing, not a drawn squiggle.
"""

import importlib.util
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_script_01():
    """01_pinn_burgers.py starts with a digit, so a plain import won't do."""
    path = os.path.join(HERE, "01_pinn_burgers.py")
    spec = importlib.util.spec_from_file_location("pinn01", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # main() is guarded by __main__
    return mod


NX_SNAP = 128        # snapshot resolution in x
NT_SNAP = 41         # snapshot resolution in t
ADAM_EVERY = 100     # checkpoint cadence during Adam
LBFGS_EVERY = 10     # ... and during L-BFGS, which moves much faster per step


def main():
    p = load_script_01()
    common = p.common
    torch.manual_seed(0)
    np.random.seed(0)
    dev = p.DEV

    model = p.MLP().to(dev)
    print(f"device={dev}  params={common.count_params(model):,}")

    # --- the grid every snapshot is evaluated on ----------------------------
    x_s = common.grid(NX_SNAP)
    t_s = np.linspace(0.0, common.T_FINAL, NT_SNAP)
    XX, TT = np.meshgrid(x_s, t_s, indexing="xy")
    x_eval = torch.tensor(XX.reshape(-1, 1), dtype=torch.float32, device=dev)
    t_eval = torch.tensor(TT.reshape(-1, 1), dtype=torch.float32, device=dev)

    frames, resid, losses, steps, stages = [], [], [], [], []

    def snapshot(step, stage, loss_tuple):
        u = model(x_eval, t_eval)
        r = p.pde_residual(model, x_eval.clone(), t_eval.clone())
        frames.append(u.detach().cpu().numpy().reshape(XX.shape))
        resid.append(np.abs(r.detach().cpu().numpy().reshape(XX.shape)))
        losses.append([v.item() for v in loss_tuple])
        steps.append(step)
        stages.append(stage)

    # --- same IC / BC / collocation setup as 01 -----------------------------
    x_i = (torch.rand(p.N_INITIAL, 1, device=dev) * 2 - 1)
    t_i = torch.zeros_like(x_i)
    u_i = -torch.sin(np.pi * x_i)

    t_b = torch.rand(p.N_BOUNDARY, 1, device=dev) * common.T_FINAL
    x_b = torch.where(torch.rand_like(t_b) < 0.5,
                      -torch.ones_like(t_b), torch.ones_like(t_b))
    u_b = torch.zeros_like(t_b)

    x_f, t_f = p.sample_points(dev)
    mse = torch.nn.MSELoss()

    def total_loss():
        l_pde = mse(p.pde_residual(model, x_f, t_f), torch.zeros_like(t_f))
        l_ic = mse(model(x_i, t_i), u_i)
        l_bc = mse(model(x_b, t_b), u_b)
        return l_pde + l_ic + l_bc, l_pde, l_ic, l_bc

    t_start = time.time()
    snapshot(0, 0, total_loss())          # frame 0: the untrained network

    # --- stage 1: Adam ------------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)
    for it in range(1, p.ADAM_STEPS + 1):
        if it % 500 == 0:
            x_f, t_f = p.sample_points(dev)
        opt.zero_grad()
        lt = total_loss()
        lt[0].backward()
        opt.step()
        sched.step()
        if it % ADAM_EVERY == 0:
            snapshot(it, 0, lt)
            if it % 1000 == 0:
                print(f"  adam {it:5d}  loss={lt[0].item():.3e}")

    # --- stage 2: L-BFGS ----------------------------------------------------
    print("  switching to L-BFGS...")
    opt2 = torch.optim.LBFGS(model.parameters(), lr=1.0,
                             max_iter=p.LBFGS_STEPS, history_size=50,
                             tolerance_grad=1e-9, tolerance_change=1e-11,
                             line_search_fn="strong_wolfe")
    state = {"n": 0}

    def closure():
        opt2.zero_grad()
        lt = total_loss()
        lt[0].backward()
        state["n"] += 1
        if state["n"] % LBFGS_EVERY == 0:
            snapshot(p.ADAM_STEPS + state["n"], 1, lt)
            if state["n"] % 200 == 0:
                print(f"  lbfgs {state['n']:5d}  loss={lt[0].item():.3e}")
        return lt[0]

    opt2.step(closure)
    snapshot(p.ADAM_STEPS + state["n"], 1, total_loss())
    train_time = time.time() - t_start
    print(f"trained in {train_time:.1f}s, {len(frames)} snapshots")

    # --- spectral reference on the snapshot grid ----------------------------
    x_ref = common.grid(512)
    t_ref, u_ref_full = common.burgers_solve(-np.sin(np.pi * x_ref)[None, :],
                                             dt=5e-4, n_save=201)
    u_ref_full = u_ref_full[0]
    u_ref = np.stack([
        np.interp(x_s, x_ref, u_ref_full[np.argmin(np.abs(t_ref - tv))])
        for tv in t_s
    ])

    err = common.rel_l2(frames[-1], u_ref)
    print(f"final rel-L2 vs spectral reference: {err:.3%}")

    out = os.path.join(common.CACHE_DIR, "pinn_training.npz")
    os.makedirs(common.CACHE_DIR, exist_ok=True)
    np.savez_compressed(
        out,
        x=x_s, t_snap=t_s,
        frames=np.array(frames, dtype=np.float32),
        resid=np.array(resid, dtype=np.float32),
        loss=np.array(losses, dtype=np.float64),
        step=np.array(steps), stage=np.array(stages),
        u_ref=u_ref.astype(np.float32),
        train_time=train_time, rel_l2=err,
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
