"""
ani.py -- a manimgl animation of what script 01 actually does.

    venv./bin/manimgl ani.py PINNFull -w              # everything, ~2.5 min
    venv./bin/manimgl ani.py Training -w              # just the payoff
    venv./bin/manimgl ani.py TheNetwork -w -l         # low quality, fast

Scenes, in order:

    TheNetwork      the MLP from 01, and a signal walking through it
    AutogradTrick   where u_t, u_x, u_xx come from, and why create_graph=True
    WhereLossLives  the three point sets on the (x,t) domain
    Training        REAL recorded footage of the network learning
    Scoreboard      the honest comparison against the spectral solver
    PINNFull        all five, back to back

`Training` and `Scoreboard` play back a recording made by record_pinn.py --
every curve, every residual value and every loss number in them came out of an
actual training run of 01, not out of a rate function. Make the recording once:

    python record_pinn.py            # writes data/pinn_training.npz

That needs torch (see requirements.txt), so run it with the project's own
interpreter -- the venv that has manimgl in it does not have torch, and does
not need it: ani.py itself only reads the npz.

Without that file those two scenes render a card telling you to run it. The
other three are self-contained; the reference solution they draw comes from the
spectral solver in common.py, computed live.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from manimlib import *

import common

# ----------------------------------------------------------------------------
# One palette, used consistently: each quantity keeps its colour across scenes.
# ----------------------------------------------------------------------------
C_X = BLUE_B          # the spatial input
C_T = TEAL_B          # the temporal input
C_U = YELLOW          # the network's output, u_theta
C_R = RED_B           # the PDE residual / the PDE loss term
C_IC = GREEN_B        # initial condition
C_BC = MAROON_B       # boundary condition
C_REF = GREY_B        # the spectral reference solution
C_DIM = GREY_C

REC_PATH = os.path.join(common.CACHE_DIR, "pinn_training.npz")


def load_recording():
    """The npz written by record_pinn.py, or None if it was never run."""
    if not os.path.exists(REC_PATH):
        return None
    return dict(np.load(REC_PATH))


def missing_recording_card(scene):
    """Shown instead of faking data we do not have. Once per scene object, so
    PINNFull -- where Training and Scoreboard share one scene -- says it once."""
    if getattr(scene, "_said_no_recording", False):
        return
    scene._said_no_recording = True
    msg = VGroup(
        Text("no recording found", font_size=44, color=C_R),
        Text("data/pinn_training.npz does not exist yet.", font_size=28),
        Text("Run:  python record_pinn.py", font_size=30, color=C_U),
        Text("(trains the PINN from 01 once, ~1 min, and saves what it saw;",
             font_size=22, color=C_DIM),
        Text("needs torch, so use the interpreter from requirements.txt)",
             font_size=22, color=C_DIM),
    )
    msg.arrange(DOWN, buff=0.42)
    scene.play(FadeIn(msg, shift=0.3 * UP))
    scene.wait(3)
    scene.play(FadeOut(msg))


def clear_scene(scene, run_time=0.8):
    """Fade out the scene's content -- self.mobjects also holds the camera
    frame, which must not be faded or the next scene renders into nothing."""
    doomed = [m for m in scene.mobjects if m is not scene.camera.frame]
    if doomed:
        scene.play(*[FadeOut(m) for m in doomed], run_time=run_time)


def section_title(text, subtitle=None):
    title = Text(text, font_size=40)
    if subtitle is None:
        return title
    sub = Text(subtitle, font_size=24, color=C_DIM)
    group = VGroup(title, sub).arrange(DOWN, buff=0.22)
    return group


# ============================================================================
# Scene 1 -- the network itself
# ============================================================================

LAYER_SIZES = (2, 7, 7, 7, 7, 1)      # 7 drawn nodes stands in for 64 real ones
NODE_R = 0.11


def build_network(width=9.0, height=4.2):
    """The MLP from 01: 2 -> 64 -> 64 -> 64 -> 64 -> 1, tanh between."""
    xs = np.linspace(-width / 2, width / 2, len(LAYER_SIZES))
    layers = VGroup()
    for xi, n in zip(xs, LAYER_SIZES):
        col = VGroup()
        ys = np.linspace(height / 2, -height / 2, n) if n > 1 else [0.0]
        for y in ys:
            dot = Dot(np.array([xi, y, 0.0]), radius=NODE_R)
            dot.set_fill(GREY_D, 1).set_stroke(GREY_A, 1.2)
            col.add(dot)
        layers.add(col)

    edges = VGroup()
    for a, b in zip(layers[:-1], layers[1:]):
        group = VGroup()
        for pa in a:
            for pb in b:
                line = Line(pa.get_center(), pb.get_center())
                line.set_stroke(GREY_D, 1.0, opacity=0.45)
                group.add(line)
        edges.add(group)

    layers[0][0].set_fill(C_X, 1).set_stroke(C_X, 2)
    layers[0][1].set_fill(C_T, 1).set_stroke(C_T, 2)
    layers[-1][0].set_fill(C_U, 1).set_stroke(C_U, 2)
    return layers, edges


def network_pulse(scene, layers, edges, color=C_U, lag=0.28, run_time=0.5):
    """Send one signal from the inputs to the output, layer by layer."""
    for i, group in enumerate(edges):
        flash = group.copy().set_stroke(color, 2.5, opacity=1)
        scene.play(
            ShowPassingFlash(flash, time_width=0.6, run_time=run_time + lag),
            AnimationGroup(*[
                Indicate(d, scale_factor=1.5, color=color)
                for d in layers[i + 1]
            ], lag_ratio=0.05, run_time=run_time + lag),
        )


class TheNetwork(Scene):
    def construct(self):
        head = section_title(
            "A PINN declares that the solution IS a network",
            "01_pinn_burgers.py",
        )
        head.to_edge(UP, buff=0.45)
        self.play(FadeIn(head, shift=0.3 * DOWN))

        claim = Tex(
            r"u(x,t) \;\approx\; u_\theta(x,t)",
            t2c={"x": C_X, "t": C_T, r"u_\theta": C_U},
            font_size=54,
        )
        self.play(Write(claim))
        self.wait(1.2)
        self.play(claim.animate.scale(0.62).next_to(head, DOWN, buff=0.35))

        layers, edges = build_network()
        net = VGroup(edges, layers)
        net.shift(0.55 * DOWN)

        in_labels = VGroup(
            Tex("x", color=C_X).next_to(layers[0][0], LEFT, buff=0.35),
            Tex("t", color=C_T).next_to(layers[0][1], LEFT, buff=0.35),
        )
        out_label = Tex(r"u_\theta", color=C_U)
        out_label.next_to(layers[-1][0], RIGHT, buff=0.35)

        self.play(
            ShowCreation(edges, lag_ratio=0.0, run_time=1.4),
            FadeIn(layers, lag_ratio=0.02, run_time=1.4),
        )
        self.play(FadeIn(in_labels), FadeIn(out_label))

        spec = Text(
            "4 hidden layers of 64,  tanh,  12,737 parameters",
            font_size=26, color=C_DIM,
        )
        spec.to_edge(DOWN, buff=0.55)
        self.play(FadeIn(spec, shift=0.2 * UP))

        for _ in range(2):
            network_pulse(self, layers, edges)
            self.wait(0.25)

        # The activation choice is a hard constraint here, not a preference.
        why = VGroup(
            Text("tanh, not ReLU", font_size=30, color=C_U),
            Text("the loss contains u_xx, and a ReLU net's second derivative",
                 font_size=23, color=C_DIM),
            Text("is zero almost everywhere -- the viscous term would vanish",
                 font_size=23, color=C_DIM),
        ).arrange(DOWN, buff=0.16)
        why.to_edge(DOWN, buff=0.4)
        self.play(FadeOut(spec), FadeIn(why, shift=0.2 * UP))
        self.wait(2.5)
        clear_scene(self)


# ============================================================================
# Scene 2 -- where the derivatives come from
# ============================================================================

class AutogradTrick(Scene):
    def construct(self):
        head = section_title(
            "The trick: the derivatives come from autograd",
            "the same autograd that computes the weight gradients",
        )
        head.to_edge(UP, buff=0.45)
        self.play(FadeIn(head, shift=0.3 * DOWN))

        layers, edges = build_network(width=5.4, height=2.4)
        net = VGroup(edges, layers).scale(0.8)
        net.to_edge(LEFT, buff=0.7).shift(0.3 * DOWN)
        self.play(ShowCreation(edges, lag_ratio=0.0), FadeIn(layers))

        u_node = layers[-1][0]
        u_lab = Tex(r"u_\theta", color=C_U, font_size=40)
        u_lab.next_to(u_node, UP, buff=0.25)
        self.play(FadeIn(u_lab))

        # u_theta is a closed-form differentiable function of (x, t), so its
        # derivatives are exact and available at any point -- no mesh anywhere.
        grads = VGroup(
            Tex(r"u_t \;=\; \partial u_\theta / \partial t", t2c={"u_t": C_T}),
            Tex(r"u_x \;=\; \partial u_\theta / \partial x", t2c={"u_x": C_X}),
            Tex(r"u_{xx} \;=\; \partial u_x / \partial x",
                t2c={"u_{xx}": C_R}),
        )
        for g in grads:
            g.scale(0.75)
        grads.arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        grads.to_edge(RIGHT, buff=1.1).shift(0.55 * UP)

        arrows = VGroup(*[
            Arrow(u_node.get_right(), g.get_left(), buff=0.18,
                  stroke_width=3).set_color(GREY_B)
            for g in grads
        ])

        for a, g in zip(arrows, grads):
            self.play(GrowArrow(a), FadeIn(g, shift=0.2 * RIGHT), run_time=0.6)
        self.wait(0.6)

        code = Text(
            "torch.autograd.grad(u, x, create_graph=True)",
            font_size=24, font="Monospace", color=GREY_A,
        )
        code.move_to([0.6, -2.1, 0])
        box = SurroundingRectangle(code, buff=0.18, color=C_U)
        box.set_stroke(width=2)
        self.play(FadeIn(code), ShowCreation(box))

        note = Text(
            "create_graph=True is the whole line:  without it the residual\n"
            "term contributes no gradient to the optimiser at all",
            font_size=22, color=C_DIM,
        )
        note.next_to(box, DOWN, buff=0.3)
        self.play(FadeIn(note))
        self.wait(2.2)

        self.play(
            FadeOut(VGroup(code, box, note)),
            FadeOut(arrows),
            FadeOut(net), FadeOut(u_lab),
        )

        resid = Tex(
            r"r(x,t) \;=\; u_t \;+\; u_\theta\,u_x \;-\; \nu\,u_{xx}",
            t2c={"r(x,t)": C_R, "u_t": C_T, "u_x": C_X, "u_{xx}": C_R,
                 r"u_\theta": C_U, r"\nu": GREY_A},
            font_size=52,
        )
        resid.move_to(0.55 * UP)

        # arrange() must run on the target copy, not inside .animate -- chaining
        # it onto an animate call re-lays-out the live mobject and the three
        # definitions end up on top of the equation
        grads_target = grads.copy()
        grads_target.arrange(RIGHT, buff=0.8)
        grads_target.set_width(min(grads_target.get_width(), 11.4))
        grads_target.to_edge(DOWN, buff=1.85)
        self.play(Transform(grads, grads_target), run_time=1.0)
        self.play(Write(resid), run_time=1.6)

        zero = Text("= 0 for the true solution, everywhere",
                    font_size=28, color=C_DIM)
        zero.next_to(resid, DOWN, buff=0.4)
        self.play(FadeIn(zero, shift=0.2 * UP))
        self.play(FlashAround(resid, color=C_R, run_time=1.6))
        self.wait(1.6)
        clear_scene(self)


# ============================================================================
# Scene 3 -- the three point sets the loss is built from
# ============================================================================

class WhereLossLives(Scene):
    def construct(self):
        head = section_title(
            "Three point sets, three loss terms",
            "and not a mesh in sight -- the points are just random draws",
        )
        head.to_edge(UP, buff=0.45)
        self.play(FadeIn(head, shift=0.3 * DOWN))

        # A box, not a pair of crossed axes: the domain is a region, and an
        # axis drawn at x=0 would run straight through the point cloud.
        cx, cy, w, h = -3.0, -0.25, 6.0, 3.7

        def dom(xv, tv):
            return np.array([cx + w * xv / 2, cy - h / 2 + h * tv, 0.0])

        box = Rectangle(w, h).move_to([cx, cy, 0]).set_stroke(GREY_B, 2)
        x_ticks = VGroup(*[
            Tex(f"{v:g}", font_size=24, color=GREY_B).move_to(
                dom(v, 0) + 0.32 * DOWN)
            for v in (-1, 0, 1)
        ])
        t_ticks = VGroup(*[
            Tex(f"{v:g}", font_size=24, color=GREY_B).move_to(
                dom(-1, v) + 0.38 * LEFT)
            for v in (0, 0.5, 1)
        ])
        x_lab = Tex("x", color=C_X).move_to(dom(0, 0) + 0.85 * DOWN)
        t_lab = Tex("t", color=C_T).move_to(dom(-1, 0.5) + 0.92 * LEFT)
        self.play(ShowCreation(box), FadeIn(x_ticks), FadeIn(t_ticks),
                  FadeIn(x_lab), FadeIn(t_lab))

        rng = np.random.default_rng(0)

        def scatter(xs, ts, color, radius=0.032):
            return VGroup(*[
                Dot(dom(xv, tv), radius=radius).set_fill(color, 0.85)
                    .set_stroke(width=0)
                for xv, tv in zip(xs, ts)
            ])

        n_f = 420                                   # 5,000 in the real script
        pts_f = scatter(rng.uniform(-1, 1, n_f), rng.uniform(0, 1, n_f), C_R)
        n_i = 46                                    # 300 in the real script
        pts_i = scatter(np.linspace(-1, 1, n_i), np.zeros(n_i), C_IC, 0.045)
        n_b = 22                                    # 300 in the real script
        tb = np.linspace(0, 1, n_b)
        pts_b = VGroup(
            scatter(-np.ones(n_b), tb, C_BC, 0.045),
            scatter(np.ones(n_b), tb, C_BC, 0.045),
        )

        terms = VGroup(
            VGroup(
                Tex(r"\mathbb{E}\big[\,r(x,t)^2\,\big]", color=C_R).scale(0.7),
                Text("PDE, anywhere inside", font_size=20, color=C_DIM),
            ).arrange(DOWN, buff=0.14, aligned_edge=LEFT),
            VGroup(
                Tex(r"\mathbb{E}\big[\,(u_\theta - u_0)^2\,\big]",
                    color=C_IC).scale(0.7),
                Text("initial condition, t = 0", font_size=20, color=C_DIM),
            ).arrange(DOWN, buff=0.14, aligned_edge=LEFT),
            VGroup(
                Tex(r"\mathbb{E}\big[\,u_\theta(\pm 1, t)^2\,\big]",
                    color=C_BC).scale(0.7),
                Text("boundaries, x = ±1", font_size=20, color=C_DIM),
            ).arrange(DOWN, buff=0.14, aligned_edge=LEFT),
        )
        terms.arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        terms.to_edge(RIGHT, buff=0.7).shift(0.35 * DOWN)

        for pts, term in ((pts_f, terms[0]), (pts_i, terms[1]),
                          (pts_b, terms[2])):
            self.play(
                FadeIn(pts, lag_ratio=0.004, run_time=1.1),
                FadeIn(term, shift=0.2 * LEFT, run_time=0.8),
            )
            self.wait(0.35)

        total = Tex(
            r"\mathcal{L} \;=\; \mathcal{L}_{pde} \;+\; \mathcal{L}_{ic}"
            r" \;+\; \mathcal{L}_{bc}",
            t2c={r"\mathcal{L}_{pde}": C_R, r"\mathcal{L}_{ic}": C_IC,
                 r"\mathcal{L}_{bc}": C_BC},
            font_size=40,
        )
        total.next_to(terms, DOWN, buff=0.5)
        # next_to centres on `terms`, which is narrower -- pin it to the margin
        total.set_x(FRAME_WIDTH / 2 - 0.4 - total.get_width() / 2)
        self.play(Write(total))

        caveat = Text(
            "equal weights work here; balancing these terms is THE\n"
            "practical pain point of PINNs on stiffer problems",
            font_size=21, color=C_DIM,
        )
        caveat.move_to([cx, -3.55, 0])
        self.play(FadeIn(caveat))
        self.wait(2.4)
        clear_scene(self)


# ============================================================================
# Scene 4 -- the recording: what training actually looked like
# ============================================================================

# Waterfall geometry (left panel)
WF_CX, WF_SX, WF_SU, WF_DY, WF_Y0 = -3.45, 2.25, 0.44, 1.0, -2.5
WF_TI = (0, 10, 20, 30, 40)           # snapshot rows -> t = 0, .25, .5, .75, 1

# Residual panel (top right)
RS_CX, RS_CY, RS_W, RS_H = 3.75, 1.65, 4.2, 2.0

# Loss panel (bottom right)
LS_CX, LS_CY, LS_W, LS_H = 3.75, -2.5, 4.2, 1.7


def _wf_points(x, u, row):
    """Map a profile u(x) onto its shelf in the waterfall."""
    pts = np.zeros((len(x), 3))
    pts[:, 0] = WF_CX + WF_SX * x
    pts[:, 1] = WF_Y0 + WF_DY * row + WF_SU * u
    return pts


class Training(Scene):
    def construct(self):
        rec = load_recording()
        if rec is None:
            missing_recording_card(self)
            return

        x = rec["x"]
        t_snap = rec["t_snap"]
        frames = rec["frames"]                     # (F, n_t, n_x)
        resid = rec["resid"]
        loss = rec["loss"]                         # (F, 4): total, pde, ic, bc
        step = rec["step"]
        stage = rec["stage"]
        u_ref = rec["u_ref"]
        n_frames = len(frames)
        i_switch = int(np.argmax(stage == 1)) if (stage == 1).any() else n_frames

        head = Text("What training actually looked like", font_size=32)
        head.to_edge(UP, buff=0.22)
        sub = Text("recorded from a real run of 01", font_size=19, color=C_DIM)
        sub.next_to(head, DOWN, buff=0.1)
        self.add(head, sub)

        # --- left panel: the solution profiles, stacked by time -------------
        shelves = VGroup()
        ref_curves = VGroup()
        labels = VGroup()
        for row, ti in enumerate(WF_TI):
            base = _wf_points(x, np.zeros_like(x), row)
            shelf = Line(base[0], base[-1]).set_stroke(GREY_D, 1, opacity=0.6)
            shelves.add(shelf)

            ref = VMobject().set_stroke(C_REF, 5.0, opacity=0.45)
            ref.set_points_as_corners(_wf_points(x, u_ref[ti], row))
            ref_curves.add(ref)

            lab = Tex(f"t={t_snap[ti]:.2f}", font_size=22, color=GREY_B)
            lab.move_to(base[0] + 0.55 * LEFT)
            labels.add(lab)

        pinn_curves = VGroup()
        for _ in WF_TI:
            pinn_curves.add(VMobject().set_stroke(C_U, 2.2))

        legend = VGroup(
            VGroup(Line(ORIGIN, 0.45 * RIGHT).set_stroke(C_REF, 5.0, opacity=0.45),
                   Text("spectral reference", font_size=19, color=GREY_B)
                   ).arrange(RIGHT, buff=0.15),
            VGroup(Line(ORIGIN, 0.45 * RIGHT).set_stroke(C_U, 2.2),
                   Text("PINN", font_size=19, color=C_U)
                   ).arrange(RIGHT, buff=0.15),
        ).arrange(RIGHT, buff=0.5)
        legend.move_to([WF_CX, WF_Y0 + WF_DY * (len(WF_TI) - 1) + 0.9, 0])

        self.add(shelves, labels, ref_curves, pinn_curves, legend)

        # --- top right: the PDE residual field, cooling toward zero ---------
        # Each cell is the MEAN |r| over the patch it covers, not a single
        # sampled point. Point sampling lands on zero-crossings of r and turns
        # a converged field into black speckle.
        n_gx, n_gt = 24, 12
        xb = np.linspace(0, len(x), n_gx + 1).astype(int)
        tb = np.linspace(0, len(t_snap), n_gt + 1).astype(int)
        res_cells = np.array([
            [[resid[f, tb[j]:tb[j + 1], xb[i]:xb[i + 1]].mean()
              for i in range(n_gx)]
             for j in range(n_gt)]
            for f in range(n_frames)
        ])

        res_dots = VGroup()
        cell_w = RS_W / n_gx
        cell_h = RS_H / n_gt
        for j in range(n_gt):
            for i in range(n_gx):
                pos = np.array([
                    RS_CX - RS_W / 2 + cell_w * (i + 0.5),
                    RS_CY - RS_H / 2 + cell_h * (j + 0.5),
                    0.0,
                ])
                cell = Rectangle(cell_w, cell_h).move_to(pos)
                cell.set_stroke(width=0).set_fill(BLACK, 1)
                res_dots.add(cell)
        res_frame = Rectangle(RS_W + 0.1, RS_H + 0.1)
        res_frame.move_to([RS_CX, RS_CY, 0]).set_stroke(GREY_D, 1.5)
        res_title = Text("PDE residual magnitude over the (x,t) domain",
                         font_size=19, color=GREY_A)
        res_title.next_to(res_frame, UP, buff=0.12)
        self.add(res_dots, res_frame, res_title)

        # log10 colour scale for the residual, fixed across the whole run so
        # the cooling is a real change and not a rescaling artefact
        r_lo, r_hi = -3.5, 0.5

        def res_color(v):
            a = np.clip((np.log10(v + 1e-12) - r_lo) / (r_hi - r_lo), 0, 1)
            if a < 0.5:
                return interpolate_color(BLACK, C_R, a / 0.5)
            return interpolate_color(C_R, YELLOW, (a - 0.5) / 0.5)

        # A legend for that scale, so "it got darker" is readable as a number.
        # The residual never reaches zero -- the final field is dim, not black,
        # and the bar is what lets you see that.
        bar_w, bar_h, n_bar = 2.6, 0.16, 40
        bar = VGroup()
        for i in range(n_bar):
            v = 10 ** (r_lo + (r_hi - r_lo) * (i + 0.5) / n_bar)
            seg = Rectangle(bar_w / n_bar, bar_h)
            seg.move_to([RS_CX - bar_w / 2 + bar_w * (i + 0.5) / n_bar,
                         RS_CY - RS_H / 2 - 0.42, 0.0])
            seg.set_stroke(width=0).set_fill(res_color(v), 1)
            bar.add(seg)
        bar_lo = Tex(f"10^{{{r_lo:g}}}", font_size=17, color=GREY_B)
        bar_lo.next_to(bar, LEFT, buff=0.16)
        bar_hi = Tex("10^{0.5}", font_size=17, color=GREY_B)
        bar_hi.next_to(bar, RIGHT, buff=0.16)
        self.add(bar, bar_lo, bar_hi)

        # --- bottom right: the real loss curve ------------------------------
        log_loss = np.log10(loss[:, 0])
        y_lo = float(np.floor(log_loss.min()))
        y_hi = float(np.ceil(log_loss.max()))

        def loss_pt(i, v):
            return np.array([
                LS_CX - LS_W / 2 + LS_W * (i / max(n_frames - 1, 1)),
                LS_CY - LS_H / 2 + LS_H * ((v - y_lo) / (y_hi - y_lo)),
                0.0,
            ])

        loss_pts = np.array([loss_pt(i, v) for i, v in enumerate(log_loss)])
        loss_frame = Rectangle(LS_W + 0.1, LS_H + 0.1)
        loss_frame.move_to([LS_CX, LS_CY, 0]).set_stroke(GREY_D, 1.5)
        loss_curve = VMobject().set_stroke(C_R, 3.0)
        loss_curve.set_points_as_corners(loss_pts[:2])
        y_ticks = VGroup()
        for v in range(int(y_lo), int(y_hi) + 1):
            lab = Tex(f"10^{{{v}}}", font_size=18, color=GREY_B)
            lab.move_to(loss_pt(0, v) + 0.42 * LEFT)
            y_ticks.add(lab)
        loss_title = Text("total loss", font_size=19, color=GREY_A)
        loss_title.next_to(loss_frame, UP, buff=0.12)

        switch_line = DashedLine(loss_pt(i_switch, y_lo), loss_pt(i_switch, y_hi))
        switch_line.set_stroke(GREY_B, 1.6, opacity=0.8)
        self.add(loss_frame, y_ticks, loss_title, switch_line, loss_curve)

        # --- readouts -------------------------------------------------------
        read_y = LS_CY + LS_H / 2 + 0.75
        stage_txt = Text("Adam", font_size=26, color=C_U)
        stage_txt.move_to([LS_CX - LS_W / 2 + 0.6, read_y, 0])
        step_lab = Text("step", font_size=20, color=GREY_A)
        step_num = Integer(0, font_size=24, color=WHITE)
        step_grp = VGroup(step_lab, step_num).arrange(RIGHT, buff=0.2)
        step_grp.move_to([LS_CX + 1.2, read_y, 0])
        self.add(stage_txt, step_grp)

        # --- drive everything from one tracker ------------------------------
        f_track = ValueTracker(0.0)

        def interp(arr, f):
            i0 = int(np.floor(f))
            i1 = min(i0 + 1, n_frames - 1)
            a = f - i0
            return (1 - a) * arr[i0] + a * arr[i1]

        def make_curve_updater(row, ti):
            def upd(mob):
                u = interp(frames[:, ti, :], f_track.get_value())
                mob.set_points_as_corners(_wf_points(x, u, row))
            return upd

        for row, ti in enumerate(WF_TI):
            pinn_curves[row].add_updater(make_curve_updater(row, ti))

        def upd_res(group):
            r = interp(res_cells, f_track.get_value())
            k = 0
            for j in range(n_gt):
                for i in range(n_gx):
                    group[k].set_fill(res_color(r[j, i]), 1)
                    k += 1
        res_dots.add_updater(upd_res)

        def upd_loss(mob):
            n = max(2, int(f_track.get_value()) + 1)
            mob.set_points_as_corners(loss_pts[:n])
        loss_curve.add_updater(upd_loss)

        def upd_stage(mob):
            want = "L-BFGS" if f_track.get_value() >= i_switch else "Adam"
            if mob.get_text() != want:
                new = Text(want, font_size=26, color=C_U)
                new.move_to(mob)
                mob.become(new)
        stage_txt.add_updater(upd_stage)

        step_num.add_updater(
            lambda m: m.set_value(int(interp(step.astype(float),
                                             f_track.get_value())))
        )

        self.wait(0.6)

        # frame 0 is the untrained network: a small tanh wiggle, nothing more
        note = Text("at initialisation: a small wiggle, and a residual that is\n"
                    "wrong everywhere",
                    font_size=22, color=C_DIM)
        note.move_to([WF_CX, WF_Y0 - 0.75, 0])
        self.play(FadeIn(note))
        self.wait(1.4)
        self.play(FadeOut(note))

        # Adam first -- it finds the right basin fast, then flattens out
        self.play(f_track.animate.set_value(i_switch),
                  run_time=11, rate_func=linear)
        n1 = Text("Adam has the shape, and is grinding",
                  font_size=22, color=C_DIM)
        n1.move_to([WF_CX, WF_Y0 - 0.75, 0])
        self.play(FadeIn(n1))
        self.wait(0.9)
        self.play(FadeOut(n1))

        # then L-BFGS, which is what actually makes a PINN accurate
        self.play(f_track.animate.set_value(n_frames - 1),
                  run_time=8, rate_func=linear)

        n2 = Text("L-BFGS: the second-order pass most tutorials skip",
                  font_size=22, color=C_U)
        n2.move_to([WF_CX, WF_Y0 - 0.75, 0])
        self.play(FadeIn(n2))
        self.wait(2.6)

        for m in (pinn_curves, res_dots, loss_curve, stage_txt, step_num):
            m.clear_updaters()
        clear_scene(self)


# ============================================================================
# Scene 5 -- the honest scoreboard
# ============================================================================

class Scoreboard(Scene):
    def construct(self):
        rec = load_recording()
        if rec is None:
            missing_recording_card(self)
            return

        train_time = float(rec["train_time"])
        err = float(rec["rel_l2"])

        head = Text("So how did it do?", font_size=40)
        head.to_edge(UP, buff=0.6)
        self.play(FadeIn(head, shift=0.3 * DOWN))

        def card(title, lines, color):
            body = VGroup(*[Text(s, font_size=26, color=c)
                            for s, c in lines])
            body.arrange(DOWN, buff=0.28)
            ttl = Text(title, font_size=30, color=color)
            grp = VGroup(ttl, body).arrange(DOWN, buff=0.42)
            # a fixed box for both cards -- letting SurroundingRectangle size
            # itself makes the two sides look weighted against each other
            rect = Rectangle(5.0, 3.3).move_to(grp)
            rect.set_stroke(color, 2)
            return VGroup(rect, grp)

        left = card("PINN", [
            (f"{train_time:.0f} s to train", WHITE),
            (f"rel-L2 error  {err:.2%}", WHITE),
            ("never saw the answer,", C_DIM),
            ("only the equation", C_DIM),
        ], C_U)
        right = card("spectral solver", [
            ("~0.14 s", WHITE),
            ("essentially exact", WHITE),
            ("pseudo-spectral, pure numpy", C_DIM),
            ("in common.py", C_DIM),
        ], C_REF)
        cards = VGroup(left, right).arrange(RIGHT, buff=1.2)
        cards.shift(0.3 * DOWN)

        self.play(FadeIn(left, shift=0.3 * RIGHT))
        self.wait(0.8)
        self.play(FadeIn(right, shift=0.3 * LEFT))
        self.wait(1.4)

        verdict = Text(
            f"For a plain forward problem, the PINN loses by ~{train_time/0.14:.0f}x.",
            font_size=28, color=C_R,
        )
        verdict.next_to(cards, DOWN, buff=0.6)
        self.play(FadeIn(verdict, shift=0.2 * UP))
        self.wait(1.8)

        why = VGroup(
            Text("PINNs earn their keep where the classical solver cannot be written down:",
                 font_size=24),
            Text("inverse problems  ·  sparse noisy data + physics  ·  "
                 "irregular geometry  ·  high dimensions",
                 font_size=23, color=C_U),
        ).arrange(DOWN, buff=0.25)
        why.next_to(verdict, DOWN, buff=0.5)
        self.play(FadeOut(cards), FadeOut(verdict))
        self.play(why.animate.move_to(0.2 * DOWN))
        self.wait(3)
        clear_scene(self)


# ============================================================================
# Everything, back to back
# ============================================================================

class PINNFull(Scene):
    def construct(self):
        for cls in (TheNetwork, AutogradTrick, WhereLossLives,
                    Training, Scoreboard):
            cls.construct(self)
            self.wait(0.4)
