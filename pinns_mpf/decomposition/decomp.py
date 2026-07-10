"""Domain decomposition for PINNs-MPF — multi-NN spatial subdivision with continuity.

Restores the original framework's load-bearing feature, upgraded: the spatial domain
[0,L]^2 is split into nx*ny OVERLAPPING sub-boxes, each with its own WORKER network
(MLP). Workers are trained JOINTLY on GPU (one summed loss + one optimizer over all
worker params) instead of the legacy CPU Master/Worker sync. Continuity across
sub-domains is enforced by an interfacial loss on the overlap halos (each pair of
neighbours must agree there) — the direct analogue of the legacy interfacial-BC scheme.

Why it fixes curvature-driven motion: each worker sees a SMALL region where the
interface is locally near-flat, so it can learn the local recession without the global
"grow to lower curvature" residual basin that defeats a single network.

Single-phase 2D here (B2/B4); multi-phase (B5) reuses the same machinery with softmax
workers. Time-marching wraps this via ``decomp_time_march``.
"""
from __future__ import annotations

import time

import numpy as np
import tensorflow as tf

from ..config import DTYPE
from ..model.mlp import MLP
from ..optimizers import lbfgs
from ..physics.single_phase_2d import pde_residual_2d, profile_residual_2d


def _phi_tau(worker, x, y, t):
    """d(phi)/d(tau) at points (x,y,t) for one worker -- the local interface velocity.
    Used by the temporal SLOPE-continuity loss to pin the initial recession rate at tau=0."""
    with tf.GradientTape() as tp:
        tp.watch(t)
        phi = worker(tf.concat([x, y, t], axis=1))
    return tp.gradient(phi, t)


def make_boxes(L, nx, ny, overlap, x_edges=None, y_edges=None):
    """nx*ny boxes tiling [0,L]^2; each has a 'core' (disjoint) and 'ext' (core+overlap halo).

    ``x_edges``/``y_edges`` (optional) give NON-UNIFORM core boundaries instead of an even split,
    e.g. [0,12,52,64] for a center-large 3x3 whose central core [12,52] contains the whole shrinking
    grain for R<=20 (so the small grain is NOT split across the seam cross). nx/ny are then derived
    from the edges."""
    xe = np.asarray(x_edges, dtype=float) if x_edges is not None else np.linspace(0.0, L, nx + 1)
    ye = np.asarray(y_edges, dtype=float) if y_edges is not None else np.linspace(0.0, L, ny + 1)
    nx = len(xe) - 1
    ny = len(ye) - 1
    boxes = []
    for i in range(nx):
        for j in range(ny):
            cx0, cx1, cy0, cy1 = xe[i], xe[i + 1], ye[j], ye[j + 1]
            ext = (max(0.0, cx0 - overlap), min(L, cx1 + overlap),
                   max(0.0, cy0 - overlap), min(L, cy1 + overlap))
            boxes.append({"i": i, "j": j, "core": (cx0, cx1, cy0, cy1), "ext": ext})
    return boxes


def neighbor_pairs(boxes, nx, ny):
    """Adjacent (right/up) box index pairs, for continuity on their shared overlap."""
    idx = {(b["i"], b["j"]): k for k, b in enumerate(boxes)}
    pairs = []
    for (i, j), k in idx.items():
        if (i + 1, j) in idx:
            pairs.append((k, idx[(i + 1, j)]))
        if (i, j + 1) in idx:
            pairs.append((k, idx[(i, j + 1)]))
    return pairs


class MultiNN(tf.Module):
    """Ensemble of worker MLPs (one per box). tf.Module aggregates trainable_variables."""

    def __init__(self, boxes, hidden, out_dim, dt, output_activation="sigmoid",
                 periodic_dims=None, domain=None, seed=1234):
        super().__init__()
        self.boxes = boxes
        self.out_dim = out_dim
        self.dt = dt
        self.workers = []
        pdims = set(periodic_dims or [])
        for k, b in enumerate(boxes):
            ex0, ex1, ey0, ey1 = b["ext"]
            lb = [ex0, ey0, 0.0]; ub = [ex1, ey1, dt]
            # Periodic spatial dims must embed with the FULL-domain period L (sin/cos
            # of 2*pi*x/L), not the box extent -- otherwise each worker would be
            # "periodic" over its own box, which is wrong. ``domain`` = L.
            if domain is not None:
                for d in (0, 1):
                    if d in pdims:
                        lb[d] = 0.0; ub[d] = float(domain)
            self.workers.append(MLP([3] + list(hidden) + [out_dim],
                                    lb=lb, ub=ub,
                                    output_activation=output_activation,
                                    periodic_dims=periodic_dims, seed=seed + k))

    def predict_grid(self, xs, tau):
        """Assemble the full field at local time tau by routing each grid cell to the
        worker whose CORE box contains it. Returns [N,N] (out_dim=1) or [P,N,N]."""
        N = len(xs)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        out = np.zeros((N, N)) if self.out_dim == 1 else np.zeros((self.out_dim, N, N))
        for k, b in enumerate(self.boxes):
            cx0, cx1, cy0, cy1 = b["core"]
            m = (X >= cx0) & (X < cx1) & (Y >= cy0) & (Y < cy1)
            ii, jj = np.where(m)
            if ii.size == 0:
                continue
            pts = np.column_stack([X[ii, jj], Y[ii, jj], np.full(ii.size, tau)])
            pred = self.workers[k](tf.constant(pts, dtype=DTYPE)).numpy()
            if self.out_dim == 1:
                out[ii, jj] = pred[:, 0]
            else:
                for c in range(self.out_dim):
                    out[c, ii, jj] = pred[:, c]
        return out

    # flat params over ALL workers (for TFP L-BFGS over the whole ensemble)
    def get_flat_params(self):
        return tf.concat([tf.reshape(v, [-1]) for v in self.trainable_variables], axis=0)

    def set_flat_params(self, flat):
        flat = tf.convert_to_tensor(flat, dtype=DTYPE)
        i = 0
        for v in self.trainable_variables:
            n = int(np.prod(v.shape))
            v.assign(tf.reshape(flat[i:i + n], v.shape))
            i += n


def _deep_bulk_mask(phi_k, xs, dn_margin=0.0, core_margin=None):
    """Cells eligible for denoising.

    The raw phi threshold marks cells that are bulk-like at the beginning of the
    time window. A positive ``dn_margin`` further requires distance from the
    current interface, avoiding pins on cells that the interface can plausibly
    consume before tau=dt.

    FUTURE-SAFE mode (``core_margin`` given): treat matrix and grain-core
    asymmetrically -- the MATRIX (phi<0.005) is not the shrinking object, so it is
    pinned strongly (always); the shrinking CORE (phi>0.995) is pinned only beyond
    ``core_margin`` from the interface (a future-safe distance ~ the displacement
    the front can travel this interval), so core pins never become a hidden
    obstacle to shrinkage. Targets late-time mass over-retention.
    """
    matrix = phi_k < 0.005
    core = phi_k > 0.995
    if core_margin is not None:
        interface = (phi_k > 0.02) & (phi_k < 0.98)
        if not np.any(interface):
            return matrix | core
        from scipy import ndimage
        dx = float(xs[1] - xs[0])
        dist = ndimage.distance_transform_edt(~interface) * dx
        return matrix | (core & (dist >= core_margin))

    deep = matrix | core
    if dn_margin <= 0.0:
        return deep

    interface = (phi_k > 0.02) & (phi_k < 0.98)
    if not np.any(interface):
        return deep

    from scipy import ndimage  # local import keeps NumPy-only users cheap

    dx = float(xs[1] - xs[0])
    dist_to_interface = ndimage.distance_transform_edt(~interface) * dx
    return deep & (dist_to_interface >= dn_margin)


def build_batch(multinn, phi_k, xs, dt, n_f, n_ic, n_dn, n_cont, eta, seed,
                dn_margin=0.0, v_target=None, n_sl=0, trend_target=None, n_tr=0,
                pde_band=True, pde_bulk_frac=0.0, core_margin=None):
    """Per-box training data as LISTS of tf tensors (variable sizes per box):
      - Xf: PDE collocation, ONLY near the interface (global band routed to the box whose
        CORE contains each point -> pure-bulk boxes get ~none, avoiding the bulk-grow);
      - Xic/Phic: IC at tau=0 sampled in the ext box (interface-concentrated);
      - Xdn/Phidn: DENOISING on safe deep-bulk cells -> pin phi=0/1 for all tau;
      - Xc: continuity points in each neighbour overlap;
      - Xsl/Vsl: SLOPE-continuity points (interface band, tau=0) with target initial velocity
        ``v_target`` (= PDE RHS on the carried-forward IC). Pins d(phi)/d(tau)|tau=0 so the net
        starts each interval receding at the curvature velocity instead of relaxing to a static
        (freeze) or growing profile. Active only if v_target is given and n_sl>0.
    Returns (batch dict of per-box lists, neighbour pairs)."""
    rng = np.random.default_rng(seed)
    dx = xs[1] - xs[0]; L = xs[-1] + dx
    Xg, Yg = np.meshgrid(xs, xs, indexing="ij")
    interface = (phi_k > 0.02) & (phi_k < 0.98)
    deep = _deep_bulk_mask(phi_k, xs, dn_margin=dn_margin, core_margin=core_margin)
    ii_i, jj_i = np.where(interface)
    boxes = multinn.boxes
    EMPTY = tf.constant(np.empty((0, 3)), DTYPE); EMPTY1 = tf.constant(np.empty((0, 1)), DTYPE)
    # collocation, routed to boxes by CORE. pde_band=True -> interface band only (from-scratch
    # default, needs denoising to hold the bulk); pde_band=False -> uniform over the whole domain
    # (LEGACY-faithful: PDE everywhere, bulk held by the PDE itself + tiny dt, no denoising).
    def _band_pts(m):
        """m interface-band collocation points (+-1.5 eta around the contour)."""
        sb = rng.integers(0, ii_i.size, m)
        xb = np.clip(xs[ii_i[sb]] + rng.uniform(-1.5 * eta, 1.5 * eta, m), 0, L - 1e-9)
        yb = np.clip(xs[jj_i[sb]] + rng.uniform(-1.5 * eta, 1.5 * eta, m), 0, L - 1e-9)
        return np.column_stack([xb, yb, rng.uniform(0, dt, m)])

    def _bulk_pts(m):
        """m uniform-over-domain collocation points (for the projected-RHS bulk term)."""
        return np.column_stack([rng.uniform(0, L - 1e-9, m), rng.uniform(0, L - 1e-9, m),
                                rng.uniform(0, dt, m)])

    if not pde_band:
        Xf_all = _bulk_pts(n_f)
    elif ii_i.size:
        # Hybrid: front resolution (band) + bulk coverage so the PROJECTED RHS can make the
        # saturated bulk inert (without bulk collocation the projection never fires -> denoising
        # would remain the only bulk mechanism). pde_bulk_frac in [0,1] splits n_f.
        n_bulk = int(round(n_f * float(pde_bulk_frac)))
        n_band = n_f - n_bulk
        parts = [_band_pts(n_band)] if n_band > 0 else []
        if n_bulk > 0:
            parts.append(_bulk_pts(n_bulk))
        Xf_all = np.concatenate(parts, axis=0) if parts else np.empty((0, 3))
    else:
        Xf_all = np.empty((0, 3))
    Xf, Xic, Phic, Xdn, Phidn, Xsl, Vsl, Xtr, Phitr = [], [], [], [], [], [], [], [], []
    v_target = None if v_target is None else np.asarray(v_target)
    trend_target = None if trend_target is None else np.asarray(trend_target)
    for b in boxes:
        cx0, cx1, cy0, cy1 = b["core"]; ex0, ex1, ey0, ey1 = b["ext"]
        if Xf_all.shape[0]:
            m = ((Xf_all[:, 0] >= cx0) & (Xf_all[:, 0] < cx1)
                 & (Xf_all[:, 1] >= cy0) & (Xf_all[:, 1] < cy1))
            Xf.append(tf.constant(Xf_all[m], DTYPE) if m.any() else EMPTY)
        else:
            Xf.append(EMPTY)
        inbox = (Xg >= ex0) & (Xg < ex1) & (Yg >= ey0) & (Yg < ey1)
        im = inbox & interface
        ai, aj = np.where(im if im.any() else inbox)
        s = rng.integers(0, ai.size, n_ic)
        Xic.append(tf.constant(np.column_stack([xs[ai[s]], xs[aj[s]], np.zeros(n_ic)]), DTYPE))
        Phic.append(tf.constant(phi_k[ai[s], aj[s]].reshape(-1, 1), DTYPE))
        # SLOPE-continuity: interface-band points at tau=0, target = initial velocity v_target
        if v_target is not None and n_sl > 0 and im.any():
            bi, bj = np.where(im); si = rng.integers(0, bi.size, n_sl)
            Xsl.append(tf.constant(np.column_stack([xs[bi[si]], xs[bj[si]], np.zeros(n_sl)]), DTYPE))
            Vsl.append(tf.constant(v_target[bi[si], bj[si]].reshape(-1, 1), DTYPE))
        else:
            Xsl.append(EMPTY); Vsl.append(EMPTY1)
        # TREND-endpoint: interface-band points at tau=dt, target = trend-extrapolated field
        if trend_target is not None and n_tr > 0 and im.any():
            ti, tj = np.where(im); st = rng.integers(0, ti.size, n_tr)
            Xtr.append(tf.constant(np.column_stack([xs[ti[st]], xs[tj[st]], np.full(n_tr, dt)]), DTYPE))
            Phitr.append(tf.constant(trend_target[ti[st], tj[st]].reshape(-1, 1), DTYPE))
        else:
            Xtr.append(EMPTY); Phitr.append(EMPTY1)
        dm = inbox & deep
        if dm.any():
            di, dj = np.where(dm); sd = rng.integers(0, di.size, n_dn)
            Xdn.append(tf.constant(np.column_stack([xs[di[sd]], xs[dj[sd]],
                                                    rng.uniform(0, dt, n_dn)]), DTYPE))
            Phidn.append(tf.constant(phi_k[di[sd], dj[sd]].reshape(-1, 1), DTYPE))
        else:
            Xdn.append(EMPTY); Phidn.append(EMPTY1)
    pairs = neighbor_pairs(boxes, max(b["i"] for b in boxes) + 1, max(b["j"] for b in boxes) + 1)
    cont = []
    for (a, c) in pairs:
        ax0, ax1, ay0, ay1 = boxes[a]["ext"]; bx0, bx1, by0, by1 = boxes[c]["ext"]
        ox0, ox1 = max(ax0, bx0), min(ax1, bx1); oy0, oy1 = max(ay0, by0), min(ay1, by1)
        if ox1 > ox0 and oy1 > oy0:
            xc = rng.uniform(ox0, ox1, n_cont); yc = rng.uniform(oy0, oy1, n_cont)
            cont.append(tf.constant(np.column_stack([xc, yc, rng.uniform(0, dt, n_cont)]), DTYPE))
        else:
            cont.append(EMPTY)
    return {"Xf": Xf, "Xic": Xic, "Phic": Phic, "Xdn": Xdn, "Phidn": Phidn, "Xc": cont,
            "Xsl": Xsl, "Vsl": Vsl, "Xtr": Xtr, "Phitr": Phitr}, pairs


def make_loss(multinn, p, pairs, w_pde=1.0, w_ic=2.0, w_dn=1.0, w_cont=1.0, w_slope=0.0, w_trend=0.0,
              pde_project=False, pde_eps=1e-3, w_profile=0.0):
    """Joint decomposition loss with MUTABLE term weights W (for optional GradNorm).
    Returns (loss, set_batch, W, terms): terms() = unweighted per-term means
    (pde/ic/dn/cont/slope/trend, each averaged over the contributing boxes/pairs; empty per-box
    tensors skipped); loss() applies the current W. W values are tf.Variables so GradNorm can
    reassign them online. ``slope`` = temporal velocity-continuity at tau=0 (Xsl/Vsl); ``trend`` =
    endpoint value-continuation at tau=dt toward the trend-extrapolated field (Xtr/Phitr).
    ``pde_project`` uses the bound-aware effective RHS (matches the clipped reference; lets the
    bulk be physically inert so denoising is only a light safety control)."""
    B = len(multinn.workers)
    batch_ref = {}
    W = {"pde": tf.Variable(w_pde, dtype=DTYPE), "ic": tf.Variable(w_ic, dtype=DTYPE),
         "dn": tf.Variable(w_dn, dtype=DTYPE), "cont": tf.Variable(w_cont, dtype=DTYPE),
         "slope": tf.Variable(w_slope, dtype=DTYPE), "trend": tf.Variable(w_trend, dtype=DTYPE),
         "profile": tf.Variable(w_profile, dtype=DTYPE)}
    use_profile = float(w_profile) > 0.0

    def terms():
        pde = tf.constant(0.0, DTYPE); ic = tf.constant(0.0, DTYPE); dn = tf.constant(0.0, DTYPE)
        sl = tf.constant(0.0, DTYPE); tr = tf.constant(0.0, DTYPE); prof = tf.constant(0.0, DTYPE)
        npde = 0; ndn = 0; nsl = 0; ntr = 0; nprof = 0
        for k in range(B):
            wk = multinn.workers[k]
            Xf = batch_ref["Xf"][k]
            if Xf.shape[0] > 0:
                r = pde_residual_2d(wk, Xf[:, 0:1], Xf[:, 1:2], Xf[:, 2:3], p,
                                    project=pde_project, eps=pde_eps)
                pde += tf.reduce_mean(tf.square(r)); npde += 1
                if use_profile:
                    pr = profile_residual_2d(wk, Xf[:, 0:1], Xf[:, 1:2], Xf[:, 2:3], p)
                    prof += tf.reduce_mean(tf.square(pr)); nprof += 1
            ic += tf.reduce_mean(tf.square(wk(batch_ref["Xic"][k]) - batch_ref["Phic"][k]))
            Xdn = batch_ref["Xdn"][k]
            if Xdn.shape[0] > 0:
                dn += tf.reduce_mean(tf.square(wk(Xdn) - batch_ref["Phidn"][k])); ndn += 1
            Xsl = batch_ref.get("Xsl", [None] * B)[k]
            if Xsl is not None and Xsl.shape[0] > 0:
                vt = _phi_tau(wk, Xsl[:, 0:1], Xsl[:, 1:2], Xsl[:, 2:3])
                sl += tf.reduce_mean(tf.square(vt - batch_ref["Vsl"][k])); nsl += 1
            Xtr = batch_ref.get("Xtr", [None] * B)[k]
            if Xtr is not None and Xtr.shape[0] > 0:
                tr += tf.reduce_mean(tf.square(wk(Xtr) - batch_ref["Phitr"][k])); ntr += 1
        cont = tf.constant(0.0, DTYPE); nc = 0
        for pi, (a, c) in enumerate(pairs):
            Xc = batch_ref["Xc"][pi]
            if Xc.shape[0] > 0:
                cont += tf.reduce_mean(tf.square(multinn.workers[a](Xc) - multinn.workers[c](Xc)))
                nc += 1
        return {"pde": pde / max(1, npde), "ic": ic / B, "dn": dn / max(1, ndn),
                "cont": cont / max(1, nc), "slope": sl / max(1, nsl), "trend": tr / max(1, ntr),
                "profile": prof / max(1, nprof)}

    def loss():
        t = terms()
        total = (W["pde"] * t["pde"] + W["ic"] * t["ic"] + W["dn"] * t["dn"]
                 + W["cont"] * t["cont"] + W["slope"] * t["slope"] + W["trend"] * t["trend"]
                 + W["profile"] * t["profile"])
        return total, {"total": total, **t}

    def set_batch(batch):
        batch_ref.clear(); batch_ref.update(batch)

    return loss, set_batch, W, terms


def decomp_fit_interval(multinn, loss, adam_steps, lbfgs_iters, lr=1e-3, max_restarts=2,
                        gradnorm=False, terms=None, W=None, gn_every=40,
                        num_correction_pairs=50, pde_eager=False,
                        nonpde_loss=None, pde_grads_fn=None,
                        cycles=1, adam_per_cycle=None, loss_thresh=None, log_cycles=False,
                        restore_best_cycle=False):
    """Adam (+optional GradNorm reweighting every gn_every steps) then TFP L-BFGS.
    GradNorm: w_k <- mean_grad_norm / ||grad of term k||, so all terms contribute equally.
    ``num_correction_pairs`` caps the L-BFGS history (memory ~ 2*pairs*n_params); reduce
    it for large/dense ensembles (e.g. 16 workers x 6x128) to fit float64 on an 8 GB GPU.

    CYCLING (paper-faithful, opt-in — default cycles=1 reproduces the legacy single pass):
    ``cycles>1`` repeats [Adam(adam_per_cycle) -> L-BFGS(lbfgs_iters)] up to ``cycles`` times,
    breaking early when the total (weighted) loss falls below ``loss_thresh``. The original
    PINNs-MPF trainer alternated Adam and L-BFGS-B and advanced a time window only once the
    loss dropped below ~7e-5; a single Adam->L-BFGS pass plateaus far above that (the driving-
    force B3 diamond artifact). ``adam_per_cycle`` (default = adam_steps) is the Adam step count
    PER cycle. With cycles=1 and loss_thresh=None the behaviour is byte-identical to before.

    ``restore_best_cycle`` (opt-in, default False): while cycling, snapshot the trainable weights
    of the cycle with the lowest total loss; after the final cycle, restore that snapshot if it
    beats the final cycle's loss. Fixes the "keep last cycle" divergence, where a window that
    reaches a low loss mid-cycling then drifts up would otherwise end on the worse final weights
    (the Phase-B W5 case). Default False leaves every existing run/option byte-identical.

    pde_eager=True + nonpde_loss + pde_grads_fn:
      Gradient accumulation mode for large float64 nets on memory-constrained GPUs.
      The Adam step splits into:
        1. One GradientTape over nonpde_loss() (IC + DN + cont + PBC — no Hessian, fast).
        2. pde_grads_fn() accumulates PDE gradients via one tape per worker-chunk so
           peak GPU memory = one chunk's pfor intermediates (~21 MB for 64×6, 300 pts).
      Eliminates the 4 GB outer-tape problem from holding all workers × chunks at once.
    """
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    vars_ = multinn.trainable_variables

    if pde_eager and nonpde_loss is not None and pde_grads_fn is not None:
        def astep():
            # Non-PDE: one fast tape (IC + DN + cont + PBC, no batch_jacobian)
            with tf.GradientTape() as t_np:
                np_total = nonpde_loss()
            np_grads = t_np.gradient(np_total, vars_)
            # PDE: gradient accumulation, one tape per worker-chunk
            pde_grads = pde_grads_fn()
            # Combine: sum non-PDE and PDE gradient contributions
            combined = [(p + (n if n is not None else tf.zeros_like(v)))
                        for p, n, v in zip(pde_grads, np_grads, vars_)]
            opt.apply_gradients(zip(combined, vars_))
    else:
        def _astep_body():
            with tf.GradientTape() as t:
                total, _ = loss()
            opt.apply_gradients(zip(t.gradient(total, vars_), vars_))
        astep = _astep_body if pde_eager else tf.function(_astep_body)

    def gn_update():
        with tf.GradientTape(persistent=True) as t:
            tm = terms()
        norms = {}
        for k in ("pde", "ic", "dn", "cont"):
            gk = [g for g in t.gradient(tm[k], vars_) if g is not None]
            norms[k] = float(tf.linalg.global_norm(gk).numpy()) if gk else 1.0
        del t
        mean_n = sum(norms.values()) / 4.0
        for k in ("pde", "ic", "dn", "cont"):
            W[k].assign(float(np.clip(mean_n / (norms[k] + 1e-12), 0.05, 200.0)))

    per_cycle = adam_per_cycle if adam_per_cycle is not None else adam_steps
    n_cycles = max(1, int(cycles))
    gstep = 0                                    # global Adam step (for GradNorm cadence)
    best_loss = None                             # best-cycle restore (opt-in): argmin total loss
    best_params = None                           #   host snapshot of that cycle's weights
    for c in range(n_cycles):
        for _ in range(per_cycle):
            if gradnorm and terms is not None and W is not None and gstep % gn_every == 0:
                gn_update()
            astep()
            gstep += 1
        if lbfgs_iters > 0:
            lbfgs.minimize(multinn, lambda: loss()[0], max_iterations=lbfgs_iters,
                           max_restarts=max_restarts, num_correction_pairs=num_correction_pairs)
        # End-of-cycle total loss: computed if the early-break gate OR best-cycle restore needs it.
        cur = float(loss()[0]) if (loss_thresh is not None or restore_best_cycle) else None
        if restore_best_cycle and (best_loss is None or cur < best_loss):
            best_loss = cur                      # keep the best cycle's weights (host copy)
            best_params = [v.numpy() for v in vars_]
        if loss_thresh is not None:
            if log_cycles:
                print(f"    [cycle {c + 1}/{n_cycles}] total_loss={cur:.3e} "
                      f"(thresh={loss_thresh:.1e})")
            if cur < loss_thresh:
                break
    # Opt-in: restore the best-loss cycle if it beats the final cycle (fixes keep-last divergence).
    if restore_best_cycle and best_params is not None:
        final_loss = float(loss()[0])
        if best_loss < final_loss:
            for v, pbest in zip(vars_, best_params):
                v.assign(pbest)
            if log_cycles:
                print(f"    [best-cycle restore] restored loss={best_loss:.3e} "
                      f"(final cycle was {final_loss:.3e})")
        elif log_cycles:
            print(f"    [best-cycle restore] no-op: final cycle {final_loss:.3e} "
                  f"already best ({best_loss:.3e})")


def decomp_time_march(multinn, p, pairs, set_batch, build_batch_fn, xs, dt, ref_grids,
                      adam_steps=300, lbfgs_iters=200, lr=1e-3, weights=(1.0, 1.0, 5.0),
                      log=True):
    loss, _set = set_batch  # set_batch is (loss, set_batch_fn) tuple
    n_intervals = len(ref_grids) - 1
    phi_k = np.asarray(ref_grids[0])
    traj, mses = [phi_k], []
    for k in range(n_intervals):
        batch, _ = build_batch_fn(phi_k, k)
        _set(batch)
        decomp_fit_interval(multinn, loss, adam_steps, lbfgs_iters, lr=lr)
        phi_next = np.asarray(multinn.predict_grid(xs, dt))
        mse = float(np.mean((phi_next - np.asarray(ref_grids[k + 1])) ** 2))
        traj.append(phi_next); mses.append(mse)
        if log:
            print(f"  [interval {k + 1}/{n_intervals}] t={(k + 1) * dt:.2f} mse_vs_ref={mse:.3e}")
        phi_k = phi_next
    return traj, mses
