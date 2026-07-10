"""Multiphase domain decomposition for B5 triple junction — 4-phase MPF.

Adapts decomp.py machinery to a 4-phase field (phi_k shape [4, N, N]):

  Architecture: Upgraded — one 4-output (logit) worker per spatial batch.
    Softmax converts logits -> (phi_1,...,phi_4) with sum=1 by construction.
    This replaces the paper's 16 phase-specific workers with 4 spatial workers,
    each carrying all 4 phases simultaneously. Phase coupling is preserved through
    the shared MPF PDE residual in pde_residual_multiphase.

  Differences from single-phase decomp.py:
    - phi_k: [4, N, N] (4 phases, periodic grid)
    - Interface mask: pixels where NO phase is dominant (max_alpha phi_alpha < thresh)
    - IC/denoising targets: 4-component vectors
    - Continuity: compares all 4 phases between neighbor workers
    - PDE loss: uses pde_residual_multiphase (softmax + batch_jacobian)
    - Phase-sum: enforced by architecture (no explicit loss term)

  Discrete loss activation (paper-faithful):
    - first window: w_pde + w_ic only
    - subsequent windows: add w_dn + w_cont

Usage:
    boxes   = make_boxes(L, nx=2, ny=2, overlap=0.1)
    multinn = MultiNN(boxes, hidden=(128,)*6, out_dim=4, dt=dt_window,
                      output_activation='linear', periodic_dims=[0,1], domain=L)
    pairs   = neighbor_pairs(boxes, 2, 2)
    batch, pairs = build_batch_multiphase(multinn, phi_k, xs, dt_window, ...)
    loss, set_batch, W, terms = make_loss_multiphase(multinn, p, pairs, ...)
    set_batch(batch)
    decomp_fit_interval(multinn, loss, adam_steps, lbfgs_iters)
    phi_next = predict_grid_multiphase(multinn, xs, tau=dt_window)
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from ..config import DTYPE
from ..physics.multiphase import pde_residual_multiphase, phases_from_logits
from .decomp import make_boxes, neighbor_pairs, MultiNN, decomp_fit_interval  # noqa: F401

NPHASES = 4


# ── grid helpers ─────────────────────────────────────────────────────────────

def predict_grid_multiphase(multinn: MultiNN, xs, tau: float):
    """Assemble full [4, N, N] field at local time tau, applying softmax to logits.

    Each grid cell is routed to the worker whose CORE box contains it.
    """
    N = len(xs)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    out = np.zeros((NPHASES, N, N), dtype=np.float64)
    for k, b in enumerate(multinn.boxes):
        cx0, cx1, cy0, cy1 = b["core"]
        m = (X >= cx0) & (X < cx1) & (Y >= cy0) & (Y < cy1)
        ii, jj = np.where(m)
        if ii.size == 0:
            continue
        pts = np.column_stack([X[ii, jj], Y[ii, jj], np.full(ii.size, tau)])
        logits = multinn.workers[k](tf.constant(pts, dtype=DTYPE))
        phi = tf.nn.softmax(logits, axis=1).numpy()   # [m, 4]
        for c in range(NPHASES):
            out[c, ii, jj] = phi[:, c]
    return out                                         # [4, N, N]


# ── batch building ────────────────────────────────────────────────────────────

def _interface_mask(phi_k, thresh=0.95):
    """Pixels where NO phase is fully dominant — the diffuse interface region."""
    return phi_k.max(axis=0) < thresh                 # [N, N]


def _bulk_target(phi_k):
    """4-phase target for deep-bulk pixels: one-hot of the dominant phase."""
    dom = phi_k.argmax(axis=0)                        # [N, N]
    target = np.zeros_like(phi_k)                     # [4, N, N]
    N = phi_k.shape[-1]
    ii, jj = np.mgrid[:N, :N]
    target[dom, ii, jj] = 1.0
    return target                                      # [4, N, N] one-hot


def build_batch_multiphase(multinn: MultiNN, phi_k, xs, dt, n_f, n_ic, n_dn, n_cont,
                            eta, seed, pde_bulk_frac=0.2, interface_thresh=0.95,
                            n_pbc=None):
    """Per-box training data for 4-phase MPF decomposition.

    phi_k : [4, N, N] float64 — full phase field at the start of this window.
    Returns (batch dict of per-box tf.constant lists, neighbour pairs).

    Batch keys:
      Xf    : [m_f, 3]  PDE collocation (interface band + bulk fraction)
      Xic   : [n_ic, 3] IC points at tau=0
      Phic  : [n_ic, 4] IC targets (4 phase values)
      Xdn   : [n_dn, 3] denoising (deep bulk)
      Phidn : [n_dn, 4] denoising targets (one-hot of dominant phase)
      Xc    : [n_cont, 3] continuity points in overlap halos (per pair)
    """
    rng = np.random.default_rng(seed)
    N = len(xs); dx = float(xs[1] - xs[0]); L = float(xs[-1] + dx)

    Xg, Yg = np.meshgrid(xs, xs, indexing="ij")   # [N, N]
    iface = _interface_mask(phi_k, interface_thresh)  # [N, N]
    bulk_target = _bulk_target(phi_k)              # [4, N, N]
    deep_bulk = phi_k.max(axis=0) > 0.99          # [N, N]

    ii_i, jj_i = np.where(iface)                  # interface pixel indices

    EMPTY   = tf.constant(np.empty((0, 3)), DTYPE)
    EMPTY4  = tf.constant(np.empty((0, 4)), DTYPE)

    # --- global interface-band + bulk PDE collocation ---
    if ii_i.size:
        n_band = int(n_f * (1 - pde_bulk_frac))
        n_bulk = n_f - n_band
        sb = rng.integers(0, ii_i.size, n_band)
        xb = np.clip(xs[ii_i[sb]] + rng.uniform(-1.5*eta, 1.5*eta, n_band), 0, L-1e-9)
        yb = np.clip(xs[jj_i[sb]] + rng.uniform(-1.5*eta, 1.5*eta, n_band), 0, L-1e-9)
        tb = rng.uniform(0, dt, n_band)
        Xf_band = np.column_stack([xb, yb, tb])
        Xf_bulk = np.column_stack([rng.uniform(0, L-1e-9, n_bulk),
                                   rng.uniform(0, L-1e-9, n_bulk),
                                   rng.uniform(0, dt, n_bulk)])
        Xf_all = np.vstack([Xf_band, Xf_bulk])
    else:
        Xf_all = np.column_stack([rng.uniform(0, L-1e-9, n_f),
                                  rng.uniform(0, L-1e-9, n_f),
                                  rng.uniform(0, dt, n_f)])

    Xf_list, Xic_list, Phic_list, Xdn_list, Phidn_list = [], [], [], [], []

    for b in multinn.boxes:
        cx0, cx1, cy0, cy1 = b["core"]
        ex0, ex1, ey0, ey1 = b["ext"]

        # PDE: route global band to this box's core
        m_f = ((Xf_all[:,0] >= cx0) & (Xf_all[:,0] < cx1)
              & (Xf_all[:,1] >= cy0) & (Xf_all[:,1] < cy1))
        Xf_list.append(tf.constant(Xf_all[m_f], DTYPE) if m_f.any() else EMPTY)

        # IC: interface-concentrated in extended box
        inbox_ext = (Xg >= ex0) & (Xg < ex1) & (Yg >= ey0) & (Yg < ey1)
        im = inbox_ext & iface
        src = im if im.any() else inbox_ext
        ai, aj = np.where(src)
        s = rng.integers(0, ai.size, n_ic)
        Xic = np.column_stack([xs[ai[s]], xs[aj[s]], np.zeros(n_ic)])
        Phic = phi_k[:, ai[s], aj[s]].T              # [n_ic, 4]
        Xic_list.append(tf.constant(Xic, DTYPE))
        Phic_list.append(tf.constant(Phic, DTYPE))

        # Denoising: deep-bulk cells in extended box
        dm = inbox_ext & deep_bulk
        if dm.any():
            di, dj = np.where(dm)
            sd = rng.integers(0, di.size, n_dn)
            Xdn = np.column_stack([xs[di[sd]], xs[dj[sd]], rng.uniform(0, dt, n_dn)])
            Phidn = bulk_target[:, di[sd], dj[sd]].T  # [n_dn, 4]
            Xdn_list.append(tf.constant(Xdn, DTYPE))
            Phidn_list.append(tf.constant(Phidn, DTYPE))
        else:
            Xdn_list.append(EMPTY)
            Phidn_list.append(EMPTY4)

    # Continuity: uniform in overlap halos between neighbor pairs
    boxes = multinn.boxes
    pairs = neighbor_pairs(boxes, max(b["i"] for b in boxes)+1,
                                  max(b["j"] for b in boxes)+1)
    cont = []
    for (a, c) in pairs:
        ax0, ax1, ay0, ay1 = boxes[a]["ext"]
        bx0, bx1, by0, by1 = boxes[c]["ext"]
        ox0, ox1 = max(ax0, bx0), min(ax1, bx1)
        oy0, oy1 = max(ay0, by0), min(ay1, by1)
        if ox1 > ox0 and oy1 > oy0:
            xc = rng.uniform(ox0, ox1, n_cont)
            yc = rng.uniform(oy0, oy1, n_cont)
            cont.append(tf.constant(
                np.column_stack([xc, yc, rng.uniform(0, dt, n_cont)]), DTYPE))
        else:
            cont.append(EMPTY)

    # Periodic seam batches (enforce phi continuity across the periodic wrap).
    # For a 2×2 grid, the four outer wrap seams are:
    #   x-wrap lower: Box 0 (x=0 left edge) ↔ Box 1 (x=L right edge), y∈[0,L/2)
    #   x-wrap upper: Box 2 (x=0 left edge) ↔ Box 3 (x=L right edge), y∈[L/2,L)
    #   y-wrap left:  Box 0 (y=0 bottom edge) ↔ Box 2 (y=L top edge), x∈[0,L/2)
    #   y-wrap right: Box 1 (y=0 bottom edge) ↔ Box 3 (y=L top edge), x∈[L/2,L)
    # Each seam stores (worker_a_idx, worker_b_idx, X_a, X_b) with DISTINCT
    # coordinate tensors even though the two sides are the same physical location.
    n_pbc = n_cont if n_pbc is None else n_pbc
    pbc_seams = []
    for k, bk in enumerate(multinn.boxes):
        cx0k, cx1k, cy0k, cy1k = bk["core"]
        if abs(cx0k) < 1e-10:           # left-edge box (x=0 boundary)
            for m, bm in enumerate(multinn.boxes):
                cx0m, cx1m, cy0m, cy1m = bm["core"]
                if abs(cx1m - L) < 1e-10 and abs(cy0m - cy0k) < 1e-10:
                    # right-edge box with same y-core range → x-wrap seam pair
                    ys = rng.uniform(cy0k, cy1k, n_pbc)
                    ts = rng.uniform(0, dt, n_pbc)
                    X_a = tf.constant(np.c_[np.zeros(n_pbc), ys, ts], DTYPE)
                    X_b = tf.constant(np.c_[np.full(n_pbc, L), ys, ts], DTYPE)
                    pbc_seams.append((k, m, X_a, X_b))
        if abs(cy0k) < 1e-10:           # bottom-edge box (y=0 boundary)
            for m, bm in enumerate(multinn.boxes):
                cx0m, cx1m, cy0m, cy1m = bm["core"]
                if abs(cy1m - L) < 1e-10 and abs(cx0m - cx0k) < 1e-10:
                    # top-edge box with same x-core range → y-wrap seam pair
                    xs_s = rng.uniform(cx0k, cx1k, n_pbc)
                    ts = rng.uniform(0, dt, n_pbc)
                    X_a = tf.constant(np.c_[xs_s, np.zeros(n_pbc), ts], DTYPE)
                    X_b = tf.constant(np.c_[xs_s, np.full(n_pbc, L), ts], DTYPE)
                    pbc_seams.append((k, m, X_a, X_b))

    batch = {
        "Xf":    Xf_list,
        "Xic":   Xic_list,
        "Phic":  Phic_list,
        "Xdn":   Xdn_list,
        "Phidn": Phidn_list,
        "Xc":    cont,
        "pbc":   pbc_seams,            # list of (k_a, k_b, X_a, X_b) — one per seam
    }
    return batch, pairs


# ── loss ─────────────────────────────────────────────────────────────────────

def make_loss_multiphase(multinn: MultiNN, p, pairs, w_pde=1.0, w_ic=2.0,
                          w_dn=1.0, w_cont=1.0, w_pbc=0.0, pde_chunk=None):
    """Joint multiphase decomposition loss with mutable term weights.

    Returns (loss, set_batch, W, terms, nonpde_loss, pde_grads_fn, terms_per_worker):
      loss()              -> (total, {total, pde, ic, dn, cont, pbc})
      set_batch(b)        -> update the active batch
      W                   -> dict of tf.Variable weights (mutable)
      terms()             -> unweighted per-term scalar means (for diagnostics/GradNorm)
      nonpde_loss()       -> weighted scalar of IC + DN + cont + PBC (no Hessian)
      pde_grads_fn()      -> list of gradient tensors for PDE loss only, computed via
      terms_per_worker()  -> per-box breakdown dict (Python floats); use after training
                             for pyramid threshold calibration (not in training loop)
                         gradient accumulation (one GradientTape per chunk) so peak
                         GPU memory = one chunk's pfor Hessian intermediates (~21 MB
                         for 64×6, 300 pts) instead of all workers at once.
                         Use with pde_eager=True in decomp_fit_interval.

    pde_chunk : int | None
      If set, the PDE batch_jacobian is computed in mini-batches of this size.
      In the regular loss() path this is unrolled at graph trace time (fine for small nets).
      In the pde_grads_fn() path it controls the gradient-accumulation chunk size
      (one GradientTape per chunk = true sequential memory use regardless of @tf.function).

    Discrete activation:
      Start with w_dn=0, w_cont=0, w_pbc=0 (first window, establish interface solution).
      Call W['dn'].assign(1.0); W['cont'].assign(1.0); W['pbc'].assign(1.0) for window 2+.
    """
    B = len(multinn.workers)
    batch_ref = {}
    W = {
        "pde":  tf.Variable(w_pde,  dtype=DTYPE),
        "ic":   tf.Variable(w_ic,   dtype=DTYPE),
        "dn":   tf.Variable(w_dn,   dtype=DTYPE),
        "cont": tf.Variable(w_cont, dtype=DTYPE),
        "pbc":  tf.Variable(w_pbc,  dtype=DTYPE),
    }

    def _pde_worker(wk, Xf):
        """Mean squared PDE residual, optionally chunked to cap peak GPU memory."""
        if pde_chunk is None or Xf.shape[0] <= pde_chunk:
            r = pde_residual_multiphase(wk, Xf, p, NPHASES)
            return tf.reduce_mean(tf.square(r))
        # chunked: unrolled at trace time (pde_chunk and Xf.shape[0] are Python ints)
        acc = tf.constant(0.0, DTYPE); n_ch = 0
        for s in range(0, int(Xf.shape[0]), pde_chunk):
            r = pde_residual_multiphase(wk, Xf[s:s + pde_chunk], p, NPHASES)
            acc += tf.reduce_mean(tf.square(r)); n_ch += 1
        return acc / tf.cast(n_ch, DTYPE)

    def terms():
        pde = tf.constant(0.0, DTYPE); npde = 0
        ic  = tf.constant(0.0, DTYPE)
        dn  = tf.constant(0.0, DTYPE); ndn  = 0

        for k in range(B):
            wk = multinn.workers[k]

            # --- PDE residual (4-phase, batch_jacobian, optionally chunked) ---
            Xf = batch_ref["Xf"][k]
            if Xf.shape[0] > 0:
                pde += _pde_worker(wk, Xf); npde += 1

            # --- IC (4-phase targets) ---
            phi_pred = phases_from_logits(wk(batch_ref["Xic"][k]))   # [n_ic, 4]
            ic += tf.reduce_mean(tf.square(phi_pred - batch_ref["Phic"][k]))

            # --- Denoising (one-hot bulk targets) ---
            Xdn = batch_ref["Xdn"][k]
            if Xdn.shape[0] > 0:
                phi_dn = phases_from_logits(wk(Xdn))                 # [n_dn, 4]
                dn += tf.reduce_mean(tf.square(phi_dn - batch_ref["Phidn"][k]))
                ndn += 1

        # --- Continuity (all 4 phases in overlap halos) ---
        cont = tf.constant(0.0, DTYPE); nc = 0
        for pi, (a, c) in enumerate(pairs):
            Xc = batch_ref["Xc"][pi]
            if Xc.shape[0] > 0:
                phi_a = phases_from_logits(multinn.workers[a](Xc))   # [m, 4]
                phi_c = phases_from_logits(multinn.workers[c](Xc))   # [m, 4]
                cont += tf.reduce_mean(tf.square(phi_a - phi_c))
                nc += 1

        # --- Periodic seam (outer wrap: x=0 vs x=L, y=0 vs y=L) ---
        pbc = tf.constant(0.0, DTYPE); n_pbc_c = 0
        for (k_a, k_b, X_a, X_b) in batch_ref.get("pbc", []):
            if X_a.shape[0] > 0:
                phi_a = phases_from_logits(multinn.workers[k_a](X_a))   # [m, 4]
                phi_b = phases_from_logits(multinn.workers[k_b](X_b))   # [m, 4]
                pbc += tf.reduce_mean(tf.square(phi_a - phi_b))
                n_pbc_c += 1

        return {
            "pde":  pde  / max(1, npde),
            "ic":   ic   / B,
            "dn":   dn   / max(1, ndn),
            "cont": cont / max(1, nc),
            "pbc":  pbc  / max(1, n_pbc_c),
        }

    def loss():
        t = terms()
        total = (W["pde"] * t["pde"] + W["ic"] * t["ic"]
                 + W["dn"] * t["dn"] + W["cont"] * t["cont"]
                 + W["pbc"] * t["pbc"])
        return total, {"total": total, **t}

    def nonpde_loss():
        """Weighted IC + DN + cont + PBC loss — no Hessian, no batch_jacobian.
        Used for fast tape in gradient accumulation mode."""
        ic  = tf.constant(0.0, DTYPE)
        dn  = tf.constant(0.0, DTYPE); ndn = 0
        for k in range(B):
            wk = multinn.workers[k]
            phi_pred = phases_from_logits(wk(batch_ref["Xic"][k]))
            ic += tf.reduce_mean(tf.square(phi_pred - batch_ref["Phic"][k]))
            Xdn = batch_ref["Xdn"][k]
            if Xdn.shape[0] > 0:
                phi_dn = phases_from_logits(wk(Xdn))
                dn += tf.reduce_mean(tf.square(phi_dn - batch_ref["Phidn"][k]))
                ndn += 1
        cont = tf.constant(0.0, DTYPE); nc = 0
        for pi, (a, c) in enumerate(pairs):
            Xc = batch_ref["Xc"][pi]
            if Xc.shape[0] > 0:
                phi_a = phases_from_logits(multinn.workers[a](Xc))
                phi_c = phases_from_logits(multinn.workers[c](Xc))
                cont += tf.reduce_mean(tf.square(phi_a - phi_c)); nc += 1
        pbc = tf.constant(0.0, DTYPE); n_pbc_c = 0
        for (k_a, k_b, X_a, X_b) in batch_ref.get("pbc", []):
            if X_a.shape[0] > 0:
                phi_a = phases_from_logits(multinn.workers[k_a](X_a))
                phi_b = phases_from_logits(multinn.workers[k_b](X_b))
                pbc += tf.reduce_mean(tf.square(phi_a - phi_b)); n_pbc_c += 1
        return (W["ic"] * ic / B + W["dn"] * dn / max(1, ndn)
                + W["cont"] * cont / max(1, nc) + W["pbc"] * pbc / max(1, n_pbc_c))

    def pde_grads_fn():
        """PDE gradient accumulation: one GradientTape per worker-chunk.

        Peak GPU memory = one chunk's pfor Hessian intermediates at a time.
        Gradients are summed then averaged over the number of contributing chunks.
        Workers with no PDE points contribute nothing (None grads → zero increment).
        """
        vars_ = multinn.trainable_variables
        accum = [tf.zeros_like(v) for v in vars_]
        n_chunks = 0
        chunk_size = pde_chunk  # may be None (whole batch at once)
        for k in range(B):
            wk = multinn.workers[k]
            Xf = batch_ref["Xf"][k]
            n = int(Xf.shape[0])
            if n == 0:
                continue
            effective_chunk = n if (chunk_size is None or chunk_size >= n) else chunk_size
            for s in range(0, n, effective_chunk):
                X_c = Xf[s:s + effective_chunk]
                with tf.GradientTape() as tape_c:
                    r = pde_residual_multiphase(wk, X_c, p, NPHASES)
                    closs = W["pde"] * tf.reduce_mean(tf.square(r))
                grads = tape_c.gradient(closs, vars_)
                for i, (a, g) in enumerate(zip(accum, grads)):
                    if g is not None:
                        accum[i] = a + g
                n_chunks += 1
        if n_chunks > 1:
            inv = tf.constant(1.0 / n_chunks, DTYPE)
            accum = [g * inv for g in accum]
        return accum

    def terms_per_worker():
        """Per-box diagnostic breakdown (Python floats, not TF scalars).

        Returns dict with lists of length B (one entry per spatial box):
          pde_per_worker : mean squared PDE residual per box
          ic_per_worker  : mean squared IC loss per box
          dn_per_worker  : mean squared denoising loss per box (0.0 if Xdn empty)

        Call after training each window to capture per-box losses for pyramid
        threshold calibration (sets T = 5× per-box PDE loss from this run).
        Not for use inside a training loop — returns numpy floats, not TF scalars.
        """
        pde_per = []; ic_per = []; dn_per = []
        for k in range(B):
            wk = multinn.workers[k]
            Xf = batch_ref["Xf"][k]
            pde_val = float(_pde_worker(wk, Xf).numpy()) if Xf.shape[0] > 0 else 0.0
            phi_ic = phases_from_logits(wk(batch_ref["Xic"][k]))
            ic_val = float(tf.reduce_mean(tf.square(phi_ic - batch_ref["Phic"][k])).numpy())
            Xdn = batch_ref["Xdn"][k]
            if Xdn.shape[0] > 0:
                phi_dn = phases_from_logits(wk(Xdn))
                dn_val = float(tf.reduce_mean(
                    tf.square(phi_dn - batch_ref["Phidn"][k])).numpy())
            else:
                dn_val = 0.0
            pde_per.append(pde_val)
            ic_per.append(ic_val)
            dn_per.append(dn_val)
        return {"pde_per_worker": pde_per, "ic_per_worker": ic_per,
                "dn_per_worker": dn_per}

    def set_batch(batch):
        batch_ref.clear()
        # Migrate all tensors to GPU:0 so eager pfor matmuls run on GPU, not CPU.
        # tf.constant creates CPU tensors by default; eager dispatch uses input device.
        with tf.device('/GPU:0'):
            gpu_batch = {}
            for key, val in batch.items():
                if key == 'pbc':
                    gpu_batch[key] = [
                        (ka, kb, tf.identity(Xa), tf.identity(Xb))
                        for (ka, kb, Xa, Xb) in val
                    ]
                elif isinstance(val, list):
                    gpu_batch[key] = [tf.identity(t) for t in val]
                else:
                    gpu_batch[key] = val
        batch_ref.update(gpu_batch)

    return loss, set_batch, W, terms, nonpde_loss, pde_grads_fn, terms_per_worker
