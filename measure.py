"""Refine the landmarks of one test and measure its test line against its control line.

Canonical frames
  strips (Wondfo, easy@Home HCG): u = 0 at the centre of the black MAX bar, u = 1 at the centre
      of the control line; v across the strip. The test line sits at u ~ 0.8.
  First Response: origin at the centre of the control line, unit = length of the control line,
      +u pointing away from the pink cap. The test line sits at u ~ -0.8.

Metric
  A row-median profile is taken along u over the central band of the line. A straight baseline is
  fitted to the blank membrane on both flanks, and the fractional green-light absorption
  1 - G/G_baseline is summed across the line. That integral does not change with exposure, white
  balance or focus blur. Score = 100 * (test line integral) / (control line integral).
"""
import os

import cv2
import numpy as np

from common import LocateError, luminance, pinkness, rectify, srgb2lin, to_full

CFG = {
    'wondfo': dict(T0=0.805, half=0.035, f0=0.045, f1=0.09, cf1=0.075, search=0.02, white=(0.86, 0.94)),
    # no easy@Home photo so far shows a test line, so its position is assumed to match the Wondfo layout
    'hcg':    dict(T0=0.80, half=0.035, f0=0.045, f1=0.09, cf1=0.075, search=0.02, white=(0.86, 0.94)),
    'fr':     dict(T0=-0.815, half=0.065, f0=0.085, f1=0.17, search=0.035, white=(-0.5, -0.2)),
}
STRIP_FRAME = dict(ppu=1000, ur=(-0.2, 1.4), vr=(-0.16, 0.16))
FR_FRAME = dict(ppu=500, ur=(-1.8, 1.3), vr=(-0.85, 0.85))
MAX_ITER, CONVERGED_PX = 8, 0.3   # landmark refinement stops once it moves less than this


def run_near(mask, centre):
    """(first, last) index of the long True run closest to `centre`."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        raise LocateError('no run')
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    runs = [r for r in runs if len(r) > 3]
    if not runs:
        raise LocateError('no run')
    best = max(runs, key=lambda r: len(r) - 0.5 * abs((r[0] + r[-1]) / 2 - centre))
    return best[0], best[-1]


def gaussian_filter1d(x, sigma):
    """1-D Gaussian smoothing with mirrored edges (same result as scipy.ndimage.gaussian_filter1d)."""
    r = int(4.0 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return np.convolve(np.pad(np.asarray(x, np.float64), r, mode='symmetric'), k / k.sum(), mode='valid')


def _centroid_line(w, coords, rows_v):
    """Per-row weighted centroid, then a straight-line fit through the rows that have signal."""
    s = w.sum(1)
    ok = s > 1e-9
    if ok.sum() < 6:
        raise LocateError('line not found')
    cu = (w[ok] * coords).sum(1) / s[ok]
    return np.polyfit(rows_v[ok], cu, 1)


# ---------------------------------------------------------------- strips
def refine_strip(a, B, C, max_iter=MAX_ITER):
    """Refine rough full-res positions of the black bar centre B and control line centre C.

    Returns (B, C, halfw, info): halfw is the strip half-width in canonical units.
    """
    B = np.array(B, np.float64)
    C = np.array(C, np.float64)
    for _ in range(max_iter):
        unit = np.linalg.norm(C - B)
        if not np.isfinite(unit) or unit < 20:
            raise LocateError('degenerate strip')
        ppu, ur, vr = 700, (-0.3, 1.3), (-0.3, 0.3)
        r, M = rectify(a, B, C, unit, ur, vr, ppu)
        lin = srgb2lin(r)
        lum = cv2.GaussianBlur(luminance(lin), (0, 0), 2)
        H, W = lum.shape
        u = ur[0] + np.arange(W) / ppu
        v = vr[0] + np.arange(H) / ppu
        white = np.median(lum[np.abs(v) < 0.05][:, (u > 0.12) & (u < 0.55)])
        # black bar: dark-pixel centroid per row, extent across the strip gives the strip width
        ub = np.flatnonzero(np.abs(u) < 0.07)
        sub = lum[:, ub]
        # a bar row is dark at the bar and white on the pad side of it; the second condition keeps a
        # dark background beside the strip (shadow, dark table) from being counted as bar
        pad = np.median(lum[:, (u > -0.075) & (u < -0.04)], axis=1)
        rowdark = ((sub < 0.3 * white).sum(1) > 0.012 * ppu) & (pad > 0.5 * white)
        r0, r1 = run_near(rowdark, H / 2)
        rows = np.arange(r0, r1 + 1)
        pb = _centroid_line(np.clip(0.5 * white - sub[rows], 0, None), u[ub], v[rows])
        vmid = (v[r0] + v[r1]) / 2
        halfw = (v[r1] - v[r0]) / 2
        Bu = np.polyval(pb, vmid)
        # control line: pinkness centroid per row
        pk = cv2.GaussianBlur(pinkness(lin), (0, 0), 2)
        uc = np.flatnonzero(np.abs(u - 1) < 0.08)
        inner = rows[(v[rows] > vmid - 0.75 * halfw) & (v[rows] < vmid + 0.75 * halfw)]
        if len(inner) < 6:
            raise LocateError('strip too narrow')
        subp = pk[inner][:, uc]
        base = np.median(pk[inner][:, (u > 0.86) & (u < 0.92)])
        w = np.clip(subp - base - 0.4 * (subp.max(1, keepdims=True) - base), 0, None)
        pc = _centroid_line(w, u[uc], v[inner])
        Cu = np.polyval(pc, vmid)
        newB = to_full(M, (Bu - ur[0]) * ppu, (vmid - vr[0]) * ppu)
        newC = to_full(M, (Cu - ur[0]) * ppu, (vmid - vr[0]) * ppu)
        shift = max(np.linalg.norm(newB - B), np.linalg.norm(newC - C))
        B, C = newB, newC
        if shift < CONVERGED_PX:
            break
    return B, C, float(halfw), dict(shift=float(shift), tiltB=float(pb[0]), tiltC=float(pc[0]))


def line_metric(u, prof, centre, half, f0, f1, search=0.0, ppu=1):
    """Integrated fractional green absorption of a line near `centre`.

    Baseline: straight-line fit over the flanks [c-f1, c-f0] and [c+f0, c+f1].
    With search > 0 the centre is first moved to the strongest absorption within +-search.
    """
    G = prof[:, 1]

    def calc(c):
        fl = ((u > c - f1) & (u < c - f0)) | ((u > c + f0) & (u < c + f1))
        core = np.abs(u - c) <= half
        bg = np.polyval(np.polyfit(u[fl], G[fl], 1), u)
        ab = 1 - G / bg
        return dict(G=float(ab[core].sum() / ppu), peak=float(ab[core].max()), bg=bg)

    c = centre
    if search > 0:
        fl = (u > centre - f1) & (u < centre + f1)
        ab = gaussian_filter1d(1 - G / np.polyval(np.polyfit(u[fl], G[fl], 1), u), half * ppu / 2.5)
        cand = np.flatnonzero(np.abs(u - centre) <= search)
        c = float(u[cand[np.argmax(ab[cand])]])
    out = calc(c)
    out['centre'] = c
    return out


def _smooth_profile(lin, rows):
    p = np.median(lin[rows], axis=0)
    return np.stack([gaussian_filter1d(p[:, c], 2) for c in range(3)], 1)


def subband_se(lin, band, u, Tc, Cc, cfg, cf1, ppu, n=6):
    """Standard error of T/C from n independent row sub-bands (pixel noise + local unevenness)."""
    vals = []
    for rows in np.array_split(band, n):
        p = _smooth_profile(lin, rows)
        t = line_metric(u, p, Tc, cfg['half'], cfg['f0'], cfg['f1'], ppu=ppu)
        c = line_metric(u, p, Cc, cfg['half'], cfg['f0'], cf1, ppu=ppu)
        vals.append(t['G'] / c['G'])
    return float(np.std(vals, ddof=1) / np.sqrt(n))


def _debug_plot(path, r, ur, vr, u, prof, T, Cm, white, hlines, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw=dict(height_ratios=[1.3, 2]))
    ax[0].imshow(r, extent=[ur[0], ur[1], vr[1], vr[0]], aspect='auto')
    for y, col in hlines:
        ax[0].axhline(y, color=col, lw=.5)
    for x in (T['centre'], Cm['centre']):
        ax[0].axvline(x, color='lime', lw=.5)
    for c, col in enumerate('rgb'):
        ax[1].plot(u, prof[:, c], col, lw=.8)
    ax[1].plot(u, T['bg'], 'k--', lw=.5)
    ax[1].plot(u, Cm['bg'], 'k:', lw=.5)
    ax[1].set_xlim(*ur)
    ax[1].set_ylim(0, 1.2 * max(white))
    ax[1].axvline(T['centre'], color='lime', lw=.5)
    ax[1].set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def measure_strip(brand, name, a, meta, B, C, debug_dir=None):
    """Measure one strip. B, C: rough full-res positions of the black bar and the control line.

    Returns (result dict, canonical linear-RGB crop).
    """
    cfg = CFG[brand]
    B, C, halfw, info = refine_strip(a, B, C)
    unit = np.linalg.norm(C - B)
    ppu, ur, vr = STRIP_FRAME['ppu'], STRIP_FRAME['ur'], STRIP_FRAME['vr']
    r, M = rectify(a, B, C, unit, ur, vr, ppu)
    lin = srgb2lin(r)
    H, W = lin.shape[:2]
    u = ur[0] + np.arange(W) / ppu
    v = vr[0] + np.arange(H) / ppu
    band = np.flatnonzero(np.abs(v) < 0.55 * halfw)
    prof = _smooth_profile(lin, band)
    T = line_metric(u, prof, cfg['T0'], cfg['half'], cfg['f0'], cfg['f1'], search=cfg['search'], ppu=ppu)
    Cm = line_metric(u, prof, 1.0, cfg['half'], cfg['f0'], cfg['cf1'], search=0.02, ppu=ppu)
    wm = (u > cfg['white'][0]) & (u < cfg['white'][1])
    res = dict(brand=brand, image=name, dt=meta.get('dt'), unit_px=float(unit), halfw=halfw, **info,
               bar_xy=B.tolist(), control_xy=C.tolist(),
               T=T['G'], C=Cm['G'], T_peak=T['peak'], C_peak=Cm['peak'], T_pos=T['centre'], C_pos=Cm['centre'],
               ratio_se=subband_se(lin, band, u, T['centre'], Cm['centre'], cfg, cfg['cf1'], ppu),
               black=np.median(lin[band][:, np.abs(u) < 0.012].reshape(-1, 3), 0).tolist(),
               white=np.median(lin[band][:, wm].reshape(-1, 3), 0).tolist())
    if debug_dir:
        _debug_plot(os.path.join(debug_dir, f'{brand}_{name}.png'), r, ur, vr, u, prof, T, Cm, res['white'],
                    [(-0.55 * halfw, 'c'), (0.55 * halfw, 'c'), (-halfw, 'y'), (halfw, 'y')],
                    f"{brand} {name}  T={T['G'] * 1e3:.2f}  C={Cm['G'] * 1e3:.2f}  T/C={T['G'] / Cm['G']:.3f}")
    return res, lin.astype(np.float32)


# ---------------------------------------------------------------- First Response
def refine_fr(a, Cc, d, unit, max_iter=MAX_ITER):
    """Refine the control line of a First Response test.

    Cc -- rough full-res centre of the control line
    d  -- rough unit vector perpendicular to the line, pointing away from the pink cap
    unit -- rough length of the control line in full-res pixels
    Returns (Cc, d, unit, info).
    """
    Cc = np.array(Cc, np.float64)
    d = np.array(d, np.float64) / np.linalg.norm(d)
    for _ in range(max_iter):
        if not np.isfinite(unit) or unit < 20:
            raise LocateError('degenerate control line')
        ppu, ur, vr = 500, (-0.6, 0.6), (-0.9, 0.9)
        r, M = rectify(a, Cc, Cc + d, unit, ur, vr, ppu)
        lin = srgb2lin(r)
        H, W = lin.shape[:2]
        u = ur[0] + np.arange(W) / ppu
        v = vr[0] + np.arange(H) / ppu
        pk = cv2.GaussianBlur(pinkness(lin), (0, 0), 2.5)
        uc = np.flatnonzero(np.abs(u) < 0.25)
        base = np.median(pk[:, (np.abs(u) > 0.3) & (np.abs(u) < 0.5)], axis=1)
        sub = pk[:, uc] - base[:, None]
        peak = sub.max(1)
        ref = np.percentile(peak[np.abs(v) < 0.3], 50)
        if ref <= 0.02:
            raise LocateError('no control line')
        r0, r1 = run_near(peak > 0.5 * ref, H / 2)
        rows = np.arange(r0, r1 + 1)
        n6 = max(1, len(rows) // 6)
        if len(rows) - 2 * n6 < 6:
            raise LocateError('control line too short')
        w = np.clip(sub[rows] - 0.4 * peak[rows, None], 0, None)
        p = _centroid_line(w[n6:-n6], u[uc], v[rows][n6:-n6])
        vmid = (v[r0] + v[r1]) / 2
        Lc = v[r1] - v[r0]
        newC = to_full(M, (np.polyval(p, vmid) - ur[0]) * ppu, (vmid - vr[0]) * ppu)
        p0 = to_full(M, (np.polyval(p, v[r0]) - ur[0]) * ppu, r0)
        p1 = to_full(M, (np.polyval(p, v[r1]) - ur[0]) * ppu, r1)
        t = (p1 - p0) / np.linalg.norm(p1 - p0)
        nd = np.array([t[1], -t[0]])
        if nd @ d < 0:
            nd = -nd
        shift = float(np.linalg.norm(newC - Cc))
        Cc, d, unit = newC, nd, unit * Lc
        if shift < CONVERGED_PX and abs(Lc - 1) < 0.01:
            break
    return Cc, d, float(unit), dict(Lc_last=float(Lc), tilt=float(p[0]), shift=shift)


def measure_fr(name, a, meta, Cc, d, unit, debug_dir=None):
    """Measure one First Response test. See refine_fr for the arguments."""
    cfg = CFG['fr']
    Cc, d, unit, info = refine_fr(a, Cc, d, unit)
    ppu, ur, vr = FR_FRAME['ppu'], FR_FRAME['ur'], FR_FRAME['vr']
    r, M = rectify(a, Cc, Cc + d, unit, ur, vr, ppu)
    lin = srgb2lin(r)
    H, W = lin.shape[:2]
    u = ur[0] + np.arange(W) / ppu
    v = vr[0] + np.arange(H) / ppu
    band = np.flatnonzero(np.abs(v) < 0.3)
    prof = _smooth_profile(lin, band)
    T = line_metric(u, prof, cfg['T0'], cfg['half'], cfg['f0'], cfg['f1'], search=cfg['search'], ppu=ppu)
    Cm = line_metric(u, prof, 0.0, cfg['half'], cfg['f0'], cfg['f1'], search=0.03, ppu=ppu)
    wm = (u > cfg['white'][0]) & (u < cfg['white'][1])
    res = dict(brand='fr', image=name, dt=meta.get('dt'), unit_px=unit, **info,
               control_xy=Cc.tolist(), away_from_cap=d.tolist(),
               T=T['G'], C=Cm['G'], T_peak=T['peak'], C_peak=Cm['peak'], T_pos=T['centre'], C_pos=Cm['centre'],
               ratio_se=subband_se(lin, band, u, T['centre'], Cm['centre'], cfg, cfg['f1'], ppu),
               black=None, white=np.median(lin[band][:, wm].reshape(-1, 3), 0).tolist())
    if debug_dir:
        _debug_plot(os.path.join(debug_dir, f'fr_{name}.png'), r, ur, vr, u, prof, T, Cm, res['white'],
                    [(-0.3, 'c'), (0.3, 'c'), (-0.5, 'y'), (0.5, 'y')],
                    f"fr {name}  T={T['G'] * 1e3:.2f}  C={Cm['G'] * 1e3:.2f}  T/C={T['G'] / Cm['G']:.3f}")
    return res, lin.astype(np.float32)
