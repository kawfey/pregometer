"""Find the tests in a photo automatically.

Every test has one thing in common: a strongly pink control line on a white background. So:

  1. Find thin pink line segments anywhere in the photo (morphological top-hat of log(R/G)).
  2. For each segment, test two hypotheses about what surrounds it:
       strip           -- a black MAX bar, as wide as the line and parallel to it, lies a few
                          line-lengths away, with white membrane in between and a coloured
                          brand label on the other side. The label colour gives the brand.
       First Response  -- the line sits in white plastic and the pink cap lies along the
                          line's normal, 6-8 line-lengths away.
  3. Refine the surviving positions with the same sub-pixel fit the measurement uses, check
     the layout once more in the refined frame, and drop duplicates (including the test line
     being mistaken for the control line).

Run `python detect.py PHOTO_DIR --overlay OUT_DIR` to see what is found in each photo.
"""
import os
import sys

import cv2
import numpy as np

from common import LocateError, list_photos, load, luminance, pinkness, rectify, srgb2lin
from measure import refine_fr, refine_strip

DET_LONG_SIDE = 2016           # photos are searched at this size (px on the long side)
LEVELS = (0.18, 0.4, 0.7, 1.1)  # top-hat thresholds; high ones separate a line from label print
MIN_LEN, MAX_LEN = 22, 260     # accepted line length at detection scale
BRAND_NAMES = {'wondfo': 'Wondfo', 'hcg': 'easy@Home HCG', 'fr': 'First Response', 'pregmate': 'Pregmate'}
MEASURED = ('wondfo', 'hcg', 'fr')   # Pregmate strips are recognised but not measured


def _rg(c): return float(np.log((c[0] + 1e-3) / (c[1] + 1e-3)))
def _gb(c): return float(np.log((c[1] + 1e-3) / (c[2] + 1e-3)))
def _lum(c): return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def line_candidates(a):
    """Thin pink line segments: list of dict(c=centre xy, t=unit direction, L=length, W=width), full-res px."""
    s = min(1.0, DET_LONG_SIDE / max(a.shape[:2]))
    small = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else a
    pk = cv2.GaussianBlur(pinkness(srgb2lin(small)).astype(np.float32), (0, 0), 1.2)
    th = cv2.morphologyEx(pk, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)))
    out = []
    for lev in LEVELS:
        n, lab, stats, _ = cv2.connectedComponentsWithStats((th > lev).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 40 or max(w, h) < MIN_LEN or max(w, h) > MAX_LEN:
                continue
            ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
            pts = np.stack([xs + x, ys + y], 1).astype(np.float64)
            wgt = th[ys + y, xs + x].astype(np.float64)
            c = (pts * wgt[:, None]).sum(0) / wgt.sum()
            d = pts - c
            ev, evec = np.linalg.eigh((d * wgt[:, None]).T @ d / wgt.sum())
            t = evec[:, 1]
            proj = d @ t
            lo, hi = np.percentile(proj, 2), np.percentile(proj, 98)
            L, W = hi - lo, 2.355 * np.sqrt(max(ev[0], 1e-6))
            if L < MIN_LEN or L / W < 2.8 or W > 22:
                continue
            out.append(dict(c=(c + (lo + hi) / 2 * t) / s, t=t, L=L / s, W=W / s))
    return out


# ---------------------------------------------------------------- strips
def find_bar(a, cand):
    """Rough positions of a black bar parallel to the candidate line, on either side of it."""
    c, t, L = cand['c'], cand['t'], cand['L']
    n = np.array([t[1], -t[0]])
    ppu, ur, vr = 40, (-7.5, 7.5), (-0.5, 0.5)
    r, _ = rectify(a, c, c + n, L, ur, vr, ppu)
    lum = luminance(srgb2lin(r))
    u = ur[0] + np.arange(lum.shape[1]) / ppu
    v = vr[0] + np.arange(lum.shape[0]) / ppu
    # a bar is dark across the whole strip; print (arrows, "MAX") is not, so use a high percentile
    p = np.percentile(lum[np.abs(v) < 0.3], 65, axis=0)
    hits = []
    for sgn in (1, -1):
        white = np.median(p[(u * sgn > 0.35) & (u * sgn < 1.3)])
        idx = np.flatnonzero((u * sgn > 2.6) & (u * sgn < 6.8))
        dark = np.flatnonzero(p[idx] < 0.3 * white)
        if len(dark) == 0:
            continue
        for run in np.split(dark, np.flatnonzero(np.diff(dark) > 1) + 1):
            ln = len(run) / ppu
            if not 0.04 <= ln <= 0.6:
                continue
            uc = u[idx[run]].mean()
            far = (u * sgn > abs(uc) + ln / 2 + 0.15) & (u * sgn < abs(uc) + ln / 2 + 0.6)
            if far.sum() == 0 or np.median(p[far]) < 0.5 * white:   # absorbent pad beyond the bar
                continue
            hits.append(c + n * uc * L)
    return hits


def classify_label(rel):
    """Brand from the label colour relative to the membrane white (linear RGB ratios)."""
    med = np.median(rel, 0)
    rg, gb, y = _rg(med), _gb(med), _lum(med)
    if rg > 0.25 and gb > 0.2 and y > 0.45:
        return 'wondfo', (rg, gb, y)       # orange
    if rg > 0.2 and gb <= 0.2 and y > 0.55:
        return 'hcg', (rg, gb, y)          # light pink
    if rg > 0.25 and y <= 0.55:
        return 'pregmate', (rg, gb, y)     # dark purple
    return None, (rg, gb, y)


def check_strip(a, B, C, halfw, info):
    """Verify the strip layout in the refined frame. Returns dict(brand, lineness, ...) or None."""
    if not (0.085 < halfw < 0.19) or info['shift'] > 3 or max(abs(info['tiltB']), abs(info['tiltC'])) > 0.2:
        return None
    unit = np.linalg.norm(C - B)
    ppu, ur, vr = 250, (-0.3, 1.9), (-0.2, 0.2)
    r, _ = rectify(a, B, C, unit, ur, vr, ppu)
    lin = srgb2lin(r)
    u = ur[0] + np.arange(lin.shape[1]) / ppu
    v = vr[0] + np.arange(lin.shape[0]) / ppu
    band = lin[np.abs(v) < 0.6 * halfw]
    med = lambda u0, u1: np.median(band[:, (u > u0) & (u < u1)].reshape(-1, 3), 0)
    white, pad, mem, bar = med(0.86, 0.94), med(-0.2, -0.06), med(0.6, 0.75), med(-0.008, 0.008)
    lw = _lum(white)
    if not (0.5 < _lum(pad) / lw < 2.0 and 0.5 < _lum(mem) / lw < 1.7 and _lum(bar) < 0.25 * lw):
        return None
    if abs(_rg(mem) - _rg(white)) > 0.2 or abs(_gb(mem) - _gb(white)) > 0.25:
        return None
    if abs(_rg(bar) - _rg(white)) > 0.5:      # the bar is black, not dark red (a strong control line)
        return None
    # the control line must cross the whole strip and be pink: per-row peaks, low percentile
    ctl = band[:, np.abs(u - 1) < 0.045]
    G = cv2.GaussianBlur(ctl[..., 1], (0, 0), 1.5)
    lineness = float(np.percentile(1 - G.min(1) / white[1], 20))
    pink = float(np.percentile(cv2.GaussianBlur(pinkness(ctl), (0, 0), 1.5).max(1), 20)) - _rg(white)
    if lineness < 0.25 or pink < 0.25:
        return None
    brand, feat = classify_label(band[:, (u > 1.25) & (u < 1.75)].reshape(-1, 3) / white)
    if brand is None:
        return None
    return dict(brand=brand, lineness=lineness, label=feat)


# ---------------------------------------------------------------- First Response
def _cap_profile(a, c, n, unit, reach=15.0):
    """Fraction of saturated-pink pixels across the body, as a function of distance along n."""
    ppu, ur, vr = 12, (-reach, reach), (-0.8, 0.8)
    r, _ = rectify(a, c, c + n, unit, ur, vr, ppu)
    lin = srgb2lin(r)
    pink = (pinkness(lin) > 0.9) & (lin[..., 2] > lin[..., 1])
    return ur[0] + np.arange(r.shape[1]) / ppu, pink.mean(0)


def cap_side(a, c, n, unit):
    """+1 / -1: the pink cap lies along +n / -n. 0: not found on exactly one side."""
    u, frac = _cap_profile(a, c, n, unit)
    solid = {sgn: (frac[(u * sgn > 5.0) & (u * sgn < 14.5)] > 0.6).sum() / 12 for sgn in (1, -1)}
    best = max(solid, key=solid.get)
    return best if solid[best] >= 0.8 and solid[-best] < 0.3 else 0


def check_fr(a, Cc, d, unit, info):
    """Verify the First Response layout in the refined frame (d points away from the cap)."""
    if abs(info['Lc_last'] - 1) > 0.12 or abs(info['tilt']) > 0.2 or info['shift'] > 3:
        return None
    ppu, ur, vr = 40, (-6.0, 4.5), (-0.8, 0.8)
    r, _ = rectify(a, Cc, Cc + d, unit, ur, vr, ppu)
    lin = srgb2lin(r)
    u = ur[0] + np.arange(lin.shape[1]) / ppu
    v = vr[0] + np.arange(lin.shape[0]) / ppu
    med = lambda um, vm: np.median(lin[vm][:, um].reshape(-1, 3), 0)
    mem = med((np.abs(u) > 0.2) & (np.abs(u) < 0.45), np.abs(v) < 0.3)
    # white plastic on both sides of the window must look like the membrane
    for zone in (med((u > 1.6) & (u < 3.5), np.abs(v) < 0.5), med((u > -5.0) & (u < -2.6), np.abs(v) < 0.5)):
        if not 0.6 < _lum(zone) / _lum(mem) < 1.9:
            return None
        if abs(_rg(zone) - _rg(mem)) > 0.2 or abs(_gb(zone) - _gb(mem)) > 0.25:
            return None
    ctl = lin[np.abs(v) < 0.35][:, np.abs(u) < 0.1]
    G = cv2.GaussianBlur(ctl[..., 1], (0, 0), 1.0)
    lineness = float(np.percentile(1 - G.min(1) / mem[1], 20))
    pink = float(np.percentile(cv2.GaussianBlur(pinkness(ctl), (0, 0), 1.0).max(1), 20)) - _rg(mem)
    if lineness < 0.25 or pink < 0.25:
        return None
    # the cap must start 5.5 - 9 line-lengths from the control line
    uu, frac = _cap_profile(a, Cc, -d, unit)
    solid = np.flatnonzero((frac > 0.6) & (uu > 4.0))
    if len(solid) < 8 or not 5.5 < uu[solid[0]] < 9.0:
        return None
    return dict(brand='fr', lineness=lineness)


# ---------------------------------------------------------------- driver
def _dedupe(dets):
    out = []
    # 1. candidates sharing a black bar / lying on the same First Response axis: the control line is
    #    the one farthest from the bar (or from the cap); the test line sits closer.
    groups = []
    for dt in dets:
        for g in groups:
            h = g[0]
            if h['kind'] != dt['kind']:
                continue
            if dt['kind'] == 'strip':
                same = np.linalg.norm(h['B'] - dt['B']) < 0.5 * h['width']
            else:
                same = np.linalg.norm(h['C'] - dt['C']) < 1.4 * max(h['unit'], dt['unit'])
            if same:
                g.append(dt)
                break
        else:
            groups.append([dt])
    for g in groups:
        strong = [x for x in g if x['lineness'] >= 0.5 * max(y['lineness'] for y in g)]
        far = max(x['reach'] for x in strong)
        near_far = [x for x in strong if far - x['reach'] < 0.3 * x['width']]
        out.append(max(near_far, key=lambda x: x['lineness'] - 0.02 * x['shift']))
    # 2. anything still overlapping: keep strips over First Response, then the cleaner line
    out.sort(key=lambda x: (x['kind'] != 'strip', -x['lineness']))
    keep = []
    for dt in out:
        if all(np.linalg.norm(dt['C'] - k['C']) > 0.6 * max(dt['width'], k['width']) for k in keep):
            keep.append(dt)
    return keep


def detect(a):
    """Locate all tests in a full-res sRGB image.

    Returns a list of dicts. Strips: kind='strip', brand, B (bar xy), C (control line xy).
    First Response: kind='fr', brand='fr', C (control line xy), d (unit vector away from the cap),
    unit (control line length, px).
    """
    dets = []
    for cand in line_candidates(a):
        # the same line is usually a candidate at several thresholds; once it has been confirmed
        # there is nothing to gain from refining it again
        if any(np.linalg.norm(cand['c'] - dt['C']) < 0.3 * cand['L'] for dt in dets):
            continue
        found = False
        for B0 in find_bar(a, cand):
            try:
                B, C, halfw, info = refine_strip(a, B0, cand['c'], max_iter=4)
            except (LocateError, ValueError, np.linalg.LinAlgError):
                continue
            chk = check_strip(a, B, C, halfw, info)
            if chk:
                unit = float(np.linalg.norm(C - B))
                dets.append(dict(kind='strip', B=B, C=C, unit=unit, width=2 * halfw * unit, reach=unit,
                                 shift=info['shift'], **chk))
                found = True
        if found:
            continue
        n = np.array([cand['t'][1], -cand['t'][0]])
        side = cap_side(a, cand['c'], n, cand['L'])
        if side == 0:
            continue
        try:
            Cc, d, unit, info = refine_fr(a, cand['c'], -side * n, cand['L'], max_iter=4)
        except (LocateError, ValueError, np.linalg.LinAlgError):
            continue
        chk = check_fr(a, Cc, d, unit, info)
        if chk:
            dets.append(dict(kind='fr', C=Cc, d=d, unit=unit, width=unit, reach=float(Cc @ d),
                             shift=info['shift'], **chk))
    return sorted(_dedupe(dets), key=lambda x: (x['brand'], x['C'][1], x['C'][0]))


def render_overlay(a, dets, long_side=1600):
    """A reduced copy of the photo (BGR) with the located landmarks drawn on it."""
    s = min(1.0, long_side / max(a.shape[:2]))
    vis = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
    P = lambda p: (int(round(p[0] * s)), int(round(p[1] * s)))
    for dt in dets:
        col = (0, 200, 0) if dt['brand'] in MEASURED else (160, 160, 160)
        if dt['kind'] == 'strip':
            ax = (dt['C'] - dt['B']) / np.linalg.norm(dt['C'] - dt['B'])
            n = np.array([-ax[1], ax[0]]) * dt.get('width', 0.23 * np.linalg.norm(dt['C'] - dt['B'])) / 2
            cv2.line(vis, P(dt['B']), P(dt['C']), col, 1, cv2.LINE_AA)
            for q in (dt['B'], dt['C']):
                cv2.line(vis, P(q - 1.3 * n), P(q + 1.3 * n), col, 2, cv2.LINE_AA)
            anchor = dt['C'] + 1.6 * n
        else:
            t = np.array([-dt['d'][1], dt['d'][0]]) * dt['unit'] / 2
            cv2.line(vis, P(dt['C'] - 1.2 * t), P(dt['C'] + 1.2 * t), col, 2, cv2.LINE_AA)
            cv2.arrowedLine(vis, P(dt['C']), P(dt['C'] + dt['d'] * dt['unit']), col, 2, cv2.LINE_AA, tipLength=0.3)
            anchor = dt['C'] + 1.5 * t
        text = BRAND_NAMES[dt['brand']] + ('' if dt['brand'] in MEASURED else ' (not measured)')
        x, y = P(anchor)
        cv2.putText(vis, text, (x + 6, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, text, (x + 6, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def draw_overlay(a, dets, path, long_side=1600):
    """Save render_overlay() as a JPEG."""
    cv2.imwrite(path, render_overlay(a, dets, long_side), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Locate pregnancy tests in photos and print their landmarks.')
    ap.add_argument('photos', help='folder of photos')
    ap.add_argument('--overlay', metavar='DIR', help='write a marked-up copy of each photo here')
    args = ap.parse_args()
    if args.overlay:
        os.makedirs(args.overlay, exist_ok=True)
    for path in list_photos(args.photos):
        a, meta = load(path)
        dets = detect(a)
        name = os.path.splitext(os.path.basename(path))[0]
        print(os.path.basename(path), meta.get('dt'))
        for dt in dets:
            extra = f"bar=({dt['B'][0]:.0f}, {dt['B'][1]:.0f})" if dt['kind'] == 'strip' else f"line length={dt['unit']:.0f}px"
            print(f"   {BRAND_NAMES[dt['brand']]:15s} control=({dt['C'][0]:.0f}, {dt['C'][1]:.0f})  {extra}  lineness={dt['lineness']:.2f}")
        if not dets:
            print('   nothing found')
        if args.overlay:
            draw_overlay(a, dets, os.path.join(args.overlay, name + '.jpg'))
        sys.stdout.flush()
