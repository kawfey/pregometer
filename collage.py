"""Build one vertical collage per brand, oldest photo at the top, plus scores.csv."""
import csv
import datetime as dt
import functools
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import lin2srgb
from measure import FR_FRAME, STRIP_FRAME


def _s2l(v):
    v = v / 255.0
    return ((v + 0.055) / 1.055) ** 2.4


WT, BT = _s2l(232), _s2l(22)   # display targets for the reference white / black
GAIN = 3.0                     # contrast boost of the close-up panel (same for every row)
MIN_PCT = 1.5                  # below this the test line is reported as not measurable

# A = range shown in the normalised crop, E = range of the close-up (u range, v half-height),
# blur = close-up smoothing (across the line, along the line) in canonical units
GEO = {
    'wondfo': dict(**STRIP_FRAME, A=((-0.12, 1.38), (-0.15, 0.15)), Aw=1000, E=((0.685, 1.105), 0.06), Ew=1000, blur=(0.002, 0.012)),
    'hcg':    dict(**STRIP_FRAME, A=((-0.12, 1.38), (-0.15, 0.15)), Aw=1000, E=((0.64, 1.18), 0.077), Ew=1000, blur=(0.002, 0.012)),
    'fr':     dict(**FR_FRAME, A=((-1.75, 1.25), (-0.72, 0.72)), Aw=800, E=((-1.35, 0.45), 0.35), Ew=1000, blur=(0.004, 0.02)),
}
TITLES = {'wondfo': 'Wondfo strips', 'hcg': 'easy@Home HCG strips', 'fr': 'First Response'}
FILES = {'wondfo': 'collage_wondfo.png', 'fr': 'collage_first_response.png', 'hcg': 'collage_hcg.png'}
INK, SUB, RULE, BG, ACC = (28, 28, 32), (105, 105, 115), (222, 222, 228), (250, 250, 251), (190, 40, 95)


@functools.lru_cache(maxsize=None)
def font(sz, bold=False):
    names = (['seguisb.ttf', 'segoeuib.ttf', 'arialbd.ttf', 'DejaVuSans-Bold.ttf'] if bold
             else ['segoeui.ttf', 'arial.ttf', 'DejaVuSans.ttf'])
    for f in names:
        for folder in ('C:/Windows/Fonts', ''):
            try:
                return ImageFont.truetype(os.path.join(folder, f), sz)
            except OSError:
                pass
    return ImageFont.load_default(sz)


def normalise(lin, r):
    """Map this test's own blank membrane to a fixed white (and its black bar to a fixed black)."""
    w = np.array(r['white'], np.float32)
    if r.get('black') is not None:
        b = np.array(r['black'], np.float32)
        return BT + (lin - b) * (WT - BT) / (w - b)
    return lin * (WT / w)


def cut(img, g, ur, vr, width):
    ppu = g['ppu']
    x0, x1 = (int(round((x - g['ur'][0]) * ppu)) for x in ur)
    y0, y1 = (int(round((y - g['vr'][0]) * ppu)) for y in vr)
    c = img[y0:y1, x0:x1]
    h = int(round(width * c.shape[0] / c.shape[1]))
    return cv2.resize(c, (width, h), interpolation=cv2.INTER_AREA if width < c.shape[1] else cv2.INTER_CUBIC)


def to8(lin):
    return (lin2srgb(np.clip(lin, 0, 1)) * 255 + 0.5).astype(np.uint8)


def parse_dt(s):
    try:
        return dt.datetime.strptime(s, '%Y:%m:%d %H:%M:%S')
    except (TypeError, ValueError):
        return None


def order_key(r):
    """Capture time; photos without one sort after the dated ones, by filename.

    A date without a time of day (from a filename) sorts at the start of that day."""
    d = parse_dt(r['dt'])
    return (d is None, d or dt.datetime.min, r['image'])


def score_of(r):
    ratio = r['T'] / r['C']
    return ratio, int(round(min(100, max(0, 100 * ratio))))


def panels(r, lin):
    """The two pictures of one row: (normalised crop, contrast-boosted close-up, close-up u range)."""
    g = GEO[r['brand']]
    n = normalise(lin, r)
    A = to8(cut(n, g, g['A'][0], g['A'][1], g['Aw']))
    (e0, e1), ev = g['E']
    # smooth mostly along the line direction (lines run vertically in the crop) so sensor and
    # compression noise does not swamp faint lines once contrast is boosted
    sm = cv2.GaussianBlur(n, (0, 0), sigmaX=g['ppu'] * g['blur'][0], sigmaY=g['ppu'] * g['blur'][1])
    E = cut(sm, g, (e0, e1), (-ev, ev), g['Ew'])
    E = to8(WT * np.clip(E / WT, 0, None) ** GAIN)
    return A, E, (e0, e1)


def _tile(r, lin, g, Wd, M, GAP, ready=None):
    A, E, (e0, e1) = ready or panels(r, lin)
    ratio, score = score_of(r)
    d = parse_dt(r['dt'])
    if d:
        label = d.strftime('%a %b %d, %Y').replace(' 0', ' ') + '  ·  '
        label += d.strftime('%I:%M %p').lstrip('0') if r.get('dt_time', True) else 'time not recorded'
    else:
        label = 'date unknown'
    src = r['file'] + ('  ·  date from filename' if r.get('dt_source') == 'filename' else '')
    ph = max(A.shape[0], E.shape[0])
    th = 78 + ph + 92
    t = Image.new('RGB', (Wd, th), BG)
    dr = ImageDraw.Draw(t)
    dr.text((M, 10), label, font=font(34, True), fill=INK)
    dr.text((M + dr.textlength(label, font=font(34, True)) + 22, 24), src, font=font(20), fill=SUB)
    st, f1, f2, tail = f'{score}', font(46, True), font(22), ' / 100'
    wscore, wtail = dr.textlength(st, font=f1), dr.textlength(tail, font=f2)
    cap = f"test line = {100 * ratio:.1f}% of control line   " if 100 * ratio >= MIN_PCT else "no measurable test line   "
    x = Wd - M - wtail - wscore
    dr.text((x - dr.textlength(cap, font=f2) - dr.textlength('Score  ', font=f2), 28), cap + 'Score', font=f2, fill=SUB)
    dr.text((x, 2), st, font=f1, fill=ACC)
    dr.text((x + wscore, 28), tail, font=f2, fill=SUB)
    y = 74
    xe = M + g['Aw'] + GAP
    t.paste(Image.fromarray(A), (M, y + (ph - A.shape[0]) // 2))
    t.paste(Image.fromarray(E), (xe, y + (ph - E.shape[0]) // 2))
    fs = font(19)
    for pos, name in ((r['T_pos'], 'test line'), (r['C_pos'], 'control')):
        px = xe + (pos - e0) / (e1 - e0) * g['Ew']
        dr.polygon([(px, y + ph + 5), (px - 8, y + ph + 19), (px + 8, y + ph + 19)], fill=ACC)
        dr.text((px - dr.textlength(name, font=fs) / 2, y + ph + 22), name, font=fs, fill=INK)
    dr.text((M, y + ph + 52), 'white balance + brightness normalised', font=fs, fill=SUB)
    dr.text((xe, y + ph + 52), f'close-up of the result window, contrast boosted ×{GAIN:g}, lightly de-noised', font=fs, fill=SUB)
    dr.line([(M, th - 1), (Wd - M, th - 1)], fill=RULE, width=2)
    return t


def build(results, crops, out_dir, ready=None):
    """results: list of dicts from measure.py (each with 'id' and 'file'); crops: {id: linear crop}.

    ready: optional {id: panels(...) result} for rows whose pictures were already rendered."""
    ready = ready or {}
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for brand in ('wondfo', 'fr', 'hcg'):
        items = sorted([r for r in results if r['brand'] == brand], key=order_key)
        if not items:
            continue
        g = GEO[brand]
        M, GAP = 44, 34
        Wd = M + g['Aw'] + GAP + g['Ew'] + M
        tiles = [_tile(r, crops[r['id']], g, Wd, M, GAP, ready.get(r['id'])) for r in items]
        head_h, foot_h = 150, 112
        page = Image.new('RGB', (Wd, head_h + sum(t.height for t in tiles) + foot_h), BG)
        dr = ImageDraw.Draw(page)
        dr.text((M, 26), TITLES[brand] + ' — test line over time', font=font(50, True), fill=INK)
        dr.text((M, 96), f'{len(tiles)} photo{"s" if len(tiles) != 1 else ""}, oldest at top. Left: normalised crop. '
                         'Right: same crop, result window only, contrast boosted identically in every row.', font=font(22), fill=SUB)
        dr.line([(M, head_h - 6), (Wd - M, head_h - 6)], fill=RULE, width=2)
        y = head_h
        for t in tiles:
            page.paste(t, (0, y))
            y += t.height
        notes = ['Score = darkness of the test line as a percentage of the control line (green-channel light absorbed, '
                 'background-subtracted, summed across the line). 100 = as dark as control.',
                 'Each test is normalised to its own blank membrane' + (' and black MAX bar' if brand != 'fr' else '') +
                 ', so lighting and camera white balance differences between photos are removed.']
        for i, s in enumerate(notes):
            dr.text((M, y + 22 + i * 34), s, font=font(20), fill=SUB)
        page.save(os.path.join(out_dir, FILES[brand]))
        written.append((FILES[brand], page.size))
    rows = []
    for r in sorted(results, key=lambda r: (('wondfo', 'fr', 'hcg').index(r['brand']),) + order_key(r)):
        ratio, score = score_of(r)
        d = parse_dt(r['dt'])
        rows.append(dict(brand=TITLES[r['brand']], image=r['file'], taken=d.strftime('%Y-%m-%d %H:%M:%S' if r.get('dt_time', True) else '%Y-%m-%d') if d else '',
                         date_source=r.get('dt_source') or '',
                         located=r['located'], score_0_100=score, test_over_control_pct=round(100 * ratio, 1),
                         within_photo_se_pct=round(100 * r['ratio_se'], 1),
                         test_line_strength=round(r['T'] * 1e3, 2), control_line_strength=round(r['C'] * 1e3, 2)))
    if rows:
        with open(os.path.join(out_dir, 'scores.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return written, rows
