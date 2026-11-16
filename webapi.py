"""Entry points for the browser version (runs under Pyodide inside a web worker).

The page sends one photo at a time; results accumulate here until finish() builds the collages.
Nothing leaves the browser: every function works on bytes in memory.
"""
import io
import json
import os

import cv2
import numpy as np

import collage
import common
from detect import render_overlay
from pipeline import process_photo

MAX_SIDE = 4100    # larger photos are reduced first, to stay within browser memory
_state = dict(results=[], crops={}, images={}, panels={}, photos=0)


def reset():
    _state.update(results=[], crops={}, images={}, panels={}, photos=0)


def _jpeg(rgb, quality=90):
    ok, buf = cv2.imencode('.jpg', rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _keep(key, data):
    _state['images'][key] = data
    return key


def probe(name, src):
    """The photo's capture date, without analysing it. JSON: {dt, dt_time, dt_source}."""
    m = common.photo_date(src, name)
    return json.dumps(dict(dt=m['dt'], dt_time=m['dt_time'], dt_source=m['dt_source']))


def analyze(name, src, dt=None, dt_time=True, dt_source=None, pixels=None, width=0, height=0):
    """Analyse one photo (a path, or the file's bytes). dt ('YYYY:MM:DD HH:MM:SS') overrides its own date.

    pixels/width/height: a file of raw RGBA bytes, for formats the page decodes itself (HEIC).

    Returns JSON: {name, dt, tests: [...], notes: [...], overlay: image key}.
    Images are fetched separately with image(key).
    """
    if pixels:
        rgb = np.ascontiguousarray(np.fromfile(pixels, np.uint8).reshape(int(height), int(width), 4)[:, :, :3])
        a, meta = common.load_decoded(rgb, src, name=name, max_side=MAX_SIDE)
        del rgb
    else:
        a, meta = common.load(src, name=name, max_side=MAX_SIDE)
    if dt:
        meta.update(dt=dt, dt_time=bool(dt_time), dt_source=dt_source or 'entered')
    n = _state['photos']
    _state['photos'] += 1
    results, crops, located, notes = process_photo(a, meta, name)
    tests = []
    for r in results:
        r['id'] = f"{n}_{r['id']}"          # two uploads may share a filename
        lin = crops.pop(r['id'].split('_', 1)[1])
        _state['crops'][r['id']] = lin
        A, E, (e0, e1) = _state['panels'][r['id']] = collage.panels(r, lin)
        ratio, score = collage.score_of(r)
        tests.append(dict(
            id=r['id'], brand=r['brand'], brand_title=collage.TITLES[r['brand']], score=score,
            pct=round(100 * ratio, 1), se=round(100 * r['ratio_se'], 1), measurable=100 * ratio >= collage.MIN_PCT,
            test_pos=(r['T_pos'] - e0) / (e1 - e0), control_pos=(r['C_pos'] - e0) / (e1 - e0),
            crop=_keep(r['id'] + '/crop', _jpeg(A, 92)), closeup=_keep(r['id'] + '/closeup', _jpeg(E, 92))))
    _state['results'] += results
    overlay = _keep(f'{n}/overlay', _jpeg(render_overlay(a, located, 1200)[:, :, ::-1], 80))
    return json.dumps(dict(name=name, dt=meta.get('dt'), dt_time=meta.get('dt_time', True), dt_source=meta.get('dt_source'),
                           width=a.shape[1], height=a.shape[0], tests=tests, notes=notes, overlay=overlay))


def image(key):
    return _state['images'][key]


def finish():
    """Build the collages from everything analysed so far. Returns JSON: {collages: [...], csv, rows}."""
    out = '/tmp/out'
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    written, rows = collage.build(_state['results'], _state['crops'], out, _state['panels'])
    collages = []
    for fname, size in written:
        brand = next(b for b, f in collage.FILES.items() if f == fname)
        with open(os.path.join(out, fname), 'rb') as f:
            collages.append(dict(brand=brand, title=collage.TITLES[brand], file=fname, width=size[0], height=size[1],
                                 image=_keep('collage/' + brand, f.read())))
    csv = ''
    if rows:
        with open(os.path.join(out, 'scores.csv'), newline='') as f:
            csv = f.read()
    order = {r['id']: i for i, r in enumerate(sorted(_state['results'], key=collage.order_key))}
    return json.dumps(dict(collages=collages, csv=csv, rows=rows, order=order))
