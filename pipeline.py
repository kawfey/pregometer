"""Locate and measure every test in one photo. Shared by the command line and the web app."""
import os

import numpy as np

import collage
from common import LocateError
from detect import BRAND_NAMES, MEASURED, detect
from measure import measure_fr, measure_strip


MAX_SE = 0.2   # a test whose own uncertainty exceeds 20 score points is not reported


def process_photo(a, meta, fname, tests=None, debug_dir=None):
    """a, meta: from common.load(). tests: landmarks to use instead of automatic detection.

    Returns (results, crops, located, notes):
      results -- one dict per measured test (see measure.py), with id / file / located / date fields
      crops   -- {id: canonical linear-RGB crop} for collage.py
      located -- every recognised test with refined landmarks, for detect.draw_overlay()
      notes   -- one line of text per test, describing what happened
    """
    name = os.path.splitext(fname)[0]
    how = 'manual' if tests is not None else 'auto'
    if tests is None:
        tests = detect(a)
    results, crops, located, notes = [], {}, [], []
    for i, t in enumerate(tests):
        label = BRAND_NAMES[t['brand']]
        if t['brand'] not in MEASURED:
            notes.append(f'{label}: recognised, not measured')
            located.append(t)
            continue
        try:
            if t['kind'] == 'strip':
                res, crop = measure_strip(t['brand'], name, a, meta, t['B'], t['C'], debug_dir)
                t = dict(t, B=np.array(res['bar_xy']), C=np.array(res['control_xy']), width=2 * res['halfw'] * res['unit_px'])
            else:
                res, crop = measure_fr(name, a, meta, t['C'], t['d'], t['unit'], debug_dir)
                t = dict(t, C=np.array(res['control_xy']), d=np.array(res['away_from_cap']), unit=res['unit_px'])
        except LocateError as e:
            notes.append(f'{label}: could not be measured ({e})')
            continue
        if res['ratio_se'] > MAX_SE:
            # the score swings wildly across the line's own length: not a real test line pair
            # (e.g. a red speck on the table that happened to pass the layout checks)
            notes.append(f"{label}: found but not measurable (uncertainty +/- {100 * res['ratio_se']:.0f} points); skipped")
            continue
        res.update(id=f"{t['brand']}_{name}_{i}", file=fname, located=how,
                   dt_time=meta.get('dt_time', True), dt_source=meta.get('dt_source'))
        results.append(res)
        crops[res['id']] = crop
        located.append(t)
        notes.append(f"{label}: score {collage.score_of(res)[1]}  (test/control = {100 * res['T'] / res['C']:.1f}% +/- {100 * res['ratio_se']:.1f})")
    if not tests:
        notes.append('no test found')
    return results, crops, located, notes
