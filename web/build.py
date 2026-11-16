"""Assemble the static web app into dist/.

    python web/build.py [--archive pyodide-X.tar.bz2] [--out dist]

dist/ then holds everything the site needs and can be served by any static host:
    index.html, style.css, app.js, worker.js   the page
    py/*.py                                    the analysis code (same files the command line uses)
    pyodide/*                                  the Python runtime for the browser, plus the packages used
    libheif/*                                  HEIC decoder

The runtime is taken from the official Pyodide release archive. Only the handful of files that are
needed (about 30 MB of the 400 MB archive) are kept, in web/.pyodide-cache/, so the archive is
downloaded once. Pass --archive to use a copy you already have.
"""
import argparse
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request

PYODIDE = '0.29.5'
URL = f'https://github.com/pyodide/pyodide/releases/download/{PYODIDE}/pyodide-{PYODIDE}.tar.bz2'
CORE = ['pyodide.js', 'pyodide.asm.js', 'pyodide.asm.wasm', 'python_stdlib.zip', 'pyodide-lock.json']
PACKAGES = ['numpy', 'opencv-python', 'pillow']       # loaded by worker.js
PREFIXES = ('numpy-', 'opencv_python-', 'pillow-')
# HEIC decoder. Pyodide's own (libheif 1.17) rejects photos from current iPhones, so the page
# decodes HEIC with this newer build and hands the pixels to Python.
LIBHEIF = '1.23.5'
LIBHEIF_URL = f'https://registry.npmjs.org/libheif-js/-/libheif-js-{LIBHEIF}.tgz'
LIBHEIF_FILES = {'package/libheif-wasm/libheif-bundle.js': 'libheif-bundle.js', 'package/libheif-wasm/LICENSE': 'LICENSE'}
HEIF_CACHE = None   # set below
MODULES = ['common', 'measure', 'detect', 'collage', 'pipeline', 'webapi']
STATIC = ['index.html', 'style.css', 'app.js', 'worker.js']

WEB = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(WEB)
CACHE = os.path.join(WEB, '.pyodide-cache', PYODIDE)
HEIF_CACHE = os.path.join(WEB, '.pyodide-cache', 'libheif-' + LIBHEIF)


def needed_files(lock):
    """Core files plus the wheel of every package PACKAGES depend on, according to the lock file."""
    pk, seen = lock['packages'], set()

    def walk(n):
        if n not in seen:
            seen.add(n)
            for d in pk[n]['depends']:
                walk(d)
    for n in PACKAGES:
        walk(n)
    return CORE + sorted(pk[n]['file_name'] for n in seen)


def cache_complete():
    lock = os.path.join(CACHE, 'pyodide-lock.json')
    if not os.path.exists(lock):
        return False
    return all(os.path.exists(os.path.join(CACHE, f)) for f in needed_files(json.load(open(lock))))


def fetch(url, tries=4):
    """Open a URL, retrying a few times: package hosts drop the odd connection."""
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=60)
        except OSError as e:
            if attempt == tries - 1:
                sys.exit(f'could not download {url}: {e}')
            print(f'  download failed ({e}); retrying ...', flush=True)
            time.sleep(5 * (attempt + 1))


def fill_cache(archive):
    os.makedirs(CACHE, exist_ok=True)
    print(f'extracting the Pyodide {PYODIDE} runtime from {archive or URL} ...', flush=True)
    src = open(archive, 'rb') if archive else fetch(URL)
    with src, tarfile.open(fileobj=src, mode='r|bz2') as tar:      # one streaming pass, nothing else stored
        for m in tar:
            base = os.path.basename(m.name)
            if m.isfile() and (base in CORE or base.startswith(PREFIXES)):
                with tar.extractfile(m) as f, open(os.path.join(CACHE, base), 'wb') as o:
                    shutil.copyfileobj(f, o)
                if cache_complete():
                    break          # everything needed is out; skip the rest of the archive
    if not cache_complete():
        sys.exit('the archive did not contain every file the app needs; check PACKAGES / PREFIXES')


def fill_heif_cache():
    os.makedirs(HEIF_CACHE, exist_ok=True)
    print(f'fetching the HEIC decoder (libheif-js {LIBHEIF}) ...', flush=True)
    with fetch(LIBHEIF_URL) as src, tarfile.open(fileobj=src, mode='r|gz') as tar:
        for m in tar:
            if m.name in LIBHEIF_FILES:
                with tar.extractfile(m) as f, open(os.path.join(HEIF_CACHE, LIBHEIF_FILES[m.name]), 'wb') as o:
                    shutil.copyfileobj(f, o)
    if not all(os.path.exists(os.path.join(HEIF_CACHE, f)) for f in LIBHEIF_FILES.values()):
        sys.exit('the libheif-js package did not contain the expected files')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--archive', help='a local copy of the Pyodide release archive')
    ap.add_argument('--out', default=os.path.join(ROOT, 'dist'))
    args = ap.parse_args()
    if not cache_complete():
        fill_cache(args.archive)
    if not all(os.path.exists(os.path.join(HEIF_CACHE, f)) for f in LIBHEIF_FILES.values()):
        fill_heif_cache()

    out = args.out
    for sub in ('py', 'pyodide', 'libheif'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    for f in STATIC:
        shutil.copy2(os.path.join(WEB, f), os.path.join(out, f))
    for m in MODULES:
        shutil.copy2(os.path.join(ROOT, m + '.py'), os.path.join(out, 'py', m + '.py'))
    files = needed_files(json.load(open(os.path.join(CACHE, 'pyodide-lock.json'))))
    for f in set(os.listdir(os.path.join(out, 'pyodide'))) - set(files):      # left over from an older build
        os.remove(os.path.join(out, 'pyodide', f))
    for f in files:
        dst = os.path.join(out, 'pyodide', f)
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(os.path.join(CACHE, f)):
            shutil.copy2(os.path.join(CACHE, f), dst)
    for f in LIBHEIF_FILES.values():
        shutil.copy2(os.path.join(HEIF_CACHE, f), os.path.join(out, 'libheif', f))
    open(os.path.join(out, '.nojekyll'), 'w').close()
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out) for f in fs)
    print(f'built {out}  ({size / 1e6:.0f} MB, {len(files)} runtime files)')
    print(f'try it:  python -m http.server 8000 --directory "{out}"   then open http://localhost:8000')


if __name__ == '__main__':
    main()
