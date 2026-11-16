# Pregometer

A web page that compares the test line on home pregnancy tests across several days, from
ordinary phone photos.

**Use it:** https://kawfey.github.io/pregometer/

Add your photos, check the dates, and you get a report: a score for every test, a chart per
brand, colour-matched crops side by side, and the photos marked up with what was found. The
report can be saved as one HTML file or printed to PDF.

**Photos are not uploaded.** The analysis runs in your browser. There is no server, no
account, no analytics, and the page loads nothing from other domains.

## Supported tests

| Test | Looks like | Must be visible in the photo |
|---|---|---|
| Wondfo | paper strip, orange label | black MAX bar and the start of the label |
| easy@Home HCG | paper strip, light pink label | black MAX bar and the start of the label |
| First Response | white stick, oval window, pink cap | the pink cap, on the stick |

Photograph from straight above, in focus, in even light, with the test filling a good part of
the frame. One photo can show several tests. HEIC, JPEG and PNG all work.

## What the score means

Score = darkness of the test line as a percentage of the control line on the same test, from
0 to 100. A score of 100 means the two lines are equally dark.

1. The photo is converted to linear light (wide-gamut iPhone photos are converted to sRGB).
2. Each test is located and straightened into a fixed frame.
3. A profile is taken across the lines. For each line, a straight baseline is fitted to the
   blank strip on both sides, and the fraction of green light absorbed is summed across the
   line. Dividing by the baseline removes exposure and white balance; summing makes the result
   insensitive to focus blur.
4. Score = test line ÷ control line × 100.

## What it cannot tell you

- It is not a medical device and does not measure hCG. Line darkness also depends on how
  dilute the urine was, the time of day, the individual test, and how long the test had been
  drying when it was photographed.
- Compare scores only within one brand.
- Scores below about 5 are at the edge of what a phone photo can show. Two photos of the same
  test usually differ by a few points.
- A score cannot confirm or rule out a pregnancy or say how one is progressing. A clinician
  can order a blood hCG test, which is the real measurement.
- The easy@Home test line position is assumed to match the Wondfo layout and has not been
  calibrated against a visible line.

## How tests are found

Every supported test has a strongly pink control line on a white background, so the search
starts there.

1. Find thin pink line segments anywhere in the photo.
2. For each one, look at what surrounds it. A strip has a black bar as wide as the line a few
   line-lengths away, white membrane in between, and a coloured label on the far side; the
   label colour gives the brand. A First Response has white plastic on both sides and the pink
   cap 6 to 8 line-lengths away.
3. Refine each position to sub-pixel accuracy, check the layout again in the refined frame,
   and drop duplicates.

## How it runs

The analysis is Python (NumPy, OpenCV, Pillow) running in the browser through
[Pyodide](https://pyodide.org) inside a web worker. HEIC photos are decoded with
[libheif-js](https://github.com/catdad-experiments/libheif-js). Photos larger than 4100 px on
the long side are reduced first, to stay within browser memory. Expect about 6 seconds per
photo on a desktop browser.

| File | What |
|---|---|
| `web/index.html`, `web/style.css`, `web/app.js` | the page |
| `web/worker.js` | web worker: runs the Python code, decodes HEIC |
| `web/build.py` | assembles the site into `dist/` |
| `webapi.py` | the functions the page calls |
| `pipeline.py` | locate and measure every test in one photo |
| `detect.py` | finds the tests |
| `measure.py` | refines positions and measures the lines |
| `collage.py` | normalised crops and collages |
| `common.py` | image loading, colour conversion, resampling |

## Build it yourself

Needs Python 3.10 or newer; the build uses only the standard library.

```
python web/build.py
python -m http.server 8000 --directory dist
```

Then open http://localhost:8000. The first build downloads the Pyodide release archive (about
400 MB) and keeps the 30 MB it needs in `web/.pyodide-cache/`. `dist/` is the whole site and
can be served by any static host. Pushing to `main` deploys it to GitHub Pages through
`.github/workflows/pages.yml`.

## Bundled components

The deployed site includes Pyodide (MPL-2.0), NumPy (BSD-3-Clause), OpenCV (Apache-2.0),
Pillow (MIT-CMU) and libheif-js with libheif and libde265 (LGPL-3.0). Their licence texts are
inside the respective packages.
