/* Pregometer: page logic. All analysis happens in worker.js; nothing is sent anywhere. */
'use strict';

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const BRANDS = ['wondfo', 'fr', 'hcg'];
const BRAND_NAME = { wondfo: 'Wondfo', fr: 'First Response', hcg: 'easy@Home HCG' };

// ---------------------------------------------------------------- worker
const worker = new Worker('worker.js');
const pending = new Map();
let seq = 0;
worker.onmessage = (e) => {
  const m = e.data;
  if (m.progress) { $('#engine').textContent = m.progress; return; }
  const p = pending.get(m.id);
  pending.delete(m.id);
  if (m.ok) p.resolve(m.result); else p.reject(new Error(m.error));
};
function call(type, payload, transfer = []) {
  return new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, type, payload }, transfer);
  });
}
let engineOk = false;
const ready = call('init').then(() => {
  engineOk = true;
  $('#engine').textContent = '';
  refreshButtons();
}).catch((err) => {
  $('#engine').textContent = `The analysis engine could not start (${err.message}). Try a current version of Chrome, Edge, Firefox or Safari.`;
  throw err;
});
ready.catch(() => {});

// ---------------------------------------------------------------- dates
// Python side uses 'YYYY:MM:DD HH:MM:SS' (the EXIF format)
const pyToParts = (s) => (s ? { date: s.slice(0, 10).replace(/:/g, '-'), time: s.slice(11, 16) } : { date: '', time: '' });
const partsToPy = (date, time) => `${date.replace(/-/g, ':')} ${time || '00:00'}:00`;
function whenLabel(dt, hasTime) {
  if (!dt) return 'Date unknown';
  const [y, mo, d, h, mi] = dt.split(/[: ]/).map(Number);
  const t = new Date(y, mo - 1, d, h, mi);
  const day = t.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });
  return hasTime ? `${day} · ${t.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })}` : `${day} · time not recorded`;
}
const dtMillis = (dt) => { const [y, mo, d, h, mi, s] = dt.split(/[: ]/).map(Number); return new Date(y, mo - 1, d, h, mi, s).getTime(); };
const shortDay = (ms) => new Date(ms).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });

// ---------------------------------------------------------------- photo list
let photos = [];
let busy = false;
let nextKey = 1;
const canPreview = (f) => /^image\/(jpeg|png|webp|gif)$/.test(f.type);
const isPhoto = (f) => /^image\//.test(f.type) || /\.(heic|heif|jpe?g|png)$/i.test(f.name);

function addFiles(list) {
  for (const file of list) {
    if (!isPhoto(file)) continue;
    const p = { key: nextKey++, file, name: file.name, date: '', time: '', source: null, own: null, edited: false, probed: false, note: '', noteClass: '' };
    if (canPreview(file)) p.thumb = URL.createObjectURL(file);
    photos.push(p);
    probe(p);
  }
  renderList();
}

async function probe(p) {
  try {
    const buf = await p.file.arrayBuffer();
    const r = await call('probe', { name: p.name, buf }, [buf]);
    p.probed = true;
    if (!p.edited && r.dt) {
      Object.assign(p, pyToParts(r.dt));
      if (!r.dt_time) p.time = '';
      p.source = r.dt_source;
      p.own = { date: p.date, time: p.time };
    }
  } catch (err) {
    p.probed = true;
    p.unreadable = true;
  }
  updateRow(p);
}

function rowNote(p) {
  if (p.note) return [p.note, p.noteClass];
  if (p.unreadable) return ['This file could not be read as a photo', 'warn'];
  if (!p.probed) return ['Reading date…', ''];
  if (!p.date) return ['No date in this photo. Enter when it was taken', 'warn'];
  if (p.edited) return ['Date entered by you', ''];
  return [p.source === 'filename' ? 'Date from the file name' : 'Date from the photo', ''];
}

function renderList() {
  $('#list-card').hidden = photos.length === 0;
  const ul = $('#photos');
  ul.textContent = '';
  for (const p of photos) {
    const li = document.createElement('li');
    li.innerHTML = `
      ${p.thumb ? `<img class="thumb" alt="" src="${p.thumb}">` : `<div class="thumb">${esc((p.name.split('.').pop() || '').toUpperCase().slice(0, 4))}</div>`}
      <div><div class="pname">${esc(p.name)}</div><div class="pnote"></div></div>
      <div class="when">
        <input type="date" aria-label="Date taken, ${esc(p.name)}">
        <input type="time" aria-label="Time taken, ${esc(p.name)}">
      </div>
      <button class="x" title="Remove" aria-label="Remove ${esc(p.name)}">×</button>`;
    p.el = li;
    const [d, t] = li.querySelectorAll('input');
    const onEdit = () => { p.date = d.value; p.time = t.value; p.edited = !(p.own && p.own.date === p.date && p.own.time === p.time); updateRow(p, true); };
    d.addEventListener('change', onEdit);
    t.addEventListener('change', onEdit);
    li.querySelector('.x').addEventListener('click', () => {
      if (busy) return;
      if (p.thumb) URL.revokeObjectURL(p.thumb);
      photos = photos.filter((q) => q !== p);
      renderList();
    });
    ul.appendChild(li);
    updateRow(p);
  }
  refreshButtons();
}

function updateRow(p, keepInputs) {
  if (!p.el) return;
  const [text, cls] = rowNote(p);
  const note = p.el.querySelector('.pnote');
  note.textContent = text;
  note.className = `pnote ${cls}`;
  if (!keepInputs) {
    const [d, t] = p.el.querySelectorAll('input');
    d.value = p.date;
    t.value = p.time;
  }
}

function refreshButtons() {
  $('#go').disabled = busy || !engineOk || photos.length === 0;
  $('#clear').disabled = busy;
}

$('#files').addEventListener('change', (e) => { addFiles(e.target.files); e.target.value = ''; });
const drop = $('#drop');
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => addFiles(e.dataTransfer.files));
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => e.preventDefault());
$('#clear').addEventListener('click', () => { photos.forEach((p) => p.thumb && URL.revokeObjectURL(p.thumb)); photos = []; renderList(); });
$('#go').addEventListener('click', analyse);

// ---------------------------------------------------------------- analysis
const images = new Map();   // blob URL -> { bytes, type }, kept so the saved report can embed them
function imageUrl(bytes, type) {
  const url = URL.createObjectURL(new Blob([bytes], { type }));
  images.set(url, { bytes, type });
  return url;
}

async function analyse() {
  busy = true;
  refreshButtons();
  const status = $('#status'), bar = $('#bar');
  bar.hidden = false;
  $('#report').hidden = true;
  for (const url of images.keys()) URL.revokeObjectURL(url);
  images.clear();
  try {
    await ready;
    await call('reset');
    const done = [];
    for (let i = 0; i < photos.length; i++) {
      const p = photos[i];
      status.textContent = `Analysing photo ${i + 1} of ${photos.length}…`;
      bar.firstElementChild.style.width = `${(100 * i) / photos.length}%`;
      Object.assign(p, { note: 'Analysing…', noteClass: '' });
      updateRow(p, true);
      try {
        const buf = await p.file.arrayBuffer();
        const payload = { name: p.name, buf };
        if (p.date && (p.edited || !p.own)) Object.assign(payload, { dt: partsToPy(p.date, p.time), dtTime: !!p.time, dtSource: 'entered' });
        const r = await call('analyze', payload, [buf]);
        r.overlayUrl = imageUrl(r.overlay, 'image/jpeg');
        for (const t of r.tests) { t.cropUrl = imageUrl(t.crop, 'image/jpeg'); t.closeupUrl = imageUrl(t.closeup, 'image/jpeg'); }
        done.push(r);
        const n = r.tests.length;
        Object.assign(p, n ? { note: `${n} test${n > 1 ? 's' : ''} measured: ${r.tests.map((t) => `${BRAND_NAME[t.brand]} ${t.score}`).join(', ')}`, noteClass: 'done' }
          : { note: 'No supported test found in this photo', noteClass: 'warn' });
      } catch (err) {
        done.push({ name: p.name, tests: [], notes: [`could not be analysed (${err.message})`], failed: true });
        Object.assign(p, { note: 'This photo could not be analysed', noteClass: 'warn' });
      }
      updateRow(p, true);
    }
    status.textContent = 'Building the report…';
    bar.firstElementChild.style.width = '100%';
    const fin = await call('finish');
    for (const c of fin.collages) c.url = imageUrl(c.image, 'image/png');
    renderReport(done, fin);
    status.textContent = 'Done. The report is below.';
  } catch (err) {
    status.textContent = `Something went wrong: ${err.message}`;
  }
  bar.hidden = true;
  busy = false;
  refreshButtons();
}

// ---------------------------------------------------------------- report
function renderReport(done, fin) {
  const rows = [];
  for (const r of done) for (const t of r.tests) rows.push({ ...t, name: r.name, dt: r.dt, hasTime: r.dt_time, dtSource: r.dt_source });
  rows.sort((a, b) => (!a.dt - !b.dt) || (a.dt || '').localeCompare(b.dt || '') || a.name.localeCompare(b.name));
  const none = done.filter((r) => r.tests.length === 0);
  const ymax = Math.max(20, Math.min(100, Math.ceil((Math.max(0, ...rows.map((r) => r.score)) * 1.15) / 20) * 20));
  const today = new Date().toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' });

  let html = `<header>
      <div><h2>Test line report</h2>
      <p class="made">${rows.length} test${rows.length === 1 ? '' : 's'} measured in ${done.length} photo${done.length === 1 ? '' : 's'} · made ${esc(today)}</p></div>
      <div class="downloads no-print no-export">
        <button id="save-html">Save report (HTML)</button>
        <button id="save-pdf">Print or save as PDF</button>
        ${fin.collages.map((c) => `<a href="${c.url}" download="${esc(c.file)}">Collage: ${esc(BRAND_NAME[c.brand])} (PNG)</a>`).join('')}
        ${fin.csv ? '<a id="save-csv" href="#">Scores (CSV)</a>' : ''}
      </div>
    </header>`;
  if (none.length) {
    html += `<p class="notice"><strong>No supported test was found in ${none.length} photo${none.length === 1 ? '' : 's'}:</strong> ${none.map((r) => esc(r.name)).join(', ')}.
      The test may be a brand this tool does not know, too small in the frame, or missing its black bar, label or cap. See “What was located” below.</p>`;
  }
  if (!rows.length) {
    html += '<p class="notice">Nothing could be measured. Check the photo tips under “Which tests work” above and try again.</p>';
  }
  for (const brand of BRANDS) {
    const set = rows.filter((r) => r.brand === brand);
    if (!set.length) continue;
    const dated = set.filter((r) => r.dt);
    let sum = `${set.length} photo${set.length === 1 ? '' : 's'}.`;
    if (dated.length > 1) {
      const a = dated[0], b = dated[dated.length - 1];
      sum += ` First: score ${a.score} on ${shortDay(dtMillis(a.dt))}. Latest: score ${b.score} on ${shortDay(dtMillis(b.dt))}.`;
    }
    html += `<section class="brand"><h3>${esc(BRAND_NAME[brand])}</h3><p class="sum">${sum}</p>${chart(dated, ymax, brand)}`;
    for (const r of set) {
      html += `<article class="row">
        <header><h4>${esc(whenLabel(r.dt, r.hasTime))}</h4>
          <span class="file">${esc(r.name)}${r.dtSource === 'filename' ? ' · date from file name' : r.dtSource === 'entered' ? ' · date entered by hand' : ''}</span>
          <span class="score">${r.measurable ? `test line = ${r.pct.toFixed(1)}% of control line` : 'no measurable test line'}<b>${r.score}</b> / 100</span></header>
        <div class="panels">
          <figure><img src="${r.cropUrl}" alt="Colour-corrected crop of the test"><figcaption>White balance and brightness normalised</figcaption></figure>
          <figure><img src="${r.closeupUrl}" alt="Contrast-boosted close-up of the result window">
            <div class="marks"><span style="left:${(100 * r.test_pos).toFixed(2)}%">test line</span><span style="left:${(100 * r.control_pos).toFixed(2)}%">control</span></div>
            <figcaption>Close-up of the result window, contrast boosted ×3 (the same in every row)</figcaption></figure>
        </div></article>`;
    }
    html += '</section>';
  }
  if (rows.length) {
    html += `<section class="brand"><h3>All scores</h3><div class="table-wrap"><table>
      <thead><tr><th>Test</th><th>Taken</th><th>Photo</th><th class="num">Score</th><th class="num">Test ÷ control</th><th class="num">± within photo</th></tr></thead><tbody>
      ${rows.slice().sort((a, b) => BRANDS.indexOf(a.brand) - BRANDS.indexOf(b.brand)).map((r) => `<tr><td>${esc(BRAND_NAME[r.brand])}</td><td>${esc(whenLabel(r.dt, r.hasTime))}</td>
        <td>${esc(r.name)}</td><td class="num"><strong>${r.score}</strong></td><td class="num">${r.measurable ? `${r.pct.toFixed(1)}%` : 'none'}</td><td class="num">${r.se.toFixed(1)}</td></tr>`).join('')}
      </tbody></table></div></section>`;
  }
  html += `<section class="brand"><details class="located-wrap" ${none.length ? 'open' : ''}><summary><strong>What was located in each photo</strong></summary>
    <div class="located">${done.filter((r) => r.overlayUrl).map((r) => `<figure><img src="${r.overlayUrl}" alt="Photo with located tests marked" loading="lazy">
      <figcaption>${esc(r.name)}: ${esc(r.notes.join('; '))}</figcaption></figure>`).join('')}</div></details></section>
    <section class="about">
      <h3>How the score is worked out</h3>
      <ul>
        <li>Each test is found in the photo and straightened. The green light absorbed by the test line and by the control line is measured, with the blank strip beside each line as the baseline.</li>
        <li>Score = test line ÷ control line × 100. A score of 100 means the test line is as dark as the control line. Using each test's own control line and blank strip removes most differences in lighting and camera settings.</li>
        <li>“± within photo” is the uncertainty from one photo alone. Two photos of the same test usually differ by a few points.</li>
      </ul>
      <h3>What it cannot tell you</h3>
      <ul>
        <li>This is not a medical device and not a measurement of hCG. Line darkness also depends on how dilute the urine was, the time of day, the individual test, and how long it had been drying when photographed.</li>
        <li>Compare scores only within one brand. Scores below about 5 are at the edge of what a phone photo can show.</li>
        <li>A score cannot confirm or rule out a pregnancy or say how one is progressing. If you are worried or need a real number, a clinician can order a blood hCG test.</li>
      </ul>
    </section>`;
  const el = $('#report');
  el.innerHTML = html;
  el.hidden = false;
  const on = (id, fn) => { const b = document.getElementById(id); if (b) b.addEventListener('click', fn); };
  on('save-html', saveHtml);
  on('save-pdf', () => window.print());
  on('save-csv', (e) => { e.preventDefault(); download(new Blob([fin.csv], { type: 'text/csv' }), 'scores.csv'); });
  wireTips(el);
  el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// One series per chart (the heading names it), shared y scale across brands, a point per photo.
function chart(pts, ymax, brand) {
  if (pts.length < 2) return '';
  const W = 720, H = 230, L = 34, R = 26, T = 22, B = 30;
  const ms = pts.map((p) => dtMillis(p.dt));
  let x0 = Math.min(...ms), x1 = Math.max(...ms);
  if (x1 - x0 < 3600e3) { x0 -= 1800e3; x1 += 1800e3; }
  const pad = (x1 - x0) * 0.04;
  x0 -= pad; x1 += pad;
  const X = (m) => L + ((m - x0) / (x1 - x0)) * (W - L - R);
  const Y = (v) => T + (1 - v / ymax) * (H - T - B);
  let s = `<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(BRAND_NAME[brand])} scores over time; the same numbers are in the All scores table">`;
  for (let i = 0; i <= 4; i++) {
    const v = (ymax * i) / 4;
    s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`;
  }
  const days = [];
  const d = new Date(x0); d.setHours(0, 0, 0, 0);
  for (let t = d.getTime(); t <= x1; t = new Date(t).setDate(new Date(t).getDate() + 1)) if (t >= x0) days.push(t);
  const every = Math.ceil(days.length / 8) || 1;
  days.forEach((t, i) => { if (i % every === 0) s += `<line class="grid" x1="${X(t)}" x2="${X(t)}" y1="${H - B}" y2="${H - B + 5}"/><text x="${X(t)}" y="${H - 8}" text-anchor="middle">${esc(shortDay(t))}</text>`; });
  if (!days.length) s += `<text x="${(L + W - R) / 2}" y="${H - 8}" text-anchor="middle">${esc(shortDay(ms[0]))}</text>`;
  s += `<path class="line" d="${pts.map((p, i) => `${i ? 'L' : 'M'}${X(ms[i]).toFixed(1)},${Y(p.score).toFixed(1)}`).join(' ')}"/>`;
  const last = pts.length - 1;
  const top = pts.reduce((b, p, i) => (p.score > pts[b].score ? i : b), 0);
  pts.forEach((p, i) => {
    const cx = X(ms[i]).toFixed(1), cy = Y(p.score).toFixed(1);
    s += `<circle class="dot" cx="${cx}" cy="${cy}" r="5"/>`;
    if (i === last || (i === top && Math.abs(X(ms[i]) - X(ms[last])) > 28)) s += `<text class="val" x="${cx}" y="${Y(p.score) - 10}" text-anchor="middle">${p.score}</text>`;
  });
  pts.forEach((p, i) => {
    const tip = `${whenLabel(p.dt, p.hasTime)} · score ${p.score} (${p.pct.toFixed(1)}% of control) · ${p.name}`;
    s += `<circle class="hit" cx="${X(ms[i]).toFixed(1)}" cy="${Y(p.score).toFixed(1)}" r="14" data-tip="${esc(tip)}"/>`;
  });
  return `${s}</svg><p class="chart-cap">Score, 0 to 100 (test line as a percentage of the control line). Each point is one photo.</p></div>`;
}

function wireTips(root) {
  const tip = $('#tip');
  root.querySelectorAll('[data-tip]').forEach((n) => {
    n.addEventListener('pointerenter', () => { tip.textContent = n.dataset.tip; tip.hidden = false; });
    n.addEventListener('pointermove', (e) => {
      const w = tip.offsetWidth;
      tip.style.left = `${Math.max(8, Math.min(window.innerWidth - w - 8, e.clientX - w / 2))}px`;
      tip.style.top = `${e.clientY - tip.offsetHeight - 14 < 8 ? e.clientY + 18 : e.clientY - tip.offsetHeight - 14}px`;
    });
    n.addEventListener('pointerleave', () => { tip.hidden = true; });
  });
}

// ---------------------------------------------------------------- saving
function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

function dataUri({ bytes, type }) {
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return `data:${type};base64,${btoa(bin)}`;
}

/* A single self-contained file: styles and pictures are embedded, so it opens anywhere, offline. */
async function reportHtml() {
  const css = await (await fetch('style.css')).text();
  const clone = $('#report').cloneNode(true);
  clone.hidden = false;
  clone.querySelectorAll('.no-export').forEach((n) => n.remove());
  clone.querySelectorAll('img').forEach((img) => { const src = images.get(img.getAttribute('src')); if (src) img.setAttribute('src', dataUri(src)); img.removeAttribute('loading'); });
  clone.querySelectorAll('[data-tip]').forEach((n) => {
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    t.textContent = n.dataset.tip;
    n.appendChild(t);
  });
  const html = `<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">`
    + `<title>Test line report</title><style>${css}</style></head><body><main>${clone.outerHTML}</main></body></html>`;
  return html;
}

async function saveHtml() {
  download(new Blob([await reportHtml()], { type: 'text/html' }), `test-line-report-${new Date().toISOString().slice(0, 10)}.html`);
}
