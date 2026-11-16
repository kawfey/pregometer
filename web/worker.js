/* Runs the Python analysis (Pyodide) off the main thread. Photos never leave this worker's memory. */
importScripts('pyodide/pyodide.js');

const MODULES = ['common', 'measure', 'detect', 'collage', 'pipeline', 'webapi'];
let py = null, api = null, heif = null;

function progress(text) { postMessage({ progress: text }); }

async function init() {
  progress('Loading the analysis engine…');
  py = await loadPyodide({ indexURL: new URL('pyodide/', self.location.href).href });
  progress('Loading image libraries…');
  await py.loadPackage(['numpy', 'opencv-python', 'pillow'], { messageCallback: () => {} });
  progress('Starting…');
  for (const m of MODULES) {
    const r = await fetch(`py/${m}.py`, { cache: 'no-cache' });
    if (!r.ok) throw new Error(`could not load py/${m}.py`);
    py.FS.writeFile(`/home/pyodide/${m}.py`, await r.text());
  }
  api = py.pyimport('webapi');
  return true;
}

function withFile(buf, fn) {
  const path = '/tmp/photo';
  py.FS.writeFile(path, new Uint8Array(buf));
  try { return fn(path); } finally { py.FS.unlink(path); }
}

// HEIC: decoded here with libheif (loaded on first use); Python gets the raw pixels.
function isHeic(name, buf) {
  const b = new Uint8Array(buf, 0, Math.min(12, buf.byteLength));
  const brand = String.fromCharCode(...b.subarray(4, 12));
  return /^ftyp(heic|heix|hevc|hevx|heim|heis|hevm|hevs|mif1|msf1)$/.test(brand) || (/\.(heic|heif)$/i.test(name) && brand.startsWith('ftyp'));
}

async function decodeHeic(buf) {
  if (!heif) {
    importScripts('libheif/libheif-bundle.js');
    heif = self.libheif();              // the bundle exports a factory
    if (heif.ready) await heif.ready;
  }
  const images = new heif.HeifDecoder().decode(new Uint8Array(buf));
  if (!images.length) throw new Error('this HEIC file could not be decoded');
  const image = images.find((im) => im.is_primary()) || images[0];
  const width = image.get_width(), height = image.get_height();
  try {
    const data = await new Promise((resolve, reject) => image.display(
      { data: new Uint8ClampedArray(width * height * 4), width, height },
      (d) => (d ? resolve(d.data) : reject(new Error('this HEIC file could not be decoded')))));
    return { data, width, height };
  } finally {
    images.forEach((im) => im.free());
  }
}

function takeImage(key, transfer) {
  const proxy = api.image(key);
  const bytes = proxy.toJs();
  proxy.destroy();
  transfer.push(bytes.buffer);
  return bytes;
}

const handlers = {
  init,
  reset() { api.reset(); return true; },
  probe({ name, buf }) { return JSON.parse(withFile(buf, (p) => api.probe(name, p))); },
  async analyze({ name, buf, dt, dtTime, dtSource }, transfer) {
    let px = null;
    if (isHeic(name, buf)) {
      px = await decodeHeic(buf);
      py.FS.writeFile('/tmp/pixels', new Uint8Array(px.data.buffer));
      px.data = null;
    }
    let r;
    try {
      r = JSON.parse(withFile(buf, (p) => api.analyze(name, p, dt || null, !!dtTime, dtSource || null,
        px ? '/tmp/pixels' : null, px ? px.width : 0, px ? px.height : 0)));
    } finally {
      if (px) py.FS.unlink('/tmp/pixels');
    }
    r.overlay = takeImage(r.overlay, transfer);
    for (const t of r.tests) {
      t.crop = takeImage(t.crop, transfer);
      t.closeup = takeImage(t.closeup, transfer);
    }
    return r;
  },
  finish(_, transfer) {
    const r = JSON.parse(api.finish());
    for (const c of r.collages) c.image = takeImage(c.image, transfer);
    return r;
  },
};

// one request at a time, in arrival order
let queue = Promise.resolve();
onmessage = (e) => {
  const { id, type, payload } = e.data;
  queue = queue.then(async () => {
    const transfer = [];
    try {
      const result = await handlers[type](payload || {}, transfer);
      postMessage({ id, ok: true, result }, transfer);
    } catch (err) {
      const text = String((err && err.message) || err).trim().split('\n').slice(-1)[0];
      postMessage({ id, ok: false, error: text });
    }
  });
};
