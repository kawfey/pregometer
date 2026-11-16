"""Image loading, colour conversion and geometric rectification shared by all steps."""
import datetime
import io
import os
import re

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from PIL import ImageCms   # needs littleCMS, which some builds of Pillow lack (e.g. in the browser)
    ImageCms.createProfile('sRGB')
except Exception:
    ImageCms = None

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:  # HEIC support is optional; JPEG/PNG still work
    pillow_heif = None

PHOTO_EXT = ('.heic', '.heif', '.jpg', '.jpeg', '.png')


class LocateError(Exception):
    """A test could not be located / refined at the given position."""


def list_photos(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(PHOTO_EXT) and os.path.isfile(os.path.join(folder, f)))


_FNAME_DATE = re.compile(r'(?<!\d)(20\d\d)[-_.]?([01]\d)[-_.]?([0-3]\d)(?:[ _T-]?([0-2]\d)[-_.:h]?([0-5]\d)(?:[-_.:]?([0-5]\d))?)?(?!\d)')


def date_from_filename(name):
    """('YYYY:MM:DD HH:MM:SS', has_time) from names like 2026-11-16_0730.jpg or IMG_20261116_073000.jpg."""
    m = _FNAME_DATE.search(name)
    if not m:
        return None, False
    y, mo, d, h, mi, s = m.groups()
    try:
        t = datetime.datetime(int(y), int(mo), int(d), int(h or 0), int(mi or 0), int(s or 0))
    except ValueError:
        return None, False
    return t.strftime('%Y:%m:%d %H:%M:%S'), h is not None


def _rgb_to_xyz(xr, yr, xg, yg, xb, yb, xw=0.3127, yw=0.3290):
    """RGB -> XYZ matrix for the given primaries and a D65 white point."""
    P = np.array([[x / y, 1.0, (1 - x - y) / y] for x, y in ((xr, yr), (xg, yg), (xb, yb))]).T
    return P * np.linalg.solve(P, np.array([xw / yw, 1.0, (1 - xw - yw) / yw]))


_P3_TO_SRGB = (np.linalg.inv(_rgb_to_xyz(0.64, 0.33, 0.30, 0.60, 0.15, 0.06)) @
               _rgb_to_xyz(0.680, 0.320, 0.265, 0.690, 0.150, 0.060)).astype(np.float32)
_TO_LIN = np.array([(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4) for v in np.arange(256) / 255.0], np.float32)
_Q = 1 << 14
_x = np.arange(_Q + 1) / _Q
_TO_SRGB8 = np.round(255 * np.where(_x <= 0.0031308, _x * 12.92, 1.055 * _x ** (1 / 2.4) - 0.055)).astype(np.uint8)


def is_display_p3(icc):
    """True if the ICC profile is Apple's Display P3 (what iPhones tag their photos with)."""
    return b'Display P3' in icc or 'Display P3'.encode('utf-16-be') in icc


def p3_to_srgb(a):
    """Convert a Display P3 uint8 image to sRGB (same transfer curve, different primaries)."""
    out = np.empty_like(a)
    for y in range(0, a.shape[0], 256):   # in bands, to keep peak memory low
        lin = _TO_LIN[a[y:y + 256]] @ _P3_TO_SRGB.T
        out[y:y + 256] = _TO_SRGB8[np.clip(lin * _Q + 0.5, 0, _Q).astype(np.int32)]
    return out


def _open(src, name):
    if isinstance(src, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(src)), name or ''
    return Image.open(src), name or os.path.basename(src)


def _read_bytes(src):
    if isinstance(src, (bytes, bytearray, memoryview)):
        return bytes(src)
    with open(src, 'rb') as f:
        return f.read()


def _embedded_exif(data):
    """EXIF found by its signature anywhere in a file. For containers Pillow cannot open here (HEIC
    in the browser, where the picture is decoded separately)."""
    ex = Image.Exif()
    for sig in (b'Exif\x00\x00MM\x00*', b'Exif\x00\x00II*\x00'):
        i = data.find(sig)
        if i >= 0:
            try:
                ex.load(data[i:])
            except Exception:
                ex = Image.Exif()
            break
    return ex


def _date_meta(ex, name):
    try:
        ifd = ex.get_ifd(0x8769)
    except Exception:
        ifd = {}
    meta = dict(dt=ifd.get(0x9003) or ifd.get(0x9004) or ex.get(0x0132) or None, model=ex.get(0x0110),
                dt_source='exif', dt_time=True)
    if not isinstance(meta['dt'], str) or not re.fullmatch(r'\d{4}:\d\d:\d\d \d\d:\d\d:\d\d', meta['dt'].strip()):
        meta['dt'], meta['dt_time'] = date_from_filename(name)
        meta['dt_source'] = 'filename' if meta['dt'] else None
    else:
        meta['dt'] = meta['dt'].strip()
    return meta


def photo_date(src, name=None):
    """Only the date part of load(): reads the file header, does not decode the picture."""
    try:
        im, name = _open(src, name)
        return _date_meta(im.getexif(), name)
    except UnidentifiedImageError:
        name = name or ('' if isinstance(src, (bytes, bytearray, memoryview)) else os.path.basename(src))
        return _date_meta(_embedded_exif(_read_bytes(src)), name)


def _to_srgb_array(a, p3, max_side):
    if p3:
        a = p3_to_srgb(a)
    if max_side and max(a.shape[:2]) > max_side:
        s = max_side / max(a.shape[:2])
        a = cv2.resize(a, (int(round(a.shape[1] * s)), int(round(a.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return a


def load_decoded(rgb, src, name=None, max_side=None):
    """Like load(), for a picture that was decoded elsewhere (HEIC in the browser).

    rgb -- the decoded, upright picture as a uint8 array HxWx3, in the file's own colour space
    src -- the original file (path or bytes); its date and colour profile are read from it
    """
    data = _read_bytes(src)
    name = name or ('' if isinstance(src, (bytes, bytearray, memoryview)) else os.path.basename(src))
    return _to_srgb_array(rgb, is_display_p3(data), max_side), _date_meta(_embedded_exif(data), name)


def load(src, name=None, max_side=None):
    """Return (sRGB uint8 array HxWx3, meta dict).

    src      -- a file path, or the file's bytes (then pass the original filename as `name`)
    max_side -- if given, larger photos are reduced so their long side is at most this many px

    EXIF orientation is applied and wide-gamut (Display P3) photos are converted to sRGB.
    meta['dt'] is the capture time 'YYYY:MM:DD HH:MM:SS' from EXIF. Without EXIF it is read from a
    date in the filename if there is one (meta['dt_source'] = 'filename', meta['dt_time'] tells
    whether a time of day was given), else None.
    """
    im, name = _open(src, name)
    meta = _date_meta(im.getexif(), name)
    icc = im.info.get('icc_profile')
    im = ImageOps.exif_transpose(im).convert('RGB')
    if icc and not is_display_p3(icc) and ImageCms is not None:
        try:
            prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            im = ImageCms.profileToProfile(im, prof, ImageCms.createProfile('sRGB'), outputMode='RGB')
        except Exception:
            pass  # unreadable profile: treat pixels as sRGB
    return _to_srgb_array(np.asarray(im), bool(icc) and is_display_p3(icc), max_side), meta


def srgb2lin(a):
    if a.dtype == np.uint8:
        return _TO_LIN[a]          # table lookup: far faster than a power per pixel
    a = a.astype(np.float32) / 255.0
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def lin2srgb(a):
    a = np.clip(a, 0, 1)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055)


def luminance(lin):
    return lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def pinkness(lin):
    """log(R/G) in linear light: zero for neutral, large for the pink/red dye of a test line."""
    return np.log((lin[..., 0] + 0.01) / (lin[..., 1] + 0.01))


def rectify(img, origin, axis_pt, unit_px, u_range, v_range, ppu):
    """Resample img into a canonical (u, v) frame.

    origin  -- full-res (x, y) that becomes (u, v) = (0, 0)
    axis_pt -- any point in the +u direction from origin
    unit_px -- full-res pixels per canonical unit
    ppu     -- output pixels per canonical unit
    Returns (image, M) where M maps output pixel (i, j, 1) to full-res (x, y).
    """
    o = np.array(origin, np.float64)
    d = np.array(axis_pt, np.float64) - o
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])  # +v
    W = int(round((u_range[1] - u_range[0]) * ppu))
    H = int(round((v_range[1] - v_range[0]) * ppu))
    s = unit_px / ppu
    M = np.array([[d[0] * s, n[0] * s, o[0] + u_range[0] * unit_px * d[0] + v_range[0] * unit_px * n[0]],
                  [d[1] * s, n[1] * s, o[1] + u_range[0] * unit_px * d[1] + v_range[0] * unit_px * n[1]]])
    interp = cv2.INTER_AREA if s > 1 else cv2.INTER_CUBIC
    out = cv2.warpAffine(img, M, (W, H), flags=interp | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return out, M


def to_full(M, i, j):
    """Output pixel of a rectified image -> full-res (x, y)."""
    return M @ np.array([i, j, 1.0])
