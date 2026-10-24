#!/usr/bin/env python3
"""
dnz_inspect.py - find the stage where a DNZ SDR->HDR job turns into noise.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

DEFAULT_ROOT = os.environ.get("AIFX_NATIVE_ROOT", "/Users/Shared/MatteLTXNative")


# ----------------------------------------------------------------------------- helpers
def srgb_encode(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def write_png(path: Path, rgb01: np.ndarray):
    a = (np.clip(rgb01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    h, w, _ = a.shape
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def read_exr(path: Path) -> np.ndarray:
    try:
        import OpenImageIO as oiio
        buf = oiio.ImageBuf(str(path))
        if buf.has_error:
            raise RuntimeError(buf.geterror())
        a = np.asarray(buf.get_pixels(oiio.FLOAT), dtype=np.float32)
    except ImportError:
        import OpenEXR
        ch = OpenEXR.File(str(path)).channels()
        key = "RGB" if "RGB" in ch else ("RGBA" if "RGBA" in ch else next(iter(ch)))
        a = np.asarray(ch[key].pixels, dtype=np.float32)
    if a.ndim == 2:
        a = a[..., None]
    if a.shape[2] == 1:
        a = np.repeat(a, 3, 2)
    return a[:, :, :3]


def luma(rgb):
    return np.nan_to_num(rgb.mean(axis=2).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def box3(y):
    p = np.pad(y, 1, mode="edge")
    return sum(p[i:i + y.shape[0], j:j + y.shape[1]] for i in range(3) for j in range(3)) / 9.0


def roughness(rgb: np.ndarray) -> float:
    y = luma(rgb)
    b = box3(y)
    return float(np.abs(y - b).mean() / max(float(np.abs(y).mean()), 1e-3))


def rough_dark(rgb):
    y = luma(rgb)
    b = box3(y)
    m = b <= 1.0
    if m.sum() < 100:
        return float("nan")
    return float(np.abs(y - b)[m].mean() / max(float(y[m].mean()), 1e-3))


def fireflies_per_mpix(rgb):
    y = luma(rgb)
    p = np.pad(y, 1, mode="edge")
    nb = (sum(p[i:i + y.shape[0], j:j + y.shape[1]] for i in range(3) for j in range(3)) - y) / 8.0
    spikes = (y > 4.0 * np.maximum(nb, 0.05)) & (y > 1.5)
    return float(spikes.sum()) / (y.size / 1e6)


def flicker(a, b):
    ya, yb = luma(a), luma(b)
    m = (ya <= 1.0) & (yb <= 1.0)
    if m.sum() < 100:
        return float("nan")
    return float(np.abs(ya - yb)[m].mean() / max(float(ya[m].mean()), 1e-3))


def tonemap(rgb):
    y = np.maximum(luma(rgb), 1e-6)[..., None]
    return srgb_encode(rgb * (y / (1.0 + y)) / y)


def _series_from_lumas(lumas):
    raw, norm, lum = [], [], []
    prev = None
    for y in lumas:
        low = y[y <= 1.0]
        lum.append(float(low.mean()) if low.size else float("nan"))
        if prev is not None:
            m = (prev <= 1.0) & (y <= 1.0)
            if m.sum() > 100:
                A, B = prev[m], y[m]
                ma = max(float(A.mean()), 1e-3)
                raw.append(float(np.abs(A - B).mean() / ma))
                g = float(A.mean()) / max(float(B.mean()), 1e-6)
                norm.append(float(np.abs(A - B * g).mean() / ma))
            else:
                raw.append(float("nan")); norm.append(float("nan"))
        prev = y
    return {"raw": raw, "norm": norm, "lum": lum}


def _clean_luma(a):
    return np.nan_to_num(a.mean(axis=2).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def flicker_series(path: Path, n, w, h, stride=2):
    mm = np.memmap(path, dtype="<f2", mode="r", shape=(n, h, w, 3))
    return _series_from_lumas(_clean_luma(mm[i, ::stride, ::stride].astype(np.float32)) for i in range(n))


def exr_series(d: Path, stride=2):
    files = sorted(d.glob("frame_*.exr"))
    return _series_from_lumas(_clean_luma(read_exr(f)[::stride, ::stride]) for f in files)


def level_tracking(rep, lum_in, lum_out):
    a, b = np.array(lum_in, dtype=np.float64), np.array(lum_out, dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 5:
        return
    a, b = a[ok], b[ok]
    gain = b / np.maximum(a, 1e-6)
    cv_in = float(a.std() / max(a.mean(), 1e-6))
    msg = (f"  level tracking: HDR/SDR level ratio per frame  min {gain.min():.2f}  median {np.median(gain):.2f}  "
           f"max {gain.max():.2f}  (spread x{gain.max()/max(gain.min(),1e-6):.2f})")
    if cv_in > 0.03:
        r = float(np.corrcoef(a, b)[0, 1])
        msg += f"  | correlation with source level r={r:.2f}"
        if r < 0.5:
            rep.problems.append("output brightness does not follow the source's brightness over time (r<0.5): HDR exposure unstable")
    else:
        msg += "  | source level is steady"
    rep(msg)
    if gain.max() / max(gain.min(), 1e-6) > 1.5:
        rep.problems.append(f"HDR/SDR level ratio varies x{gain.max()/max(gain.min(),1e-6):.2f} across the clip: tone/exposure drifts over time")


def paired_transfer(rep, in_path, out_path, n, w, h, on, ow, oh, stride=4):
    """Transfer curve input.f16 -> hdr.f16, matched PIXEL BY PIXEL, reported as a
    PEDESTAL (out - g*in) rather than a ratio.

    Why the pedestal and not `out/in`: the ratio has a pole at in = 0. A constant
    additive offset of 0.0035 reads as x36 when the source is 1e-4 and as x1.03 when
    the source is 0.1 -- the offset never changed, only the denominator did. So a
    "shadows are lifted x11" reading is mostly a statement about how dark the source
    was, not about how much the engine added. The matched difference has no such pole:
    for the same pedestal it stays +0.0035 across that whole range. That is the number
    to watch, and it is the one that moves when a fix actually lands.

    Both bins are also taken on the SAME input threshold, so a pixel counted in a bin
    is the same pixel on both sides; binning each image on its own luma compares two
    different populations and invents structure that is not in the data.
    """
    if None in (on, ow, oh) or not in_path.exists() or not out_path.exists():
        return
    if in_path.stat().st_size != n * h * w * 3 * 2 or out_path.stat().st_size != on * oh * ow * 3 * 2:
        rep("  (paired transfer: file size does not match layout, skipped)")
        return
    mi = np.memmap(in_path, dtype="<f2", mode="r", shape=(n, h, w, 3))
    mo = np.memmap(out_path, dtype="<f2", mode="r", shape=(on, oh, ow, 3))

    # a few frames spread over the clip, not just frame 0
    picks = [int(i) for i in np.linspace(0, min(n, on) - 1, 5)]
    Li, Lo = [], []
    for i in picks:
        if (mi.shape[1], mi.shape[2]) != (mo.shape[1], mo.shape[2]):
            return                              # resolution mismatch: nothing to pair
        Li.append(_clean_luma(mi[i, ::stride, ::stride].astype(np.float32)).ravel())
        Lo.append(_clean_luma(mo[i, ::stride, ::stride].astype(np.float32)).ravel())
    li = np.concatenate(Li).astype(np.float64)
    lo = np.concatenate(Lo).astype(np.float64)

    # gain taken from mid-tones only, where an additive offset is negligible, so the
    # pedestal below is not just absorbing part of the gain
    gm = (li > 0.05) & (li < 0.30)
    if gm.sum() < 1000:
        rep("  (paired transfer: not enough mid-tone pixels to estimate gain)")
        return
    g = float(np.median(lo[gm] / li[gm]))

    rep(f"  paired transfer over frames {picks}  ({li.size} px, subsampled x{stride})")
    rep(f"  mid-tone gain (0.05<=luma<0.30): x{g:.3f}")
    rep(f"  {'input luma':>18} {'n':>9} {'in':>9} {'out':>9} {'pedestal':>10} {'ratio':>8}")
    edges = [0.0, 5e-4, 2e-3, 1e-2, 5e-2, 1.5e-1, 4e-1, 1e9]
    for a, b in zip(edges[:-1], edges[1:]):
        m = (li >= a) & (li < b)
        if m.sum() < 200:
            continue
        ped = float(np.median(lo[m] - g * li[m]))
        ratio = float(np.median(lo[m] / np.maximum(li[m], 1e-9)))
        top = f"[{a:.0e},{b:.0e})" if b < 1e8 else f"[{a:.0e},inf)"
        rep(f"  {top:>18} {int(m.sum()):>9} {float(li[m].mean()):>9.5f} {float(lo[m].mean()):>9.5f} "
            f"{ped:>+10.6f} {ratio:>7.2f}x")
    rep("  ^ read the pedestal column only BELOW ~0.4. Above it the transfer stops being linear (out>1,")
    rep("    highlights are expanded on purpose) so 'pedestal' there is just that extra gain, not an offset.")
    rep("  ^ read the ratio column nowhere: it has a pole at in=0. The pedestal column is the same")
    rep("    additive error measured against a denominator that cannot collapse.")

    # true black: what the engine puts where the source has nothing.
    # The threshold is loose on purpose -- subsampling thins the darkest pixels, and a
    # hard 1e-4 / 100-px gate silently skipped the single most important number in the
    # report. If there is genuinely no black in the clip, say so rather than say nothing.
    mb = li < 5e-4
    if mb.sum() >= 30:
        med = float(np.median(lo[mb]))
        rep(f"  source black (luma<5e-4), n={int(mb.sum())}: out luma median {med:.6f} "
            f"p90 {float(np.percentile(lo[mb], 90)):.6f}")
        rep("  ^ this is the additive floor. ~0.003-0.004 linear = 12-15/255 after sRGB: blacks read milky.")
        if med > 0.001:
            rep.problems.append(f"engine output has an additive black pedestal of ~{med:.4f} linear; "
                                "blacks will not reach 0 in the render. Not a colour-space mismatch "
                                "(an ACEScct misread would fit the curve at >90% error, see --acescct-check).")
    else:
        rep(f"  source black (luma<5e-4): only {int(mb.sum())} px in the sample -- clip has no true black to measure.")
    del mi, mo


def spark(v):
    v = np.array(v, dtype=np.float64)
    if np.isnan(v).all():
        return ""
    lo, hi = np.nanmin(v), np.nanmax(v)
    bars = "▁▂▃▄▅▆▇█"
    return "".join(bars[int((x - lo) / (hi - lo + 1e-12) * 7)] if not np.isnan(x) else "?" for x in v)


def phase_search(ys, periods=(8, 16, 24, 32)):
    a = np.array(ys, dtype=np.float64)
    idx = np.arange(len(a))
    out = {}
    for p in periods:
        if len(a) < 2 * p:
            continue
        best = None
        for ph in range(p):
            sel = (idx % p) == ph
            if sel.sum() < 2 or (~sel).sum() < 2:
                continue
            r = float(np.nanmean(a[sel]) / max(float(np.nanmean(a[~sel])), 1e-6))
            if best is None or r > best[1]:
                best = (ph, r)
        if best:
            out[p] = best
    return out


def report_series(rep, name, d, label=None):
    raw = np.array(d["raw"], dtype=np.float64)
    rep(f"  {label or name}: flicker per frame pair  mean {np.nanmean(raw):.4f}  median {np.nanmedian(raw):.4f}  "
        f"min {np.nanmin(raw):.4f}  max {np.nanmax(raw):.4f}")
    rep(f"    raw flicker      {spark(raw)}")
    rep(f"    global-gain-free {spark(d['norm'])}")
    rep(f"    mean level       {spark(d['lum'])}")
    ph = phase_search(raw)
    if ph:
        rep("    periodic test (pair i = frame i -> i+1):  " +
            "   ".join(f"every {p}: phase {f} x{r:.2f}" for p, (f, r) in ph.items()))
    rep.metrics.setdefault("series", {})[name] = {"raw": raw, "norm": np.array(d["norm"]), "lum": d["lum"], "phase": ph}


def find_key(d, *needles, exact=()):
    for k in exact:
        if isinstance(d.get(k), (int, float)) and not isinstance(d.get(k), bool):
            return int(d[k])
    for k, v in d.items():
        kl = k.lower()
        if any(b in kl for b in ("rate", "fps", "range", "start", "end", "res", "seed")):
            continue
        if needles and all(n in kl for n in needles) and isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    for v in d.values():
        if isinstance(v, dict):
            r = find_key(v, *needles, exact=exact)
            if r is not None:
                return r
    return None


def load_json(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


class Report:
    def __init__(self):
        self.lines: list[str] = []
        self.rough: list[tuple[str, float]] = []
        self.problems: list[str] = []
        self.metrics: dict[str, dict] = {}

    def __call__(self, s=""):
        print(s)
        self.lines.append(s)


# ----------------------------------------------------------------------------- checks
def geometry_check(rep, path, n, w, h, name):
    """[1b] Squeeze-and-tile detector: is the frame really one frame?

    If the plate read advanced its pixel pointer wrongly (the fetchFrame pxStride
    bug class), the gathered frame becomes the whole image squeezed to 1/k width
    and tiled k times across - it still has sane statistics (mean, range, noise),
    so every other check passes and the mosaic ships. This exact bug turned EVERY
    job's input into a 4-up mosaic with an RGBA host (comps multiplied into
    pxStride, which already contains comps; period w/4, strip corr 0.94-0.997;
    found 2026-10-16). A strong horizontal/vertical repetition period is the
    tell. Only a deliberately tiled pattern (wallpaper, contact sheet) would
    false-positive - plates do not.
    """
    p = Path(path)
    if not p.exists() or not n or not w or not h:
        return
    a = np.fromfile(p, dtype="<f2").astype(np.float32)
    need = n * h * w * 3
    if a.size < need:
        rep(f"  {name}: geometry skipped (file smaller than expected)")
        return
    f0 = a[:need].reshape(n, h, w, 3)[0]
    lum = 0.2126 * f0[..., 0] + 0.7152 * f0[..., 1] + 0.0722 * f0[..., 2]

    def corr(a, b):
        x = a.ravel() - a.mean()
        y = b.ravel() - b.mean()
        d = float(np.sqrt((x * x).sum() * (y * y).sum()))
        return float((x * y).sum() / d) if d > 0 else 0.0

    def best_period(horizontal):
        best = (0.0, 0)
        lim = (w if horizontal else h) // 2
        if lim < 60:
            return best
        for s in range(60, lim + 1, 4):
            c = corr(lum[:, : w - s], lum[:, s:]) if horizontal else corr(lum[: h - s, :], lum[s:, :])
            if c > best[0]:
                best = (c, s)
        if best[0] > 0.5:                                  # refine around the coarse peak
            for s in range(max(2, best[1] - 5), min(lim, best[1] + 5) + 1):
                c = corr(lum[:, : w - s], lum[:, s:]) if horizontal else corr(lum[: h - s, :], lum[s:, :])
                if c > best[0]:
                    best = (c, s)
        return best

    for horizontal, label in ((True, "yatay"), (False, "dikey")):
        c, s = best_period(horizontal)
        flag = "MOZAİK ŞÜPHESİ" if c >= 0.90 else "temiz"
        rep(f"  {name} kare 0 {label} tekrar: en iyi periyot {s}px, korelasyon {c:.3f}  -> {flag}")
        if c >= 0.90:
            rep.problems.append(
                f"{name}: {label} tekrar periyodu {s}px (korelasyon {c:.3f}) - kare sikistirilip "
                f"mozaike benziyor; plate okuma (pxStride/comps) veya motor tile birleşimini ara"
            )


def explain_size(rep, name, size, n, w, h):
    cands = []
    for item, dt in ((2, "float16"), (4, "float32")):
        if size % item:
            continue
        per = size // item
        if per % n:
            continue
        per //= n
        if per % w == 0:
            rowvals = per // w
            for c in (1, 3, 4):
                if rowvals % c == 0:
                    cands.append(f"{dt}, {c} channel(s), height {rowvals // c}")
    msg = (f"{name}: size {size} bytes != expected {n*h*w*3*2} (N={n} x H={h} x W={w} x 3 x float16)")
    rep(f"  !! {msg}")
    rep("     layouts that WOULD explain this size (width and frame count taken as correct):")
    for c in cands[:6] or ["none (width or frame count is also wrong)"]:
        rep(f"       - {c}")
    rep.problems.append(f"{name} has the wrong size/layout -> reader and writer disagree")


def frame_stats(rep, label, rgb, out_dir, tag, stage_key):
    nan = int(np.isnan(rgb).sum())
    inf = int(np.isinf(rgb).sum())
    fin = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    r = roughness(fin)
    rd = rough_dark(fin)
    ff = fireflies_per_mpix(fin)
    n1, n10, n100 = (float((fin.mean(axis=2) > t).mean()) * 100 for t in (1, 10, 100))
    rep(f"  {label}: {rgb.shape[1]}x{rgb.shape[0]}  min {fin.min():.4f}  max {fin.max():.1f}  mean {fin.mean():.4f}  NaN {nan}  Inf {inf}")
    rep(f"      roughness all {r:.3f} | dark/mid only {rd:.3f} | fireflies {ff:.0f}/Mpix | "
        f"pixels >1: {n1:.2f}%  >10: {n10:.2f}%  >100: {n100:.3f}%")
    if nan or inf:
        rep.problems.append(f"{stage_key}: contains NaN/Inf ({nan}/{inf})")
    if fin.max() <= 1e-6:
        rep.problems.append(f"{stage_key}: frame is all black")
    elif float(fin.std()) < 1e-4:
        rep.problems.append(f"{stage_key}: frame is constant")
    out_dir.mkdir(exist_ok=True)
    write_png(out_dir / f"{tag}_clip.png", srgb_encode(fin))
    write_png(out_dir / f"{tag}_tonemap.png", tonemap(fin))
    return {"rough": r, "rough_dark": rd, "fireflies": ff, "mean": float(fin.mean()), "max": float(fin.max()),
            "over1": n1, "over100": n100}


def _pick(n, job_n=None):
    base = job_n or n
    m = max(0, base // 2)
    return sorted({min(i, n - 1) for i in (0, m, m + 1)})


def _stage_summary(rep, stage, frames, stats):
    ks = sorted(stats)
    st = dict(stats[ks[0]])
    mid = [k for k in ks if k != 0] or ks
    st["rough_dark"] = float(np.nanmean([stats[k]["rough_dark"] for k in ks]))
    st["fireflies"] = float(np.mean([stats[k]["fireflies"] for k in ks]))
    pair = [(a, b) for a in mid for b in mid if b == a + 1]
    st["flicker"] = flicker(frames[pair[0][0]], frames[pair[0][1]]) if pair else float("nan")
    rep(f"  -> temporal flicker (frame {pair[0][0]} vs {pair[0][1]}): {st['flicker']:.4f}" if pair else "  -> (no consecutive frames)")
    rep.metrics[stage] = st
    rep.rough.append((stage, st["rough"]))


def check_f16(rep, path, n, w, h, out_dir, stage, job_n=None):
    if not path.exists():
        rep(f"  (missing) {path.name}")
        return
    size = path.stat().st_size
    rep(f"  {path.name}: {size} bytes")
    if None in (n, w, h):
        rep("  (job metadata has no frames/width/height; skipping size check)")
        return
    if size != n * h * w * 3 * 2:
        explain_size(rep, path.name, size, n, w, h)
        return
    frames, stats = {}, {}
    for k in _pick(n, job_n):
        a = np.fromfile(path, dtype="<f2", count=h * w * 3, offset=k * h * w * 3 * 2).astype(np.float32).reshape(h, w, 3)
        frames[k] = a
        stats[k] = frame_stats(rep, f"frame {k}", a, out_dir, f"{stage}_f{k}", stage)
    _stage_summary(rep, stage, frames, stats)


def check_exr_dir(rep, d, out_dir, stage, job_n=None):
    files = sorted(d.glob("frame_*.exr"))
    if not files:
        rep(f"  (no EXR frames in {d.name})")
        return
    rep(f"  {d.name}: {len(files)} frames")
    frames, stats = {}, {}
    for k in _pick(len(files), job_n):
        a = read_exr(files[k])
        frames[k] = a
        stats[k] = frame_stats(rep, files[k].name, a, out_dir, f"{stage}_f{k}", stage)
    _stage_summary(rep, stage, frames, stats)


def check_mp4(rep, mp4, out_dir):
    ff = shutil.which("ffmpeg") or next((p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg") if os.path.exists(p)), None)
    fp = shutil.which("ffprobe") or next((p for p in ("/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe") if os.path.exists(p)), None)
    if not mp4.exists():
        rep(f"  (missing) {mp4.name}")
        return
    if not (ff and fp):
        rep("  (ffmpeg/ffprobe not found, skipping mp4)")
        return
    try:
        s = json.loads(subprocess.run([fp, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                       "-of", "json", str(mp4)], capture_output=True, text=True).stdout)["streams"][0]
        w, h = int(s["width"]), int(s["height"])
        raw = subprocess.run([ff, "-v", "error", "-i", str(mp4), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True).stdout
        a = np.frombuffer(raw, dtype=np.uint8)[: w * h * 3].reshape(h, w, 3).astype(np.float32) / 255.0
    except Exception as e:
        rep(f"  (could not decode mp4: {e})")
        return
    st = frame_stats(rep, "first frame (HLG-encoded, so brighter than linear)", a, out_dir, "mp4_f0", "hdr_master.mp4")
    rep.rough.append(("hdr_master.mp4", st["rough"]))


def verdict(rep):
    rep()
    rep("=== SUMMARY ===")
    rep("(flicker column = ONE sampled frame pair; it can land on a calm or a spike pair - trust the whole-clip series below)")
    order = ["input.f16", "in_exr", "hdr_master_exr", "hdr.f16"]
    rep(f"{'stage':<16}{'rough(all)':>11}{'rough(dark)':>12}{'flicker':>9}{'fireflies/Mpix':>16}{'mean':>9}{'max':>9}")
    for k in order:
        m = rep.metrics.get(k)
        if m:
            rep(f"{k:<16}{m['rough']:>11.3f}{m['rough_dark']:>12.3f}{m['flicker']:>9.4f}{m['fireflies']:>16.0f}{m['mean']:>9.3f}{m['max']:>9.1f}")
    base = rep.metrics.get("in_exr") or rep.metrics.get("input.f16")
    out = rep.metrics.get("hdr_master_exr")
    if rep.problems:
        rep("problems:")
        for p in dict.fromkeys(rep.problems):
            rep(f"  - {p}")
    if base and out:
        flags = []
        if out["rough_dark"] > 3 * max(base["rough_dark"], 0.01) and out["rough_dark"] > 0.05:
            flags.append(f"dark/mid-tone roughness x{out['rough_dark']/max(base['rough_dark'],1e-3):.1f}")
        sers = rep.metrics.get("series", {})
        if "input.f16" in sers and "hdr.f16" in sers:
            fi, fo = float(np.nanmean(sers["input.f16"]["raw"])), float(np.nanmean(sers["hdr.f16"]["raw"]))
        else:
            fi, fo = base["flicker"], out["flicker"]
        if not np.isnan(fo) and fo > 3 * max(fi, 0.005) and fo > 0.02:
            flags.append(f"temporal flicker x{fo/max(fi,1e-4):.1f} over the whole clip (grain that changes between frames)")
        if out["fireflies"] > max(5 * base["fireflies"], 30):
            flags.append(f"{out['fireflies']:.0f} isolated hot pixels per Mpix (speckle)")
        if flags:
            rep("MODEL OUTPUT SHOWS NOISE-LIKE BEHAVIOUR vs its own input: " + "; ".join(flags))
            rep("  -> input stages are clean, so this comes from the model stage: try Seam Keyframes OFF, High Quality, "
                "another seed, full-res Analyze instead of Preview; check that the input is not very dark/low-bit.")
        else:
            rep("model output is not noisier than its input by these metrics (dark/mid roughness, flicker, speckle).")
        if out["max"] > 20 or out["over100"] > 0.05:
            rep(f"NOTE: output has extreme highlights (max {out['max']:.0f}, {out['over100']:.3f}% of pixels >100) and mean x"
                f"{out['mean']/max(base['mean'],1e-6):.1f} the input. An 8-bit/clipped view shows these as blown, hue-shifted "
                f"patches and sparkling edges. Compare *_clip.png with *_tonemap.png: if tonemap looks fine, the "
                f"'noise' is display/clipping, not data.")
    ser = rep.metrics.get("series", {}).get("hdr.f16")
    if ser and ser["phase"]:
        top = max(r for _, r in ser["phase"].values())
        p, (f, r) = next((pp, v) for pp, v in sorted(ser["phase"].items()) if v[1] >= 0.6 * top)
        if r > 1.8:
            where = {8: "the VAE's 8-frame latent borders (NOT the 24/32-frame seam keyframes)",
                     16: "every 16 frames", 24: "the 24-frame seam keyframe spacing", 32: "the 32-frame seam keyframe spacing"}[p]
            rep(f"PERIODIC PATTERN: flicker is x{r:.1f} higher on pairs with i % {p} == {f} -> tied to {where}.")
            sel = (np.arange(len(ser['raw'])) % p) == f
            raw_s, norm_s = np.nanmean(ser["raw"][sel]), np.nanmean(ser["norm"][sel])
            if norm_s < 0.4 * raw_s:
                rep(f"  at those pairs removing the global brightness step cuts flicker x{raw_s/max(norm_s,1e-6):.1f}: "
                    f"mostly an EXPOSURE/BRIGHTNESS jump between groups, not grain.")
            else:
                rep("  a global brightness step explains little of it: the image content/texture itself changes (grain-like).")
        else:
            rep(f"no periodic pattern (best: every {p}, phase {f}, x{r:.2f}): temporal noise is spread evenly over the clip.")
    rep("previews: ./inspect/<stage>_f<k>_clip.png (what 8-bit shows) and _tonemap.png (how the HDR really looks).")


def compare_mode(in_dir: Path, out_dir: Path):
    rep = Report()
    outp = out_dir.parent / (out_dir.name + "_inspect")
    rep(f"input : {in_dir}\noutput: {out_dir}")
    rep("\n[IN]")
    check_exr_dir(rep, in_dir, outp, "in_exr")
    rep("\n[OUT]")
    check_exr_dir(rep, out_dir, outp, "hdr_master_exr")
    rep("\n[series over the whole clip]")
    din, dout = exr_series(in_dir), exr_series(out_dir)
    report_series(rep, "input.f16", din, "IN ")
    report_series(rep, "hdr.f16", dout, "OUT")
    level_tracking(rep, din["lum"], dout["lum"])
    verdict(rep)
    try:
        outp.mkdir(exist_ok=True)
        (outp / "report.txt").write_text("\n".join(rep.lines))
    except OSError:
        pass


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Locate the stage where a DNZ HDR job becomes noise")
    ap.add_argument("job", nargs="?", help="job folder")
    ap.add_argument("--latest", action="store_true", help="use the newest folder in <root>/cache/hdr_jobs")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--compare", nargs=2, metavar=("IN_EXR_DIR", "OUT_EXR_DIR"),
                    help="compare any output EXR folder with its input folder (e.g. resident engine vs official CLI)")
    a = ap.parse_args()
    if a.compare:
        return compare_mode(Path(a.compare[0]), Path(a.compare[1]))
    if a.latest or not a.job:
        jobs = sorted((p for p in (Path(a.root) / "cache" / "hdr_jobs").glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
        if not jobs:
            sys.exit("no job folders found")
        job = jobs[-1]
    else:
        job = Path(a.job)
    out = job / "inspect"
    rep = Report()
    rep(f"job folder: {job}")

    jj = load_json(job / "job.json")
    meta = load_json(job / "hdr_meta.json")
    n = find_key(jj, "frame", exact=("frames", "num_frames", "n_frames", "frame_count", "nframes")) or find_key(jj, "count")
    w = find_key(jj, "width", exact=("width", "w", "work_width"))
    h = find_key(jj, "height", exact=("height", "h", "work_height"))
    rep("job.json: " + json.dumps({k: v for k, v in jj.items() if not isinstance(v, (dict, list))}, sort_keys=True)[:600])
    if meta:
        rep("hdr_meta.json: " + json.dumps(meta, sort_keys=True)[:400])
    rep(f"layout read from job.json -> frames={n} width={w} height={h}")
    mn, mw, mh = (find_key(meta, "frame", exact=("frames", "num_frames")), find_key(meta, "width", exact=("width", "w")), find_key(meta, "height", exact=("height", "h"))) if meta else (None, None, None)

    rep()
    rep("[1] input.f16  (plugin -> engine; display->linear already applied)")
    check_f16(rep, job / "input.f16", n, w, h, out, "input.f16", n)
    rep()
    rep("[1b] geometry: kare gercekten tek kare mi? (sikistirma/mozaik dedektoru)")
    for nm, fp, nn, ww, hh in (("input.f16", job / "input.f16", n, w, h),
                               ("hdr.f16", job / "hdr.f16", (meta and (find_key(meta, "frames", exact=("frames", "num_frames")) or n)) or n,
                                (meta and (find_key(meta, "width", exact=("width", "w")) or w)) or w,
                                (meta and (find_key(meta, "height", exact=("height", "h")) or h)) or h)):
        try:
            geometry_check(rep, fp, nn, ww, hh, nm)
        except Exception as e:
            rep(f"  ({nm}: geometry check failed: {e})")
    rep()
    rep("[2] in_exr/  (engine -> pipeline)")
    work = job / "work"
    if (work / "in_exr").is_dir():
        check_exr_dir(rep, work / "in_exr", out, "in_exr", n)
    elif (job / "in_exr").is_dir():
        check_exr_dir(rep, job / "in_exr", out, "in_exr", n)
    else:
        rep("  (no in_exr folder found)")
    rep()
    rep("[3] hdr_master_*_exr/  (MODEL OUTPUT)")
    dirs = sorted(work.glob("hdr_master_*_exr")) or sorted(job.glob("hdr_master_*_exr"))
    if dirs:
        check_exr_dir(rep, dirs[0], out, "hdr_master_exr", n)
    else:
        rep("  (missing)")
    rep()
    rep("[4] hdr.f16  (engine -> plugin render)")
    check_f16(rep, job / "hdr.f16", mn or n, mw or w, mh or h, out, "hdr.f16", n)
    rep()
    rep("[4b] flicker over the whole clip (is the noise uniform, or concentrated at segment seams?)")
    for nm, fp, nn, ww, hh in (("input.f16", job / "input.f16", n, w, h), ("hdr.f16", job / "hdr.f16", mn or n, mw or w, mh or h)):
        try:
            if fp.exists() and None not in (nn, ww, hh) and fp.stat().st_size == nn * hh * ww * 3 * 2 and nn >= 3:
                report_series(rep, nm, flicker_series(fp, nn, ww, hh))
        except Exception as e:
            rep(f"  ({nm}: series failed: {e})")
    sers = rep.metrics.get("series", {})
    if "input.f16" in sers and "hdr.f16" in sers:
        level_tracking(rep, sers["input.f16"]["lum"], sers["hdr.f16"]["lum"])
    rep()
    rep("[4c] paired transfer  input.f16 -> hdr.f16  (PEDESTAL, not ratio)")
    try:
        paired_transfer(rep, job / "input.f16", job / "hdr.f16", n, w, h, mn or n, mw or w, mh or h)
    except Exception as e:
        rep(f"  (paired transfer failed: {e})")
    rep()
    rep("[5] hdr_master.mp4")
    mp4 = work / "hdr_master.mp4" if (work / "hdr_master.mp4").exists() else job / "hdr_master.mp4"
    check_mp4(rep, mp4, out)
    verdict(rep)
    try:
        (out / "report.txt").write_text("\n".join(rep.lines))
    except OSError:
        pass


if __name__ == "__main__":
    main()