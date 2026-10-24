"""Shared I/O + colour helpers for the DNZ SDR->HDR engine.

Deliberately light: numpy + OpenCV + OpenImageIO only (no torch), so the engine
server stays a few hundred MB resident and the 22B model is only ever loaded by
the pipeline subprocess.

Pixel buffers are float32 HWC RGB everywhere. Job <-> plugin transfer uses
little-endian float16 raw files (N,H,W,3), the same convention the Matte engine
uses, so the OFX side stays a plain memcpy.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402

try:
    import OpenImageIO  # noqa: E402
except ImportError:                                    # engine still works; EXR via cv2
    OpenImageIO = None


# --------------------------------------------------------------------------- logging
class _Log:
    """Tiny line logger: file + stdout, same shape the Matte engine uses."""

    def __init__(self) -> None:
        self.path: Path | None = None

    def open(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if self.path:
            try:
                with open(self.path, "a") as f:
                    f.write(line + "\n")
            except OSError:
                pass


log = _Log()


# --------------------------------------------------------------------------- raw f16
def read_f16(path: str | Path, n: int, h: int, w: int, c: int = 3) -> np.ndarray:
    a = np.fromfile(path, dtype="<f2")
    need = n * h * w * c
    if a.size != need:
        raise ValueError(f"{path}: expected {need} float16 values, found {a.size}")
    return a.reshape(n, h, w, c).astype(np.float32)


def write_f16(path: str | Path, arr: np.ndarray) -> None:
    p = Path(path)
    tmp = str(p) + ".tmp"
    arr.astype("<f2").tofile(tmp)
    os.replace(tmp, p)


# --------------------------------------------------------------------------- EXR
def read_exr(path: str | Path) -> np.ndarray:
    """Read an EXR as float32 HWC RGB (3 channels, no alpha)."""
    if OpenImageIO is not None:
        buf = OpenImageIO.ImageBuf(str(path))
        if buf.has_error:                              # a property, not a method
            raise ValueError(f"OpenImageIO could not read {path}: {buf.geterror()}")
        px = buf.get_pixels(OpenImageIO.FLOAT)
        a = np.asarray(px, dtype=np.float32)
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        return a[..., :3].copy()
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if a is None:
        raise ValueError(f"could not read {path}")
    a = a.astype(np.float32)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    return a[..., :3]


def write_exr(path: str | Path, arr: np.ndarray, color_space: str | None = None) -> None:
    """Write float32 HWC RGB as a half-float EXR (what the pipeline reads).
    If color_space is provided, write it as metadata (OpenImageIO only)."""
    p = Path(path)
    tmp = str(p) + ".tmp.exr"
    a = np.ascontiguousarray(arr, dtype=np.float32)
    if OpenImageIO is not None:
        spec = OpenImageIO.ImageSpec(a.shape[1], a.shape[0], 3, OpenImageIO.HALF)
        if color_space:
            spec.attribute("oiio:ColorSpace", color_space)
        out = OpenImageIO.ImageOutput.create(tmp)
        if out is None:
            raise ValueError(f"no OpenImageIO EXR writer for {tmp}")
        if not out.open(tmp, spec):
            raise ValueError(f"could not open {tmp} for writing: {out.geterror()}")
        out.write_image(a)
        out.close()
    else:
        ok, buf = cv2.imencode(".exr", a)
        if not ok:
            raise ValueError(f"OpenCV could not encode {tmp}")
        buf.tofile(tmp)
    os.replace(tmp, p)


# --------------------------------------------------------------------------- colour
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Display-referred sRGB (gamma) -> linear Rec.709."""
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Linear Rec.709 -> display-referred sRGB (gamma)."""
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


IN_SPACES = ("acescg", "srgb_linear", "acescct", "display_srgb")
OUT_SPACES = ("acescg", "srgb_linear", "acescct", "display_srgb")
# what we hand the pipeline / ask it to write back
PIPE_IN = {"acescg": "acescg", "srgb_linear": "srgb", "acescct": "acescct", "display_srgb": "srgb"}
PIPE_OUT = {"acescg": "acescg", "srgb_linear": "srgb_linear", "acescct": "acescct",
            "display_srgb": "srgb_linear"}


# --------------------------------------------------------------------------- resize
def resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    if img.shape[0] == h and img.shape[1] == w:
        return img
    interp = cv2.INTER_AREA if (h < img.shape[0] or w < img.shape[1]) else cv2.INTER_LANCZOS4
    return cv2.resize(img, (w, h), interpolation=interp).astype(np.float32)


def snap32(n: int) -> int:
    """Nearest multiple of 32, minimum 32 (the pipeline's alignment divisor)."""
    return max(32, int(round(n / 32.0)) * 32)


def work_size(w: int, h: int, long_side: int) -> tuple[int, int]:
    """Work resolution for a clip: long side capped, both sides snapped to /32."""
    if long_side <= 0 or (w <= long_side and h <= long_side):
        return snap32(w), snap32(h)
    s = long_side / float(max(w, h))
    return snap32(max(32, int(round(w * s)))), snap32(max(32, int(round(h * s))))
