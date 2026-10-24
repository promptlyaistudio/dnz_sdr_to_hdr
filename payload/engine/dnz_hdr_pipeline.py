"""SDR -> HDR conversion for a whole clip, on top of Lightricks' own pipeline.

Wraps ``python -m ltx_pipelines.hdr_ic_lora`` (present in the LTX-2 commit the kit
pins). Everything the Flame side has to do is here:

  * accepts a float32 (N,H,W,3) clip in one of four colour spaces
  * writes an EXR folder for the pipeline, padded to the 8k+1 frame count the
    pipeline demands (last frame repeated)
  * runs the official pipeline as a subprocess (it loads the 22B transformer, so it
    must not live inside the long-lived engine server)
  * reads the HDR EXRs back, drops the padding, returns scene-linear (or log /
    display) float frames

There is no Gemma and no audio VAE: the prompt arrives precomputed as the
SDR->HDR scene-emb .safetensors the LoRA was trained with.

CLI (Write File -> script -> Read File flow, no Flame node needed):

    /Users/Shared/MatteLTXNative/venv/bin/python dnz_hdr_pipeline.py \
        --input /tmp/sdr_exr --input-colorspace acescg --frame-rate 24 \
        --output /tmp/hdr_exr
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dnz_hdr_io import (  # noqa: E402
    IN_SPACES,
    OUT_SPACES,
    PIPE_IN,
    PIPE_OUT,
    linear_to_srgb,
    log,
    read_exr,
    resize,
    srgb_to_linear,
    write_exr,
)

HDR_MODULE = "ltx_pipelines.hdr_ic_lora"
EXR_INPUT_SPACES = ("srgb", "acescg", "acescct")     # 'srgb' means LINEAR Rec.709 for an EXR folder
EXR_OUTPUT_SPACES = ("srgb_linear", "acescg", "acescct")
REQUIRED_FLAGS = ("--input", "--input-colorspace", "--frame-rate", "--output-path", "--exr-colorspace",
                  "--hdr-lora", "--text-embeddings", "--transformer-path", "--video-vae-path",
                  "--no-quantization", "--seed", "--high-quality", "--no-keyframes", "--offload")


def target_frame_count(n: int) -> int:
    """Pipeline needs 8k+1 frames; round UP (the last frame is repeated)."""
    return max(1, ((max(1, n) - 1) // 8 + 1) * 8 + 1)


def _suppress_fireflies(clip: np.ndarray, strength: float = 4.0) -> np.ndarray:
    """Replace isolated hot pixels (fireflies) with their neighbourhood mean.

    A firefly is a pixel whose luma is >strength times the mean of its 3x3 neighbours
    AND above 1.0 in absolute value. The model occasionally emits these as extreme
    spikes (up to 223x the source); they look like random sparkles in the output.
    Replacing them with the local mean removes the sparkle without touching real
    highlights, which have supporting structure around them.
    """
    if clip.size == 0:
        return clip
    # Work on luma for detection, apply the fix per-channel
    luma = np.nan_to_num(clip.mean(axis=-1), nan=0.0, posinf=0.0, neginf=0.0)

    # 3x3 neighbourhood mean (excluding centre) via shift-and-add on last two dims
    h, w = luma.shape[-2:]
    pad = np.pad(luma, ((0, 0),) * (luma.ndim - 2) + ((1, 1), (1, 1)), mode="edge")
    nb_sum = np.zeros_like(luma)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            nb_sum += pad[..., dy:dy+h, dx:dx+w]
    nb_mean = nb_sum / 8.0

    # Firefly mask: pixel is much brighter than neighbours AND absolutely bright
    mask = (luma > strength * np.maximum(nb_mean, 0.05)) & (luma > 1.0)

    if not mask.any():
        return clip

    # Replace firefly pixels with the neighbourhood mean (preserving hue roughly)
    nb_ratio = np.where(mask, nb_mean / np.maximum(luma, 1e-6), 1.0)
    return clip * nb_ratio[..., np.newaxis]


def _is_real(p: Path) -> bool:
    """True only for a genuine safetensors payload. An LFS pointer stub left behind by an
    interrupted download is a few hundred bytes of text; handing one to the pipeline
    would crash it (or worse, load garbage) instead of saying the model is missing."""
    try:
        if not p.is_file():
            return False
        if p.stat().st_size < 1_000_000:               # every real LTX model is >= 12 MB
            return False
        with open(p, "rb") as f:
            head = f.read(64)
        return not head.startswith(b"version https://git-lfs")   # LFS pointer text
    except OSError:
        return False


def _find(name_needles: tuple[str, ...], dirs: list[Path], exclude: str | None = None) -> Path | None:
    """First real file under any of `dirs` whose name contains one of `name_needles`."""
    seen: set[Path] = set()
    for d in dirs:
        if not d.is_dir() or d in seen:
            continue
        seen.add(d)
        for p in sorted(d.rglob("*.safetensors")):
            n = p.name.lower()
            if any(nd in n for nd in name_needles) and not (exclude and exclude in n):
                if _is_real(p):
                    return p
    return None


def _model_dirs(root: Path) -> list[Path]:
    """Where model files may live: the kit's own models/ tree (recursively, so it finds
    models/_ltx25/... too) plus any extra roots the install knows about."""
    models = root / "models"
    dirs = [models]
    extra = os.environ.get("LTX_MODEL_DIRS", "")       # colon-separated, for relocated kits
    for e in extra.split(os.pathsep):
        if e.strip():
            dirs.append(Path(e.strip()))
    return dirs


def resolve_models(root: Path) -> dict:
    """Model files the HDR pipeline needs. The HDR LoRA + scene-emb are downloaded by the
    installer into <root>/models/hdr/; the transformer and VAEs are the kit's own (or the
    Matte kit's / a ComfyUI folder's - we take the first genuine copy we find)."""
    models = root / "models"
    hdr_dir = models / "hdr"
    dirs = _model_dirs(root)
    transformer = _find(("22b-distilled-transformer",), dirs, exclude="fp8")
    vae = _find(("2.5-video-vae", "video_vae", "video-vae"), dirs)
    lora = _find(("sdr-to-hdr",), dirs, exclude="scene-emb")
    emb = _find(("scene-emb",), dirs)

    missing: list[str] = []
    if transformer is None:
        missing.append("  transformer: no real ltx-2.5-22b-distilled-transformer-bf16.safetensors "
                       f"found under {models} (or $LTX_MODEL_DIRS)")
    if vae is None:
        missing.append(f"  video VAE: no real ltx-2.5-video-vae-bf16.safetensors under {models}")
    if lora is None:
        missing.append(f"  HDR IC-LoRA: nothing matching 'sdr-to-hdr' under {models}")
    if emb is None:
        missing.append(f"  scene-emb: nothing matching 'scene-emb' under {models}")
    if missing:
        raise FileNotFoundError(
            "Missing model files:\n" + "\n".join(missing) +
            "\n\nThe transformer/VAE are the base LTX-2.5 models (the Matte kit or ComfyUI may "
            "already have them; set LTX_MODEL_DIRS to point at them). The SDR->HDR IC-LoRA and its "
            "scene-emb are NOT part of the base kit: ask for access and download both from "
            "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR then put them in "
            f"{hdr_dir}/, or re-run the installer.")
    return {"transformer": transformer, "vae": vae, "lora": lora, "emb": emb, "root": root}


def pipeline_env(root: Path) -> dict:
    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.setdefault("PYTORCH_ENABLE_MPS_FRACTIONAL", "1")
    env["PATH"] = os.pathsep.join([env.get("PATH", ""), "/opt/homebrew/bin", "/usr/local/bin"])
    env["PYTHONUNBUFFERED"] = "1"
    return env


def preflight(py: str, root: Path, env: dict, timeout: int = 600) -> None:
    """Fail early and loudly if this LTX-2 checkout cannot run the HDR pipeline."""
    r = subprocess.run([py, "-m", HDR_MODULE, "--help"], cwd=str(root), env=env,
                       capture_output=True, text=True, timeout=timeout)
    text = r.stdout + r.stderr
    if r.returncode != 0:
        raise RuntimeError(f"{HDR_MODULE} --help failed (exit {r.returncode}):\n"
                           + "\n".join(text.strip().splitlines()[-8:]))
    missing = [f for f in REQUIRED_FLAGS if f not in text]
    if missing:
        raise RuntimeError(f"this LTX-2 version lacks flags {missing}; the kit's LTX-2 commit is too old")


def _stream(cmd: list[str], cwd: Path, env: dict, timeout: float, progress, cancel) -> None:
    """Run the pipeline, mirroring its output into the engine log and progress callback."""
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    cancel_box = {"killed": False}

    def _kill():
        cancel_box["killed"] = True
        proc.kill()

    timer = threading.Timer(timeout, _kill)
    timer.start()
    tail: deque[str] = deque(maxlen=60)
    last = 0.0
    t0 = time.time()
    loading = True                     # until the first real progress bar appears
    last_milestone = ""

    def _milestone(line: str) -> str:
        """A short status label from a pipeline log line, so the loading phase shows life."""
        for marker in ("Building transformer", "Loading text embeddings", "Loading",
                       "vae", "VAE", "offload", "Compiling", "compiling"):
            if marker in line:
                return line.strip()[-70:]
        return ""

    try:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            tail.append(line)
            prog = "%|" in line or "it/s" in line or "s/it" in line
            now = time.time()
            if prog:
                if now - last < 5:
                    continue
                last = now
                loading = False
                progress("HDR pass running", _pipe_progress(line))
            elif loading:
                # the load phase can last 10-20 min: show what it is doing and how long
                ms = _milestone(line)
                if ms and ms != last_milestone:
                    last_milestone = ms
                    progress(f"{ms}  ({int(now - t0)}s)", 0.08)
                elif not ms and int(now - t0) % 30 == 0:
                    progress(f"loading the 22B model  ({int(now - t0)}s elapsed)", 0.08)
            log("[hdr] " + line)
            if cancel is not None and cancel():
                _kill()
                break
        rc = proc.wait()
    finally:
        timer.cancel()
    if cancel_box["killed"]:
        raise RuntimeError("cancelled")
    if rc != 0:
        hint = "\nHint: if the error mentions keyframes, re-run with keyframes off (--no-keyframes)." \
            if any("keyframe" in t.lower() for t in tail) else ""
        raise RuntimeError(f"pipeline exit {rc}\n" + "\n".join(tail) + hint)


def _pipe_progress(line: str) -> float:
    """Best-effort 0..1 from a tqdm/progress line (the pass is 8 distilled steps)."""
    try:
        pct = float(line.split("%|", 1)[0].strip().rsplit(" ", 1)[-1])
        return max(0.0, min(1.0, pct / 100.0))
    except (ValueError, IndexError):
        return 0.5


def convert(frames: np.ndarray, *, in_cs: str, out_cs: str, fps: float = 24.0, seed: int = 42,
            root: Path, python: str | None = None, work_dir: Path, high_quality: bool = False,
            keyframes: bool = True, fp8: bool = False, offload: str = "none", timeout: float = 4 * 3600,
            progress=None, cancel=None, extra_args: tuple[str, ...] = (),
            tiling: dict | None = None) -> tuple[np.ndarray, Path | None]:
    """SDR clip -> HDR clip. Returns (clip in `out_cs` space, preview MP4 path or None)."""
    t0 = time.time()
    if in_cs not in IN_SPACES or out_cs not in OUT_SPACES:
        raise ValueError(f"bad colour space: in={in_cs} out={out_cs}")
    n, h, w, c = frames.shape
    if c != 3:
        raise ValueError(f"expected RGB, got {c} channels")
    if n < 1:
        raise ValueError("empty clip")

    models = resolve_models(root)
    py = python or str(root / "venv" / "bin" / "python")
    env = pipeline_env(root)
    target = target_frame_count(n)

    def prog(msg: str, frac: float) -> None:
        if progress:
            progress(msg, max(0.0, min(1.0, frac)))

    prog("checking the HDR pipeline", 0.02)
    preflight(py, root, env)

    in_exr = work_dir / "in_exr"
    in_exr.mkdir(parents=True, exist_ok=True)
    pipe_in = PIPE_IN[in_cs]
    src = srgb_to_linear(frames) if in_cs == "display_srgb" else frames
    prog(f"writing {target} EXR frames", 0.06)
    for i in range(target):                            # pad by repeating the last frame
        write_exr(in_exr / f"frame_{i:05d}.exr", src[min(i, n - 1)], pipe_in)

    out_mp4 = work_dir / "hdr_master.mp4"

    if _resident_available() and not extra_args:
        # Preferred: run in-process, so the 22B transformer stays loaded between jobs. The
        # first job pays the ~20 min load; later ones only run the denoise pass. extra_args
        # exist for the CLI, which always goes through the subprocess.
        _run_resident(models, in_exr, out_mp4, pipe_in, out_cs, fps, seed, high_quality,
                      keyframes, fp8, offload, prog, cancel, tiling)
    else:
        cmd = [py, "-m", HDR_MODULE,
               "--input", str(in_exr), "--input-colorspace", pipe_in, "--frame-rate", str(fps),
               "--output-path", str(out_mp4), "--exr-colorspace", PIPE_OUT[out_cs],
               "--hdr-lora", str(models["lora"]), "--text-embeddings", str(models["emb"]),
               "--transformer-path", str(models["transformer"]), "--video-vae-path", str(models["vae"]),
               "--seed", str(seed)]
        if not fp8:
            cmd.append("--no-quantization")            # the pipeline defaults to FP8 cast; MPS has none
        if high_quality:
            cmd.append("--high-quality")
        if not keyframes:
            cmd.append("--no-keyframes")
        if offload and offload != "none":
            cmd += ["--offload", offload]
        cmd += list(extra_args)
        log("[hdr] Command: " + " ".join(shlex.quote(a) for a in cmd))
        _run_subprocess(cmd, root, env, timeout, prog, cancel)

    # Both paths above wrote the same thing: an EXR folder next to out_mp4, in the
    # requested output colour space. Read it back and drop the 8k+1 padding here.
    exr_dirs = sorted(work_dir.glob("hdr_master_*_exr"))
    outs = sorted(exr_dirs[0].glob("frame_*.exr")) if exr_dirs else []
    if not outs:
        raise RuntimeError("pipeline finished but wrote no EXR frames")
    if len(outs) < n:
        raise RuntimeError(f"pipeline wrote {len(outs)} EXR frames, expected at least {n}")

    prog("reading the HDR EXRs back", 0.96)
    clip = np.stack([read_exr(p) for p in outs[:n]]).astype(np.float32)
    if out_cs == "display_srgb":
        clip = linear_to_srgb(clip)
    if clip.shape[1] != h or clip.shape[2] != w:       # the pipeline may snap to its own alignment
        clip = np.stack([resize(f, h, w) for f in clip])
    prog("cleaning fireflies", 0.97)
    clip = _suppress_fireflies(clip)
    prog("done", 1.0)
    log(f"[hdr] {n} frames {w}x{h} converted in {time.time() - t0:.0f}s")
    return clip, (out_mp4 if out_mp4.exists() else None)


def _run_subprocess(cmd: list[str], root: Path, env: dict, timeout: float, prog, cancel) -> None:
    # Loading the 42 GB transformer into unified memory is genuinely slow on Apple Silicon
    # (15-20 min seen on a 128 GB machine), so say that up front instead of pretending the
    # first minute is the quiet part.
    prog("loading the 22B model - reads ~42 GB into memory, typically 10-20 min on first run",
         0.08)
    _stream(cmd, root, env, timeout, prog, cancel)


class _LogTap(logging.Handler):
    """Forward the pipeline's own log lines to the status callback. The pipeline exposes
    no progress hook, so its log is the only honest signal of what phase it is in."""
    def __init__(self, emit) -> None:
        super().__init__(level=logging.INFO)
        self._emit = emit

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._emit(record.getMessage())
        except Exception:
            pass


_RESIDENT: dict = {"pipe": None, "key": None}


def _resident_available() -> bool:
    """True when the real pipeline class can be imported in this process. The fake test
    harness stubs only the CLI, so it is False there and we keep the subprocess path."""
    try:
        import ltx_pipelines.hdr_ic_lora as m          # noqa: PLC0415
        return hasattr(m, "HDRICLoraPipeline")
    except Exception:
        return False


def _resident_pipeline(models: dict, fp8: bool, offload: str):
    """The 22B transformer, loaded once and kept resident so later jobs skip the
    15-20 min load (the matte kit's model works the same way). Rebuilt only when a
    setting that changes the weights or their placement changes."""
    key = (str(models["transformer"]), str(models["vae"]), str(models["lora"]),
           str(models["emb"]), bool(fp8), str(offload))
    if _RESIDENT["pipe"] is not None and _RESIDENT["key"] == key:
        return _RESIDENT["pipe"], False

    from ltx_core.model.video_vae import AUTO_TILING                     # noqa: PLC0415
    from ltx_pipelines.hdr_ic_lora import HDRICLoraPipeline             # noqa: PLC0415
    from ltx_pipelines.utils.model_paths import ModelPaths              # noqa: PLC0415
    from ltx_pipelines.utils.types import OffloadMode                   # noqa: PLC0415

    if _RESIDENT["pipe"] is not None:            # a different placement: drop the old copy first
        import torch                             # noqa: PLC0415  (heavy; only needed on rebuild)
        _RESIDENT["pipe"] = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    quant = None                                 # fp8 off by default: Apple GPU has no FP8
    if fp8:
        from ltx_pipelines.hdr_ic_lora import _DEFAULT_QUANTIZATION     # noqa: PLC0415
        quant = _DEFAULT_QUANTIZATION

    log("[hdr] building the resident pipeline (loads the 22B transformer once)")
    pipe = HDRICLoraPipeline(
        model_paths=ModelPaths.from_split(
            transformer_path=str(models["transformer"]),
            video_vae_path=str(models["vae"]),
        ),
        hdr_lora=str(models["lora"]),
        text_embeddings_path=str(models["emb"]),
        quantization=quant,
        offload_mode=OffloadMode(offload if offload in ("none", "cpu", "disk") else "none"),
    )
    _RESIDENT["pipe"], _RESIDENT["key"] = pipe, key
    return pipe, True


def _install_cancel_hook(cancel, prog=None):
    """Wrap the per-step denoiser for two jobs the reference loop gives us for free.

    1. Cancel: euler_denoising_loop calls the denoiser once per step, and
       SimpleDenoiser.__call__ runs BEFORE the transformer forward (the GPU is idle at
       that instant), so raising there stops the loop cleanly, frees the GPU, and leaves
       the resident model loaded. Without this the resident path only checks cancel
       before/after the whole pass, so Clear could not stop a running clip.
    2. Progress: the pipeline exposes no step callback, so the status would sit frozen at
       ~60% for the whole denoise. step_index/sigmas give a real per-step percentage, so
       the status bar now moves through the pass.
    Runtime patch only -- the LTX-2 library file is never modified."""
    if cancel is None and prog is None:
        return lambda: None
    try:
        import ltx_pipelines.hdr_ic_lora as m              # noqa: PLC0415
        from ltx_pipelines.utils.denoisers import SimpleDenoiser as _Orig  # noqa: PLC0415
    except Exception:
        return lambda: None
    if getattr(_Orig, "_dnz_cancel_hook", False):
        return lambda: None                                # already patched

    # the denoise pass owns this slice of the overall 0..1 bar (before it the model loads,
    # after it the EXRs + HLG are written).
    D0, D1 = 0.55, 0.90
    state = {"last": -1}

    class _CancellableDenoiser(_Orig):
        _dnz_cancel_hook = True
        def __call__(self, *a, **kw):
            if cancel is not None and cancel():
                raise RuntimeError("cancelled")
            if prog is not None:
                try:
                    step = kw.get("step_index")
                    if step is None:
                        step = a[4] if len(a) > 4 else None       # (transformer, vid, aud, sigmas, step)
                    sigmas = kw.get("sigmas")
                    if sigmas is None and len(a) > 3:
                        sigmas = a[3]
                    total = max(1, len(sigmas) - 1)              # last sigma is the terminal one
                    if step is not None and step != state["last"]:
                        state["last"] = step
                        frac = min(1.0, step / total)
                        prog(f"HDR pass: denoise step {step + 1}/{total}",
                             D0 + (D1 - D0) * frac)
                except Exception:
                    pass                                        # progress must never break the pass
            return super().__call__(*a, **kw)

    m.SimpleDenoiser = _CancellableDenoiser                # hdr_ic_lora uses this global
    def _restore():
        m.SimpleDenoiser = _Orig
    return _restore


def _run_resident(models: dict, in_exr: Path, out_mp4: Path, pipe_in: str, out_cs: str,
                  fps: float, seed: int, high_quality: bool, keyframes: bool, fp8: bool,
                  offload: str, prog, cancel, tiling: dict | None = None) -> None:
    """In-process run: no subprocess, so the model stays loaded between jobs."""
    from ltx_core.model.video_vae import AUTO_TILING, TileSizeConfig, DimensionSizeConfig  # noqa: PLC0415
    from ltx_pipelines.utils.media_io.color_config import EXRColorSpace  # noqa: PLC0415
    from ltx_pipelines.utils.media_io.inputs import EXRVideoInput        # noqa: PLC0415
    from ltx_pipelines.utils.media_io.encode import encode_video         # noqa: PLC0415

    src_cs = {"srgb": EXRColorSpace.SRGB_LINEAR, "acescg": EXRColorSpace.ACESCG,
              "acescct": EXRColorSpace.ACESCCT}[pipe_in]
    dst_cs = {"srgb_linear": EXRColorSpace.SRGB_LINEAR, "acescg": EXRColorSpace.ACESCG,
              "acescct": EXRColorSpace.ACESCCT}[PIPE_OUT[out_cs]]

    pipe, loaded = _resident_pipeline(models, fp8, offload)
    if loaded:
        prog("22B model loaded - resident from now on, later jobs skip this", 0.55)
    else:
        prog("22B model already resident - running the HDR pass", 0.55)
    if cancel is not None and cancel():
        raise RuntimeError("cancelled")

    # Build tiling config from user parameters (0 = auto)
    if tiling:
        tiling_cfg = TileSizeConfig(
            frames=DimensionSizeConfig(tile_size=tiling.get("tile_frames", 0) or 80, overlap=tiling.get("overlap_frames", 0) or 24),
            height=DimensionSizeConfig(tile_size=tiling.get("tile_height", 0) or 768, overlap=tiling.get("overlap_height", 0) or 64),
            width=DimensionSizeConfig(tile_size=tiling.get("tile_width", 0) or 768, overlap=tiling.get("overlap_width", 0) or 64),
        )
    else:
        tiling_cfg = AUTO_TILING

    last = [time.time()]

    def tap(msg: str) -> None:
        now = time.time()
        if now - last[0] < 5:
            return
        last[0] = now
        prog(msg[:70] if msg else "HDR pass running", 0.60)

    handler = _LogTap(tap)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    import torch                                       # noqa: PLC0415 (heavy; only here)
    uninstall = _install_cancel_hook(cancel, prog)   # abort mid-denoise + live step progress
    try:
        # The reference CLI decorates its whole main() with @torch.inference_mode().
        # Without this the engine would run the pass with autograd on, and the pipeline's
        # inference tensors ("cannot be saved for backward") would blow up.
        with torch.inference_mode():
            acescct_hdr, out_fps = pipe(
                video=EXRVideoInput(dir=in_exr, color_space=src_cs, frame_rate=fps),
                seed=int(seed),
                tiling_config=tiling_cfg,
                high_quality_hdr=bool(high_quality),
                keyframe_strength=(None if not keyframes else 0.95),
            )
    finally:
        uninstall()
        root_logger.removeHandler(handler)
    if cancel is not None and cancel():
        raise RuntimeError("cancelled")

    prog("writing the HDR EXRs + HLG preview", 0.90)
    with torch.inference_mode():                      # as in the CLI, encode runs too
        encode_video(video=acescct_hdr, fps=int(round(out_fps)), audio=None,
                     output_path=str(out_mp4), video_chunks_number=1, color_space=dst_cs)


# --------------------------------------------------------------------------- CLI
def _find_frames(d: Path) -> list[Path]:
    exts = {".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted(p for p in d.iterdir() if p.suffix.lower() in exts)


def main() -> int:
    p = argparse.ArgumentParser(description="DNZ SDR -> HDR for frame sequences (Flame Write File flow)")
    p.add_argument("--input", required=True, help="Folder of SDR frames (EXR, or PNG/JPG/TIF = display sRGB)")
    p.add_argument("--output", required=True, help="Folder for the HDR EXR frames")
    p.add_argument("--input-colorspace", choices=list(IN_SPACES), default=None,
                   help="EXR input only: acescg / srgb_linear (linear Rec.709) / acescct. Required for EXR.")
    p.add_argument("--exr-colorspace", choices=list(OUT_SPACES), default="acescg",
                   help="Colour space of the HDR EXR frames (default acescg, scene-linear)")
    p.add_argument("--frame-rate", type=float, default=24.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--root", default=os.environ.get("AIFX_NATIVE_ROOT", "/Users/Shared/MatteLTXNative"))
    p.add_argument("--python", default=None)
    p.add_argument("--high-quality", action="store_true")
    p.add_argument("--no-keyframes", action="store_true")
    p.add_argument("--fp8", action="store_true")
    p.add_argument("--offload", choices=["none", "cpu", "disk"], default="none")
    p.add_argument("--max-frames", type=int, default=145)
    p.add_argument("--pipeline-args", default="", help="Extra flags passed verbatim to the pipeline")
    p.add_argument("--preview-mp4", default=None, help="Also copy the HLG master MP4 here")
    p.add_argument("--work-dir", default=None)
    p.add_argument("--keep-temp", action="store_true")
    p.add_argument("--json-output", default=None)
    args = p.parse_args()

    log.open(Path(args.root) / "run" / "dnz_hdr_cli.log")
    res: dict = {"status": "error"}
    try:
        in_dir = Path(args.input).expanduser().resolve()
        out_dir = Path(args.output).expanduser().resolve()
        if not in_dir.is_dir():
            raise ValueError(f"--input must be a frame folder: {in_dir}")
        if out_dir == in_dir:
            raise ValueError("--output must differ from --input")
        frames = _find_frames(in_dir)
        if len(frames) < 1:
            raise ValueError("no frames found (.exr .png .jpg .tif)")
        is_exr = frames[0].suffix.lower() == ".exr"
        if is_exr and not args.input_colorspace:
            raise ValueError("EXR input needs --input-colorspace (acescg, srgb_linear, acescct). "
                             "Not guessing: the wrong choice gives wrong HDR.")
        in_cs = args.input_colorspace or "display_srgb"
        log(f"[hdr] {len(frames)} frames, input {in_cs}, output {args.exr_colorspace}")

        import cv2
        arrs = []
        for f in frames:
            if f.suffix.lower() == ".exr":
                a = read_exr(f)
            else:                                      # display sRGB 8-bit -> display float
                a = cv2.imread(str(f), cv2.IMREAD_COLOR)
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            arrs.append(a)
        h, w = arrs[0].shape[:2]
        clip = np.stack([resize(a, h, w) for a in arrs])
        if len(clip) > args.max_frames:
            raise ValueError(f"{len(clip)} frames exceeds --max-frames {args.max_frames}; split the clip")

        work = Path(args.work_dir).expanduser() if args.work_dir else Path(args.root) / "run" / "hdr_cli_tmp"
        work.mkdir(parents=True, exist_ok=True)
        hdr, mp4 = convert(clip, in_cs=in_cs, out_cs=args.exr_colorspace, fps=args.frame_rate,
                           seed=args.seed, root=Path(args.root), python=args.python, work_dir=work,
                           high_quality=args.high_quality, keyframes=not args.no_keyframes,
                           fp8=args.fp8, offload=args.offload,
                           extra_args=tuple(shlex.split(args.pipeline_args)) if args.pipeline_args else ())
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            write_exr(out_dir / f"{f.stem}.exr", hdr[i], args.exr_colorspace)
        if mp4 and args.preview_mp4:
            shutil.copy2(mp4, args.preview_mp4)
        res = {"status": "success", "output_path": str(out_dir), "frames": len(frames),
               "exr_colorspace": args.exr_colorspace}
    except Exception as e:  # noqa: BLE001
        import traceback
        log("[hdr] FAILED:\n" + traceback.format_exc())
        res = {"status": "error", "error": f"{type(e).__name__}: {e}"}
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(res, indent=2))
    if res["status"] == "success":
        log(f"[hdr] Success: {res['frames']} frames -> {res['output_path']}")
        return 0
    log(f"[hdr] FAILED: {res.get('error')}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
