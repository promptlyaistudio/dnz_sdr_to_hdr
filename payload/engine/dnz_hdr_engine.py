"""DNZ SDR->HDR engine server for Autodesk Flame.

Same shape as the Matte engine, so the OFX side is a copy of what already works:
launched on demand by the DNZSDRtoHDR plugin, talks over a Unix socket with one
JSON object per line, and never pushes pixels over the socket. Each job is a
private folder holding

    job.json     parameters (see JOB FIELDS below)
    input.f16    N*H*W*3 float16, in the node's INPUT colour space, at work resolution
    hdr.f16      N*H*W*3 float16, in the node's OUTPUT colour space (written when done)
                 values above 1.0 are expected: this is HDR
    hdr.json     signature of the conversion (hash of the input pixels + every setting)
    status.json  state/progress/message, refreshed while the job runs

Commands: ping | convert {job} | cancel {job} | status {job} | shutdown

The 22B transformer is NOT kept resident: each job runs the official pipeline as a
subprocess and exits, so this server stays a few hundred MB and can sit next to
the Matte engine. Expect ~1 minute of model load on the first job.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import socket
import threading
import time
import traceback
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import sys  # noqa: E402

sys.path.insert(0, HERE)
from dnz_hdr_io import IN_SPACES, OUT_SPACES, log, read_f16, write_f16  # noqa: E402

ROOT = os.environ.get("AIFX_NATIVE_ROOT", os.path.dirname(HERE))
SOCK = os.environ.get("DNZ_HDR_SOCK", os.path.join(ROOT, "run", "dnz_hdr.sock"))
LOG = os.path.join(ROOT, "run", "dnz_hdr_engine.log")
IDLE_EXIT_S = int(os.environ.get("DNZ_HDR_IDLE_EXIT", str(3 * 3600)))
SETTINGS = os.path.join(ROOT, "hdr_settings.json")
VERSION = 1
HDR_SIG_VERSION = 1                                   # bump when the conversion itself changes

os.makedirs(os.path.dirname(SOCK), exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                    handlers=[logging.FileHandler(LOG), logging.StreamHandler()])
logf = logging.getLogger("dnz.hdr.server")

_jobs: "queue.Queue[str]" = queue.Queue()
_current: dict = {"job": None}
_cancel: set[str] = set()
_pending: set[str] = set()                          # queued but not finished yet (incl. warmup)
_last_activity = time.time()


def settings() -> dict:
    try:
        with open(SETTINGS) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _write_status(job: str, state: str, progress: float = 0.0, message: str = "", **extra) -> None:
    st = {"state": state, "progress": round(float(progress), 4), "message": message, "time": time.time(), **extra}
    os.makedirs(job, exist_ok=True)          # a cancelled/warmup job may write before its folder exists
    tmp = os.path.join(job, "status.json.tmp")
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, os.path.join(job, "status.json"))


def _signature(job: str, p: dict) -> str:
    """Everything that changes the HDR result: the exact input pixels and every setting."""
    h = hashlib.blake2b(digest_size=20)
    with open(os.path.join(job, "input.f16"), "rb") as f:
        for blk in iter(lambda: f.read(8 << 20), b""):
            h.update(blk)
    keys = ("frames", "width", "height", "fps", "in_cs", "out_cs", "seed", "high_quality",
            "keyframes", "fp8", "offload", "pipeline_args")
    h.update(json.dumps({k: p.get(k) for k in keys} | {"v": HDR_SIG_VERSION}, sort_keys=True).encode())
    return h.hexdigest()


def _find_cached(sig: str, job: str, nbytes: int) -> str | None:
    """Newest earlier job whose finished HDR clip has the same signature."""
    root = os.path.dirname(os.path.abspath(job))
    cands = []
    for d in os.listdir(root):
        jd = os.path.join(root, d)
        meta = os.path.join(jd, "hdr.json")
        if jd != os.path.abspath(job) and os.path.exists(meta):
            cands.append((os.path.getmtime(meta), jd))
    for _, jd in sorted(cands, reverse=True):
        try:
            with open(os.path.join(jd, "hdr.json")) as f:
                if json.load(f).get("sig") != sig:
                    continue
            fp = os.path.join(jd, "hdr.f16")
            if os.path.getsize(fp) == nbytes:
                return jd
        except (OSError, ValueError, KeyError):
            continue
    return None


def _process(job: str) -> None:
    from dnz_hdr_pipeline import convert

    t0 = time.time()
    with open(os.path.join(job, "job.json")) as f:
        p = json.load(f)

    if p.get("warmup"):                              # load the model only, no clip
        from dnz_hdr_pipeline import resolve_models, _resident_available  # noqa: PLC0415
        if not _resident_available():
            # No HDRICLoraPipeline class in this environment (the fake CLI harness): say so
            # plainly instead of letting the import explode inside the worker.
            _write_status(job, "error", 0.0,
                          "resident pipeline unavailable: ltx_pipelines.hdr_ic_lora exposes "
                          "no HDRICLoraPipeline class here")
            return
        from dnz_hdr_pipeline import _resident_pipeline  # noqa: PLC0415
        models = resolve_models(Path(ROOT))
        # Honest about what this actually does: the constructor resolves the model
        # paths and loads the text embeddings, nothing else. The 22B transformer is
        # built per job and freed after it (HDRICLoraPipeline.__call__: "Build
        # transformer -> run denoising loop -> free transformer"), so it is NOT
        # preloaded here. Measured 2026-10-15: warmup finishes in ~5 s with the
        # process at ~500 MB RSS.
        _write_status(job, "running", 0.05,
                      "resolving model paths + loading text embeddings (fast)")
        _resident_pipeline(models, bool(p.get("fp8", False)), str(p.get("offload", "cpu")))
        _write_status(job, "done", 1.0,
                      f"pipeline ready ({time.time() - t0:.0f}s): paths resolved, embeddings "
                      f"resident. The 22B transformer is NOT preloaded - it builds on the "
                      f"first job (~1 min there) and is freed after every job.",
                      seconds=round(time.time() - t0, 1))
        logf.info("warmup done in %.0fs", time.time() - t0)
        return

    n, h, w = int(p["frames"]), int(p["height"]), int(p["width"])
    in_cs, out_cs = str(p.get("in_cs", "acescg")), str(p.get("out_cs", "acescg"))
    if in_cs not in IN_SPACES or out_cs not in OUT_SPACES:
        raise ValueError(f"bad colour space in job.json: in={in_cs} out={out_cs}")
    logf.info("job %s: %d frames %dx%d  %s -> %s", os.path.basename(job), n, w, h, in_cs, out_cs)
    _write_status(job, "running", 0.01, "reading the plate")
    frames = read_f16(os.path.join(job, "input.f16"), n, h, w, 3)

    sig = _signature(job, p)
    nbytes = n * h * w * 3 * 2
    src = _find_cached(sig, job, nbytes)
    if src:                                            # identical clip + settings: reuse, no model load
        _write_status(job, "running", 0.5, f"reusing the HDR clip from {os.path.basename(src)}")
        data = np.fromfile(os.path.join(src, "hdr.f16"), dtype="<f2")
        write_f16(os.path.join(job, "hdr.f16"), data.reshape(n, h, w, 3).astype(np.float32))
        # Copy metadata file too
        src_meta = os.path.join(src, "hdr_meta.json")
        if os.path.exists(src_meta):
            import shutil
            shutil.copy2(src_meta, os.path.join(job, "hdr_meta.json"))
        _finish(job, p, t0, reused=True)
        return

    s = settings()
    work = os.path.join(job, "work")
    os.makedirs(work, exist_ok=True)

    def progress(msg: str, frac: float) -> None:
        if job in _cancel:
            raise RuntimeError("cancelled")
        _write_status(job, "running", 0.02 + 0.95 * frac, msg)

    # Build tiling config from job parameters (0 = auto)
    tile_cfg = {}
    if p.get("tile_frames", 0): tile_cfg["tile_frames"] = int(p["tile_frames"])
    if p.get("tile_height", 0): tile_cfg["tile_height"] = int(p["tile_height"])
    if p.get("tile_width", 0): tile_cfg["tile_width"] = int(p["tile_width"])
    if p.get("overlap_frames", 0): tile_cfg["overlap_frames"] = int(p["overlap_frames"])
    if p.get("overlap_height", 0): tile_cfg["overlap_height"] = int(p["overlap_height"])
    if p.get("overlap_width", 0): tile_cfg["overlap_width"] = int(p["overlap_width"])

    hdr, mp4 = convert(
        frames, in_cs=in_cs, out_cs=out_cs, fps=float(p.get("fps", 24.0)), seed=int(p.get("seed", 42)),
        root=Path(ROOT), work_dir=Path(work), high_quality=bool(p.get("high_quality", False)),
        keyframes=bool(p.get("keyframes", True)), fp8=bool(p.get("fp8", False)),
        offload=str(p.get("offload", "none")), timeout=float(p.get("timeout", s.get("timeout", 4 * 3600))),
        progress=progress, cancel=lambda: job in _cancel,
        extra_args=tuple(p.get("pipeline_args") or ()),
        tiling=tile_cfg if tile_cfg else None)
    if job in _cancel:
        raise RuntimeError("cancelled")

    _write_status(job, "running", 0.98, "writing the HDR clip")
    write_f16(os.path.join(job, "hdr.f16"), hdr)
    # Write color space metadata for validation on read
    with open(os.path.join(job, "hdr_meta.json"), "w") as f:
        json.dump({"out_cs": out_cs, "frames": n, "width": w, "height": h}, f)
    if mp4 and os.path.exists(mp4):
        try:
            import shutil
            shutil.copy2(mp4, os.path.join(job, "preview.mp4"))
        except OSError:
            pass
    with open(os.path.join(job, "hdr.json"), "w") as f:
        json.dump({"sig": sig, "frames": n, "width": w, "height": h}, f)
    _finish(job, p, t0, reused=False)


def _read_exr_metadata(exr_dir: str) -> dict:
    """Read chromaticities and colorSpace from the first EXR frame."""
    try:
        import OpenImageIO as oiio
        frames = sorted(Path(exr_dir).glob("frame_*.exr"))
        if not frames:
            return {}
        buf = oiio.ImageBuf(str(frames[0]))
        if buf.has_error:
            return {}
        meta = {}
        # Read colorSpace
        cs = buf.spec().getattribute("colorSpace")
        if cs:
            meta["colorSpace"] = cs
        # Read chromaticities
        chroma = buf.spec().getattribute("chromaticities")
        if chroma:
            meta["chromaticities"] = list(chroma)
        return meta
    except Exception:
        return {}


def _finish(job: str, p: dict, t0: float, reused: bool) -> None:
    n, h, w = int(p["frames"]), int(p["height"]), int(p["width"])
    msg = (f"PREVIEW at {w}x{h}" if p.get("preview") else f"{w}x{h}") + f", done in {time.time() - t0:.0f}s"
    if reused:
        msg += " (cached, no model load)"
    # Add EXR metadata to status message
    exr_dirs = sorted(Path(job).glob("hdr_master_*_exr"))
    if exr_dirs:
        meta = _read_exr_metadata(str(exr_dirs[0]))
        if meta:
            cs = meta.get("colorSpace", "?")
            chroma = meta.get("chromaticities")
            if chroma:
                msg += f" | EXR: {cs} primaries=({chroma[0]:.4f},{chroma[1]:.4f},{chroma[2]:.4f},{chroma[3]:.4f},{chroma[4]:.4f},{chroma[5]:.4f},{chroma[6]:.4f},{chroma[7]:.4f})"
            else:
                msg += f" | EXR: {cs}"
    if reused:
        msg += " (cached, no model load)"
    _write_status(job, "done", 1.0, msg, seconds=round(time.time() - t0, 1),
                  preview=bool(p.get("preview")), cached=reused,
                  in_cs=p.get("in_cs"), out_cs=p.get("out_cs"))
    logf.info("job %s done in %.0fs%s", os.path.basename(job), time.time() - t0, " (cached)" if reused else "")


def _worker() -> None:
    global _last_activity
    while True:
        job = _jobs.get()
        _current["job"] = job
        try:
            if job in _cancel:
                _write_status(job, "cancelled", 0, "cancelled")
                continue
            _process(job)
        except Exception as e:  # noqa: BLE001
            msg = "cancelled" if "cancelled" in str(e) else f"{type(e).__name__}: {e}"
            logf.error("job %s failed: %s\n%s", os.path.basename(job), msg, traceback.format_exc())
            try:
                _write_status(job, "cancelled" if msg == "cancelled" else "error", 0, msg)
            except Exception:  # noqa: BLE001
                pass
        finally:
            _cancel.discard(job)
            _pending.discard(job)
            _current["job"] = None
            _last_activity = time.time()


def _handle(req: dict) -> dict:
    global _last_activity
    _last_activity = time.time()
    cmd = req.get("cmd")
    if cmd == "ping":
        return {"ok": True, "version": VERSION, "busy": _current["job"], "queued": _jobs.qsize(), "pid": os.getpid()}
    if cmd == "convert":
        job = req["job"]
        if not os.path.exists(os.path.join(job, "job.json")):
            return {"ok": False, "error": "job.json missing"}
        if not os.path.exists(os.path.join(job, "input.f16")):
            return {"ok": False, "error": "input.f16 missing"}
        _write_status(job, "queued", 0, f"queued ({_jobs.qsize()} ahead)")
        _pending.add(job)
        _jobs.put(job)
        return {"ok": True, "queued": True}
    if cmd == "warmup":
        # Load the 22B transformer without a clip, so the first real Analyze is fast.
        # Same shape as a job: a folder with job.json + status.json the node can poll.
        folder = os.path.join(ROOT, "run", "warmup")
        os.makedirs(folder, exist_ok=True)
        if _current["job"] == folder or folder in _pending:
            return {"ok": True, "job": folder, "already": True}
        with open(os.path.join(folder, "job.json"), "w") as f:
            json.dump({"warmup": True, "fp8": bool(req.get("fp8", False)),
                       "offload": str(req.get("offload", "cpu"))}, f)
        _write_status(folder, "queued", 0.0, "queued (warmup)")
        _pending.add(folder)
        _jobs.put(folder)
        return {"ok": True, "job": folder}
    if cmd == "cancel":
        _cancel.add(req["job"])
        return {"ok": True}
    if cmd == "status":
        try:
            with open(os.path.join(req["job"], "status.json")) as f:
                return {"ok": True, **json.load(f)}
        except FileNotFoundError:
            return {"ok": True, "state": "none"}
    if cmd == "shutdown":
        threading.Timer(0.2, lambda: os._exit(0)).start()
        return {"ok": True}
    return {"ok": False, "error": f"unknown cmd {cmd}"}


def _serve_conn(conn: socket.socket) -> None:
    with conn:
        buf = b""
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    resp = _handle(json.loads(line.decode()))
                except Exception as e:  # noqa: BLE001
                    resp = {"ok": False, "error": str(e)}
                conn.sendall((json.dumps(resp) + "\n").encode())


def _idle_watch() -> None:
    while True:
        time.sleep(60)
        if _current["job"] is None and _jobs.empty() and time.time() - _last_activity > IDLE_EXIT_S:
            logf.info("idle for %ds, exiting", IDLE_EXIT_S)
            try:
                os.unlink(SOCK)
            except OSError:
                pass
            os._exit(0)


def _apply_model_dirs() -> None:
    """Flame does not pass LTX_MODEL_DIRS to the engine, so honour it from the settings
    file too: {"model_dirs": "/path/a:/path/b"}. An already-set env var still wins."""
    if os.environ.get("LTX_MODEL_DIRS"):
        return
    dirs = settings().get("model_dirs")
    if isinstance(dirs, str) and dirs.strip():
        os.environ["LTX_MODEL_DIRS"] = dirs
        logf.info("model_dirs from settings: %s", dirs)


def main() -> None:
    log.open(LOG)
    _apply_model_dirs()
    try:
        os.unlink(SOCK)
    except OSError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    os.chmod(SOCK, 0o600)
    srv.listen(8)
    threading.Thread(target=_worker, daemon=True).start()
    threading.Thread(target=_idle_watch, daemon=True).start()
    logf.info("DNZ SDR->HDR engine v%d listening on %s (pid %d)", VERSION, SOCK, os.getpid())
    log(f"[hdr-engine] v{VERSION} listening on {SOCK} (pid {os.getpid()})")
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=_serve_conn, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
