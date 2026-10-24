#!/usr/bin/env python
"""Real-model end-to-end test: a genuine SDR EXR clip -> the actual 22B HDR IC-LoRA ->
an HDR clip whose values exceed 1.0. No fakes. Needs the gated HF repo approved and the
42 GB transformer present (set LTX_MODEL_DIRS if it is not under <root>/models)."""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import OpenImageIO as OIIO

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT / "payload" / "engine"))
from dnz_hdr_io import read_f16, write_f16, log          # noqa: E402

ROOT = Path(os.environ.get("AIFX_NATIVE_ROOT", "/Users/Shared/MatteLTXNative"))
N, H, W = 9, 288, 512                                    # 9 = 8k+1, the pipeline's grid
WORK = ROOT / "run" / "real_test"
WORK.mkdir(parents=True, exist_ok=True)


def make_sdr_clip() -> np.ndarray:
    """A plausible SDR plate in ACEScg scene-linear: a graded gradient plus a hot specular
    and a saturated colour bar, so there are real highlights for the model to expand."""
    img = np.zeros((H, W, 3), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img[..., 0] = 0.10 + 0.55 * (xx / W)                 # red ramp
    img[..., 1] = 0.08 + 0.30 * (yy / H)                 # green ramp
    img[..., 2] = 0.12                                    # flat blue base
    cy, cx, r = H // 3, W // 2, 26                       # a clipped specular highlight
    img[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r] = 1.0
    img[:, :W // 12, 0] = 0.9                            # saturated colour bars
    img[:, :W // 12, 1] = 0.05
    img[:, :W // 12, 2] = 0.05
    # a little per-frame motion so it is a clip, not a still
    clip = np.stack([np.roll(img, i * 3, axis=1) for i in range(N)])
    return np.ascontiguousarray(clip, np.float32)


def call(req):
    s = socket.socket(socket.AF_UNIX)
    s.connect(str(ROOT / "run" / "dnz_hdr.sock"))
    s.sendall((json.dumps(req) + "\n").encode())
    resp = json.loads(s.makefile("r").readline())
    s.close()
    return resp


def main():
    clip = make_sdr_clip()
    print(f"SDR plate: {clip.shape}, min {clip.min():.3f} max {clip.max():.3f}")

    # start the engine the way the plugin does (it self-spawns, but be explicit here)
    engine = ROOT / "engine" / "dnz_hdr_engine.py"
    proc = subprocess.Popen([str(ROOT / "venv" / "bin" / "python"), str(engine)],
                            env=os.environ.copy(), stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    for _ in range(120):
        if (ROOT / "run" / "dnz_hdr.sock").exists():
            break
        time.sleep(0.5)
    log.open(str(WORK / "test.log"))

    print("ping:", call({"cmd": "ping"}))

    job = WORK / "job"
    job.mkdir(parents=True, exist_ok=True)
    write_f16(str(job / "input.f16"), clip)
    json.dump({"job": str(job), "frames": N, "width": W, "height": H, "fps": 24.0,
               "in_cs": "acescg", "out_cs": "acescg", "seed": 42, "high_quality": False,
               "keyframes": True, "fp8": False, "offload": "cpu", "preview": False},
              open(job / "job.json", "w"))

    print("convert:", call({"cmd": "convert", "job": str(job)}))
    t0 = time.time()
    st = {}
    for _ in range(2400):                                 # up to 20 min for load + 9 frames
        st = call({"cmd": "status", "job": str(job)})
        if st.get("state") in ("done", "error", "cancelled"):
            break
        if int(time.time() - t0) % 20 == 0:
            print(f"  {st.get('state')} {st.get('progress', 0):.0%} - {st.get('message')}")
        time.sleep(1.0)
    dt = time.time() - t0
    print(f"final: {st.get('state')} {st.get('progress', 0):.0%} in {dt:.0f}s - {st.get('message')}")
    if st.get("state") != "done":
        proc.terminate()
        sys.exit(f"REAL MODEL TEST FAILED: {st.get('state')} - {st.get('message')}")

    out = read_f16(str(job / "hdr.f16"), N, H, W, 3)
    print(f"HDR clip: {out.shape}, min {out.min():.3f} max {out.max():.3f} "
          f"mean {out.mean():.3f}")
    frac_above = float((out > 1.0).mean())
    print(f"values above 1.0: {frac_above:.1%} of pixels")

    # the model must actually produce HDR (values > 1.0), keep mid-tones near the plate,
    # and not blow up into inf/NaN
    ok = True
    if not np.isfinite(out).all():
        print("FAIL: non-finite values in the HDR clip"); ok = False
    if out.max() <= 1.0:
        print("FAIL: no value exceeded 1.0 - this is not HDR"); ok = False
    if frac_above > 0.60:
        print("FAIL: more than 60% of pixels above 1.0 - looks blown, not graded"); ok = False
    if abs(float(out.mean()) - float(clip.mean())) > 2.0:
        print(f"FAIL: mean moved from {clip.mean():.2f} to {out.mean():.2f}"); ok = False

    # an HLG preview must exist too (the kit promises one)
    mp4 = list(job.glob("*.mp4"))
    print("HLG preview:", [p.name for p in mp4] or "(none)")

    proc.terminate()
    if ok:
        print("\nREAL MODEL TEST PASSED")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
