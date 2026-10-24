#!/usr/bin/env python
"""Watch a real job's status and prove the progress BAR MOVES during the denoise pass.

Before the step hook the status sat frozen near 60% for the whole denoise (the pipeline
exposes no step callback). This samples the engine status during the pass and prints the
distinct progress values seen; it passes if progress advances through at least 3 distinct
values while the message reports denoise steps.
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT / "payload" / "engine"))
from dnz_hdr_io import write_f16, log                    # noqa: E402

ROOT = Path(os.environ.get("AIFX_NATIVE_ROOT", "/Users/Shared/MatteLTXNative"))
N, H, W = 9, 512, 512
WORK = ROOT / "run" / "progress_test"
# A test-private socket, so this test NEVER talks to (or kills) the engine Flame launched.
SOCK = WORK / "dnz_test.sock"


def plate() -> np.ndarray:
    img = np.zeros((H, W, 3), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img[..., 0] = 0.10 + 0.55 * (xx / W)
    img[..., 1] = 0.08 + 0.30 * (yy / H)
    img[..., 2] = 0.12
    img[:40, :40] = 1.0
    return np.ascontiguousarray(np.stack([np.roll(img, i * 3, axis=1) for i in range(N)]))


def call(req):
    s = socket.socket(socket.AF_UNIX)
    s.connect(str(SOCK))
    s.sendall((json.dumps(req) + "\n").encode())
    resp = json.loads(s.makefile("r").readline())
    s.close()
    return resp


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    log.open(str(WORK / "test.log"))
    try:
        SOCK.unlink()                                   # drop any stale test socket
    except OSError:
        pass
    # Private socket keeps this test's engine isolated from the one Flame launched.
    env = os.environ.copy()
    env["DNZ_HDR_SOCK"] = str(SOCK)
    eng_log = open(WORK / "engine.log", "w")
    proc = subprocess.Popen([str(ROOT / "venv" / "bin" / "python"),
                             str(ROOT / "engine" / "dnz_hdr_engine.py")],
                            env=env, stdout=eng_log, stderr=subprocess.STDOUT)
    for _ in range(240):
        if proc.poll() is not None:
            sys.exit(f"engine exited early - see {WORK/'engine.log'}")
        try:
            s = socket.socket(socket.AF_UNIX); s.connect(str(SOCK)); s.close(); break
        except OSError:
            time.sleep(0.5)
    else:
        sys.exit("engine never came up")

    job = WORK / "job1"
    job.mkdir(parents=True, exist_ok=True)
    write_f16(str(job / "input.f16"), plate())
    json.dump({"job": str(job), "frames": N, "width": W, "height": H, "fps": 24.0,
               "in_cs": "srgb_linear", "out_cs": "acescg", "seed": 3, "high_quality": False,
               "keyframes": True, "fp8": False, "offload": "none", "preview": False},
              open(job / "job.json", "w"))
    call({"cmd": "convert", "job": str(job)})

    seen = []          # (progress, message) samples during denoise
    t0 = time.time()
    while True:
        st = call({"cmd": "status", "job": str(job)})
        if st.get("state") in ("done", "error", "cancelled"):
            break
        prog = st.get("progress", 0)
        msg = str(st.get("message", ""))
        if "denoise step" in msg or 0.54 <= prog < 0.91:
            if not seen or seen[-1][0] != prog:
                seen.append((prog, msg))
        if time.time() - t0 > 420:
            break
        time.sleep(0.4)

    print("distinct denoise progress samples:")
    for p, m in seen:
        print(f"  {int(p*100):3d}%  {m}")
    proc.terminate()

    steps = [p for p, m in seen if "denoise step" in m]
    if len(seen) >= 3 and steps:
        print(f"\nPROGRESS MOVES: {len(seen)} distinct values, {len(steps)} step-lines "
              f"({int(min(steps)*100)}% -> {int(max(steps)*100)}%)")
    else:
        print(f"\nFAIL: only {len(seen)} distinct values / {len(steps)} step lines"); sys.exit(1)


if __name__ == "__main__":
    main()
