#!/usr/bin/env python
"""Prove Clear/Cancel actually STOPS a running clip mid-denoise.

Under the old resident design cancel was only checked before/after the whole pass, so
pressing Clear on a 1080p job let it run to completion and kept the GPU pegged. This
test submits a real job, waits until the denoise loop is running, sends cancel, and
measures how fast the job flips to "cancelled". If the denoise-step hook works it stops
within a step or two (seconds), not at full completion.
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
N, H, W = 9, 512, 512                                     # enough to enter denoise, quick load
WORK = ROOT / "run" / "cancel_test"
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


def status(job):
    return call({"cmd": "status", "job": str(job)})


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    log.open(str(WORK / "test.log"))
    try:
        SOCK.unlink()                                   # drop any stale test socket
    except OSError:
        pass
    # The engine picks its socket up from DNZ_HDR_SOCK, so this test's engine is fully
    # isolated from the one Flame launched (its socket + process are untouched).
    env = os.environ.copy()
    env["DNZ_HDR_SOCK"] = str(SOCK)
    eng_log = open(WORK / "engine.log", "w")
    proc = subprocess.Popen([str(ROOT / "venv" / "bin" / "python"),
                             str(ROOT / "engine" / "dnz_hdr_engine.py")],
                            env=env, stdout=eng_log, stderr=subprocess.STDOUT)
    for _ in range(240):
        if proc.poll() is not None:
            sys.exit(f"engine exited early ({proc.returncode}) - see {WORK/'engine.log'}")
        try:
            s = socket.socket(socket.AF_UNIX); s.connect(str(SOCK)); s.close(); break
        except OSError:
            time.sleep(0.5)
    else:
        sys.exit(f"engine never came up - see {WORK/'engine.log'}")

    job = WORK / "job1"
    job.mkdir(parents=True, exist_ok=True)
    write_f16(str(job / "input.f16"), plate())
    json.dump({"job": str(job), "frames": N, "width": W, "height": H, "fps": 24.0,
               "in_cs": "srgb_linear", "out_cs": "acescg", "seed": 7, "high_quality": False,
               "keyframes": True, "fp8": False, "offload": "none", "preview": False},
              open(job / "job.json", "w"))
    call({"cmd": "convert", "job": str(job)})

    # wait until the denoise loop is actually running (progress ~0.6 = "HDR pass running")
    t0 = time.time()
    while True:
        st = status(job)
        msg = str(st.get("message", ""))
        if st.get("state") in ("done", "error", "cancelled"):
            print(f"job finished before cancel ({st.get('state')}) - test cannot run"); proc.terminate(); sys.exit(1)
        if "HDR pass running" in msg or st.get("progress", 0) >= 0.55:
            break
        if time.time() - t0 > 600:
            print("never reached denoise in 10 min"); proc.terminate(); sys.exit(1)
        time.sleep(1.0)
    in_denoise = time.time() - t0
    print(f"denoise running after {in_denoise:.0f}s - sending cancel")

    call({"cmd": "cancel", "job": str(job)})
    tc = time.time()
    while True:
        st = status(job)
        if st.get("state") in ("cancelled", "done", "error"):
            break
        if time.time() - tc > 120:
            print(f"FAIL: still {st.get('state')} 120s after cancel - hook did NOT stop it")
            proc.terminate(); sys.exit(1)
        time.sleep(0.5)
    took = time.time() - tc
    print(f"job -> {st.get('state')} in {took:.1f}s after cancel")

    proc.terminate()
    if st.get("state") == "cancelled" and took < 30:
        print(f"\nCANCEL WORKS: stopped {took:.1f}s after cancel (no full run)")
    else:
        print(f"\nFAIL: state={st.get('state')} took={took:.1f}s"); sys.exit(1)


if __name__ == "__main__":
    main()
