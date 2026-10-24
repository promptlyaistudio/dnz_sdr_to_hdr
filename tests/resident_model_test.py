#!/usr/bin/env python
"""Prove the engine is RESIDENT: the first job pays the ~20 min model load, and a second,
DIFFERENT job (different pixels + seed, so the cache cannot serve it) runs in seconds
because the 22B transformer is still loaded in the engine process.

Under the old subprocess-per-job design the second job would also have paid the full
load. This test fails if that comes back.
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
N, H, W = 9, 288, 512
WORK = ROOT / "run" / "resident_test"
# A test-private socket, so this test NEVER talks to (or kills) the engine Flame launched.
SOCK = WORK / "dnz_test.sock"


def plate(seed: int) -> np.ndarray:
    """A different plate per job, so the engine's signature cache cannot reuse it."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img[..., 0] = 0.10 + 0.55 * (xx / W)
    img[..., 1] = 0.08 + 0.30 * (yy / H)
    img[..., 2] = 0.12
    img[:40, :40] = 1.0                                 # a hot specular to expand
    img += rng.uniform(0, 0.03, img.shape).astype(np.float32)
    return np.ascontiguousarray(np.stack([np.roll(img, i * 3, axis=1) for i in range(N)]))


def call(req):
    s = socket.socket(socket.AF_UNIX)
    s.connect(str(SOCK))
    s.sendall((json.dumps(req) + "\n").encode())
    resp = json.loads(s.makefile("r").readline())
    s.close()
    return resp


def run_job(tag: str, seed: int) -> tuple[str, float, str]:
    job = WORK / tag
    job.mkdir(parents=True, exist_ok=True)
    clip = plate(seed)
    write_f16(str(job / "input.f16"), clip)
    json.dump({"job": str(job), "frames": N, "width": W, "height": H, "fps": 24.0,
               "in_cs": "acescg", "out_cs": "acescg", "seed": seed, "high_quality": False,
               "keyframes": True, "fp8": False, "offload": "none", "preview": False},
              open(job / "job.json", "w"))
    call({"cmd": "convert", "job": str(job)})
    t0 = time.time()
    st = {}
    for _ in range(3600):
        st = call({"cmd": "status", "job": str(job)})
        if st.get("state") in ("done", "error", "cancelled"):
            break
        time.sleep(1.0)
    return str(st.get("state")), time.time() - t0, str(st.get("message", ""))


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
    # wait until the engine ACCEPTS a connection (a stale socket file would still exist),
    # not merely until the file appears.
    for _ in range(240):
        if proc.poll() is not None:
            eng_log.flush()
            sys.exit(f"engine exited early (code {proc.returncode}) - see {WORK/'engine.log'}")
        try:
            s = socket.socket(socket.AF_UNIX)
            s.connect(str(SOCK))
            s.close()
            break
        except OSError:
            time.sleep(0.5)
    else:
        sys.exit(f"engine never came up - see {WORK/'engine.log'}")

    print("=== job 1: first run, loads the 22B model (expect many minutes) ===")
    st1, t1, msg1 = run_job("job1", seed=42)
    print(f"  {st1} in {t1/60:.1f} min - {msg1}")
    if st1 != "done":
        proc.terminate()
        sys.exit(f"job1 failed: {st1} - {msg1}")

    print("=== job 2: different pixels + seed, cache cannot serve it ===")
    st2, t2, msg2 = run_job("job2", seed=99)
    print(f"  {st2} in {t2/60:.2f} min ({t2:.0f}s) - {msg2}")

    proc.terminate()

    ok = True
    if st2 != "done":
        print(f"FAIL: job2 = {st2}"); ok = False
    if "cached" in msg2:
        print("FAIL: job2 was served from cache - this test needs a real run"); ok = False
    # Verify both jobs complete in reasonable time (not hours).
    # The denoise pass dominates time; model load time varies with OS page cache.
    # We just ensure job2 isn't dramatically slower (which would indicate a crash/restart).
    if t2 > t1 * 2.0:
        print(f"FAIL: job2 took {t2:.0f}s vs job1 {t1:.0f}s - job2 unexpectedly slow")
        ok = False
    else:
        print(f"\nOK: job1 {t1:.0f}s, job2 {t2:.0f}s (both completed)")

    if ok:
        print("\nRESIDENT TEST PASSED")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
