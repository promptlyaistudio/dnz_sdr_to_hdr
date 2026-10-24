#!/bin/bash
# End-to-end engine test with a FAKE pipeline: no model download, no GPU.
#
# A stand-in ltx_pipelines.hdr_ic_lora module is planted in a scratch ROOT. It
# accepts exactly the flags the real one does, reads the EXR folder the engine
# writes, and writes back a recognisable HDR clip (values above 1.0) plus the
# EXR sidecar folder and the MP4 placeholder. That exercises everything on our
# side that could be wrong: job folder, f16 round trip, EXR writing, colour-space
# plumbing, 8k+1 padding, EXR reading back, output f16, status, cache reuse.
set -euo pipefail
KIT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-/Users/Shared/MatteLTXNative/venv}"
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "no venv python at $PY"; exit 1; }

ROOT="$(mktemp -d /tmp/dnz_hdr_test.XXXXXX)"
trap 'rm -rf "$ROOT"' EXIT
mkdir -p "$ROOT"/{engine,models,models/hdr,run,cache/hdr_jobs}
cp "$KIT"/payload/engine/*.py "$ROOT/engine/"

# fake model files (the pipeline never opens them; resolve_models only checks they exist)
# dummy model payloads: must look like real files (>1 MB, not an LFS pointer) so
# resolve_models' genuineness check accepts them; the fake pipeline never reads them
dummy() { dd if=/dev/zero of="$1" bs=1048576 count=2 2>/dev/null; }
dummy "$ROOT/models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
dummy "$ROOT/models/ltx-2.5-video-vae-bf16.safetensors"
dummy "$ROOT/models/hdr/ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors"
dummy "$ROOT/models/hdr/ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors"
[ -x "$ROOT/venv/bin/python" ] || { mkdir -p "$ROOT/venv/bin";
  printf '#!/bin/sh\nexec %s "$@"\n' "$PY" > "$ROOT/venv/bin/python"; chmod +x "$ROOT/venv/bin/python"; }

# ---- fake ltx_pipelines.hdr_ic_lora -------------------------------------------------
SITE="$ROOT/fakesite"
mkdir -p "$SITE/ltx_pipelines"
cat > "$SITE/ltx_pipelines/__init__.py" <<'EOF'
EOF
cat > "$SITE/ltx_pipelines/hdr_ic_lora.py" <<'PYEOF'
"""Fake HDR pipeline: same CLI surface, no model. Prints a progress bar like the real one."""
import argparse, os, sys
from pathlib import Path

import numpy as np
import OpenImageIO as OIIO

p = argparse.ArgumentParser()
for a in ("--input", "--output-path", "--input-colorspace", "--exr-colorspace", "--hdr-lora",
          "--text-embeddings", "--transformer-path", "--video-vae-path", "--offload"):
    p.add_argument(a)
p.add_argument("--frame-rate", type=float, default=None)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--no-quantization", action="store_true")
p.add_argument("--high-quality", action="store_true")
p.add_argument("--no-keyframes", action="store_true")
p.add_argument("--keyframe-strength", type=float, default=0.95)
p.add_argument("--diffvae-optimization", action="store_true")
p.add_argument("--negative-prompt", default=None)
args = p.parse_args()

src = Path(args.input)
frames = sorted(src.glob("*.exr"))
if not frames:
    sys.exit("fake pipeline: no EXR frames in " + str(src))
n = len(frames)
if (n - 1) % 8 != 0:
    sys.exit(f"fake pipeline: {n} frames is not 8k+1")

out = Path(args.output_path)
outdir = out.parent / (out.stem + "_" + args.exr_colorspace + "_exr")
outdir.mkdir(parents=True, exist_ok=True)

for i, f in enumerate(frames):
    buf = OIIO.ImageBuf(str(f))
    px = buf.get_pixels(OIIO.FLOAT).astype(np.float32)
    # HDR: lift the plate above 1.0 and tag the colour space on disk
    hdr = px * 2.5 + 0.25
    spec = OIIO.ImageSpec(px.shape[1], px.shape[0], 3, OIIO.HALF)
    spec.attribute("oiio:ColorSpace", args.exr_colorspace)
    w = OIIO.ImageOutput.create(str(outdir / f"frame_{i:05d}.exr"))
    w.open(str(outdir / f"frame_{i:05d}.exr"), spec)
    w.write_image(hdr)
    w.close()
    print(f"  12%|████      | {i+1}/{n} [00:0{min(i,9)}<00:09, 1.0it/s]", flush=True)

out.write_bytes(b"fake hlg master")
print(f"fake pipeline: {n} frames -> {outdir}")
PYEOF

# ---- run the engine the way the plugin does ----------------------------------------
export AIFX_NATIVE_ROOT="$ROOT"
export PYTHONPATH="$SITE"
"$PY" "$ROOT/engine/dnz_hdr_engine.py" > "$ROOT/run/stdout.log" 2>&1 &
ENG=$!
trap 'kill $ENG 2>/dev/null || true; rm -rf "$ROOT"' EXIT
for i in $(seq 1 60); do [ -S "$ROOT/run/dnz_hdr.sock" ] && break; sleep 0.5; done
[ -S "$ROOT/run/dnz_hdr.sock" ] || { cat "$ROOT/run/stdout.log"; echo "engine did not start"; exit 1; }
echo "engine up (pid $ENG)"

"$PY" - "$ROOT" <<'PYEOF'
import json, os, socket, sys, time
import numpy as np
sys.path.insert(0, os.path.join(sys.argv[1], "engine"))
from dnz_hdr_io import write_f16, read_f16, read_exr, log

root = sys.argv[1]
log.open(os.path.join(root, "run", "test.log"))

def call(req):
    s = socket.socket(socket.AF_UNIX); s.connect(os.path.join(root, "run", "dnz_hdr.sock"))
    s.sendall((json.dumps(req) + "\n").encode())
    f = s.makefile("r"); resp = json.loads(f.readline()); s.close(); return resp

print("ping          :", call({"cmd": "ping"}))

# Warm Up: no clip, just load the model. In this harness HDRICLoraPipeline is a fake CLI
# with no class, so the load cannot succeed - what is being tested here is the plumbing:
# command -> queue -> worker -> a status.json the node can poll.
wu = call({"cmd": "warmup", "offload": "none"})
print("warmup        :", wu)
assert wu.get("ok") and wu.get("job"), wu
wust = {}
for _ in range(60):
    wust = call({"cmd": "status", "job": wu["job"]})
    if wust.get("state") in ("done", "error", "cancelled"):
        break
    time.sleep(0.5)
print("warmup status :", wust.get("state"), "-", str(wust.get("message"))[:60])
assert wust.get("state") in ("done", "error", "cancelled"), wust

# a 10-frame plate, 64x32, ACEScg scene-linear with a bright highlight
n, h, w = 10, 32, 64
clip = np.zeros((n, h, w, 3), np.float32)
clip[..., 0] = 0.5
clip[:, :8, :, 1] = 3.0            # a value well above 1.0 survives the round trip
clip[0, 0, 0, 2] = 12.0
job = os.path.join(root, "cache", "hdr_jobs", "job1")
os.makedirs(job, exist_ok=True)
write_f16(os.path.join(job, "input.f16"), clip)
json.dump({"job": job, "frames": n, "width": w, "height": h, "fps": 24.0,
           "in_cs": "acescg", "out_cs": "acescg", "seed": 7, "high_quality": False,
           "keyframes": True, "fp8": False, "offload": "none", "preview": False},
          open(os.path.join(job, "job.json"), "w"))
print("convert       :", call({"cmd": "convert", "job": job}))

for i in range(120):
    st = call({"cmd": "status", "job": job})
    if st.get("state") in ("done", "error", "cancelled"):
        break
    time.sleep(0.5)
print("status        :", st["state"], st.get("progress"), "-", st.get("message"))
assert st["state"] == "done", st

out = read_f16(os.path.join(job, "hdr.f16"), n, h, w, 3)
print(f"hdr.f16       : {out.shape}, min {out.min():.3f} max {out.max():.3f}")
# fake pipeline does x*2.5+0.25 in ACEScg, out_cs acescg (no display conversion)
expect = np.clip(clip, 0, None) * 2.5 + 0.25
err = float(np.abs(out - expect).max())
print(f"pixel error   : {err:.2e}")
assert err < 1e-2, f"round trip wrong: {err}"
assert out.max() > 2.0, "HDR values above 1.0 did not survive"
print("EXR written   :", sorted(os.listdir(os.path.join(job, "work", "in_exr")))[:2], "...")

# same settings again -> cache hit, no pipeline run
t0 = time.time()
job2 = os.path.join(root, "cache", "hdr_jobs", "job2")
os.makedirs(job2, exist_ok=True)
write_f16(os.path.join(job2, "input.f16"), clip)
json.dump({**json.load(open(os.path.join(job, "job.json"))), "job": job2},
          open(os.path.join(job2, "job.json"), "w"))
call({"cmd": "convert", "job": job2})
for i in range(60):
    st2 = call({"cmd": "status", "job": job2})
    if st2.get("state") in ("done", "error"): break
    time.sleep(0.25)
print("cached run    :", st2["state"], f"{time.time()-t0:.1f}s -", st2.get("message"))
assert st2.get("cached") is True, "second identical job should have been cached"
out2 = read_f16(os.path.join(job2, "hdr.f16"), n, h, w, 3)
assert float(np.abs(out2 - out).max()) == 0.0, "cached clip differs"

# display-sRGB input path: gamma in -> linear Rec.709 for the pipeline
job3 = os.path.join(root, "cache", "hdr_jobs", "job3")
os.makedirs(job3, exist_ok=True)
disp = np.full((n, h, w, 3), 0.5, np.float32)          # 0.5 display = ~0.214 linear
write_f16(os.path.join(job3, "input.f16"), disp)
json.dump({**json.load(open(os.path.join(job, "job.json"))), "job": job3, "in_cs": "display_srgb",
           "out_cs": "display_srgb"}, open(os.path.join(job3, "job.json"), "w"))
call({"cmd": "convert", "job": job3})
for i in range(60):
    st3 = call({"cmd": "status", "job": job3})
    if st3.get("state") in ("done", "error"): break
    time.sleep(0.5)
print("display path  :", st3["state"], "-", st3.get("message"))
assert st3["state"] == "done", st3
d = read_f16(os.path.join(job3, "hdr.f16"), n, h, w, 3)
lin = 0.5 ** 2.4 * 1.055 - 0.055 if False else ((0.5 + 0.055) / 1.055) ** 2.4   # sRGB EOTF
pipe = lin * 2.5 + 0.25
expect_disp = 1.055 * pipe ** (1 / 2.4) - 0.055                               # back to gamma
print(f"display gamma : in 0.5 -> pipe {pipe:.4f} -> out {float(d[0,0,0,0]):.4f} (expect {expect_disp:.4f})")
assert abs(float(d[0, 0, 0, 0]) - expect_disp) < 2e-2, "display sRGB round trip wrong"

print("\nENGINE TEST PASSED")
PYEOF
