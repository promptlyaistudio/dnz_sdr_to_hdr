#!/bin/bash
# Build and run the C++ <-> Python integration test. Plants the fake pipeline and
# dummy model files in a scratch root, starts nothing by hand - the C++ client is
# expected to spawn the engine itself, exactly like the plugin does.
set -euo pipefail
KIT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-/Users/Shared/MatteLTXNative/venv}"
PY="$VENV/bin/python"
ROOT="${ROOT:-/tmp/dnz_it_root}"
OFX="${OFX_SDK_DIR:-$HOME/projets/openfx}"
SDK="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"

rm -rf "$ROOT"
mkdir -p "$ROOT"/{engine,models/hdr,run,cache/hdr_jobs,venv/bin,fakesite/ltx_pipelines}
cp "$KIT"/payload/engine/*.py "$ROOT/engine/"
printf '#!/bin/sh\nexec %s "$@"\n' "$PY" > "$ROOT/venv/bin/python"
chmod +x "$ROOT/venv/bin/python"
# dummy model payloads: must look like real files (>1 MB, not an LFS pointer) so
# resolve_models' genuineness check accepts them; the fake pipeline never reads them
dummy() { dd if=/dev/zero of="$1" bs=1048576 count=2 2>/dev/null; }
dummy "$ROOT/models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
dummy "$ROOT/models/ltx-2.5-video-vae-bf16.safetensors"
dummy "$ROOT/models/hdr/ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors"
dummy "$ROOT/models/hdr/ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors"

# ---- fake ltx_pipelines.hdr_ic_lora (same CLI surface, no model)
: > "$ROOT/fakesite/ltx_pipelines/__init__.py"
cat > "$ROOT/fakesite/ltx_pipelines/hdr_ic_lora.py" <<'PYEOF'
import argparse
from pathlib import Path
import numpy as np, OpenImageIO as OIIO
p = argparse.ArgumentParser()
for a in ("--input","--output-path","--input-colorspace","--exr-colorspace","--hdr-lora",
          "--text-embeddings","--transformer-path","--video-vae-path","--offload"):
    p.add_argument(a)
p.add_argument("--frame-rate", type=float, default=None); p.add_argument("--seed", type=int, default=42)
p.add_argument("--no-quantization", action="store_true"); p.add_argument("--high-quality", action="store_true")
p.add_argument("--no-keyframes", action="store_true"); p.add_argument("--keyframe-strength", type=float, default=0.95)
p.add_argument("--diffvae-optimization", action="store_true"); p.add_argument("--negative-prompt", default=None)
args = p.parse_args()
frames = sorted(Path(args.input).glob("*.exr")); n = len(frames)
out = Path(args.output_path)
outdir = out.parent / (out.stem + "_" + args.exr_colorspace + "_exr"); outdir.mkdir(parents=True, exist_ok=True)
for i, f in enumerate(frames):
    px = OIIO.ImageBuf(str(f)).get_pixels(OIIO.FLOAT).astype(np.float32)
    hdr = px * 2.5 + 0.25
    spec = OIIO.ImageSpec(px.shape[1], px.shape[0], 3, OIIO.HALF)
    w = OIIO.ImageOutput.create(str(outdir / f"frame_{i:05d}.exr"))
    w.open(str(outdir / f"frame_{i:05d}.exr"), spec); w.write_image(hdr); w.close()
    print(f"  12%|████ | {i+1}/{n}", flush=True)
out.write_bytes(b"fake hlg master")
PYEOF

# ---- build the test
xcrun clang++ -isysroot "$SDK" -std=c++17 -O1 -I"$OFX/include" -I"$KIT/src" \
    "$KIT/tests/engine_client_it.cpp" "$KIT/src/render.cpp" "$KIT/src/effect.cpp" \
    -o "$KIT/tests/engine_client_it"

# ---- run it (the test spawns the engine itself)
AIFX_NATIVE_ROOT="$ROOT" PYTHONPATH="$ROOT/fakesite" "$KIT/tests/engine_client_it" "$ROOT"
rc=$?
echo
echo "engine log tail:"
tail -5 "$ROOT/run/dnz_hdr_engine.log" 2>/dev/null || true
exit $rc
