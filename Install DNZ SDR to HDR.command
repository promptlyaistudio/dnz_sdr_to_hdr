#!/bin/bash
# DNZ SDR to HDR - installer for Autodesk Flame.
#
# This kit is a SIBLING of the Matte LTX Native kit, not a replacement: it drops its
# own engine files and its own OFX plugin next to whatever is already installed, and
# reuses the existing Python environment, LTX-2 checkout and the 42 GB transformer
# when it finds them (so a second 42 GB download is never triggered).
set -u
KIT="$(cd "$(dirname "$0")" && pwd)"
LTX_COMMIT="9ec55f9f22798a3198d9c923856824821bc3317e"
DEFAULT_ROOT="/Users/Shared/MatteLTXNative"
PLUGIN_DIR="/Library/OFX/Plugins"
PLUGIN="DNZSDRtoHDR.ofx.bundle"
HDR_REPO="Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR"

bold() { printf "\n\033[1m%s\033[0m\n" "$*"; }
fail() { printf "\n\033[31mERROR: %s\033[0m\n" "$*"; echo; read -r -p "Press Return to close."; exit 1; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
note() { printf "  ! %s\n" "$*"; }

clear
bold "DNZ SDR to HDR - installer"
echo "LTX-2.5 SDR-to-HDR IC-LoRA for Autodesk Flame, running locally (no ComfyUI)."

# ---------------------------------------------------------------- checks
bold "1/6  Checking this Mac"
[ "$(uname -m)" = "arm64" ] || fail "Apple Silicon (M-series) Mac required."
RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
[ "$RAM_GB" -lt 64 ] && fail "${RAM_GB} GB RAM found; at least 64 GB is required."
[ "$RAM_GB" -lt 128 ] && note "${RAM_GB} GB RAM: keep Work Res at 1024 and close other GPU-heavy apps."
ok "Apple Silicon, ${RAM_GB} GB RAM, macOS $(sw_vers -productVersion)"
if [ -d "/Library/OFX/Plugins/MatteLTXNative.ofx.bundle" ]; then
  ok "Matte LTX Native kit found (this install leaves it alone)"
fi

# ---------------------------------------------------------------- location
bold "2/6  Install location"
echo "Reuses an existing kit folder when there is one, so nothing is downloaded twice."
read -r -p "Install folder [$DEFAULT_ROOT]: " ROOT
ROOT="${ROOT:-$DEFAULT_ROOT}"
ROOT="${ROOT/#\~/$HOME}"
mkdir -p "$ROOT" 2>/dev/null || sudo mkdir -p "$ROOT" || fail "Cannot create $ROOT"
[ -w "$ROOT" ] || sudo chown "$(id -un)" "$ROOT" || fail "Cannot write to $ROOT"
mkdir -p "$ROOT"/{engine,models/hdr,cache/hdr_jobs,run}
ok "$ROOT"

# ---------------------------------------------------------------- python env
bold "3/6  Python environment"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "  Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh || fail "Could not install uv."
  export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version | awk '{print $2}')"

if [ ! -d "$ROOT/LTX-2/packages/ltx-core" ]; then
  echo "  Downloading Lightricks LTX-2 (pinned commit)..."
  curl -L --fail -s "https://github.com/Lightricks/LTX-2/archive/${LTX_COMMIT}.tar.gz" -o "$ROOT/ltx2.tgz" \
    || fail "Could not download LTX-2 from GitHub."
  rm -rf "$ROOT/LTX-2" && mkdir -p "$ROOT/LTX-2" \
    && tar -xzf "$ROOT/ltx2.tgz" -C "$ROOT/LTX-2" --strip-components 1 && rm "$ROOT/ltx2.tgz"
  ok "LTX-2 code"
else
  ok "LTX-2 code (already installed)"
fi

[ -x "$ROOT/venv/bin/python" ] || uv venv --python 3.12 "$ROOT/venv" >/dev/null || fail "Could not create the Python environment."
echo "  Installing packages (fast if the Matte kit already did it)..."
uv pip install --python "$ROOT/venv/bin/python" -q \
  -e "$ROOT/LTX-2/packages/ltx-core" -e "$ROOT/LTX-2/packages/ltx-pipelines" \
  opencv-python-headless "huggingface_hub>=1.0" || fail "Package install failed."
"$ROOT/venv/bin/python" -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null \
  || fail "PyTorch cannot see the Apple GPU (Metal)."
"$ROOT/venv/bin/python" -c "import OpenImageIO, cv2" 2>/dev/null \
  || fail "OpenImageIO / OpenCV missing; the HDR engine needs both to read and write EXR."
ok "PyTorch with Metal, OpenImageIO, OpenCV"

cp "$KIT/payload/engine/"*.py "$ROOT/engine/"
ok "engine files -> $ROOT/engine/"

# ---------------------------------------------------------------- models
bold "4/6  SDR-to-HDR IC-LoRA (about 3 GB)"
M="$ROOT/models"
TRANSFORMER="$M/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
VAE="$M/ltx-2.5-video-vae-bf16.safetensors"
[ -f "$TRANSFORMER" ] && ok "22B transformer already present - it will be reused" \
  || note "the 42 GB transformer is not in $M. It will be reused if you install the Matte kit first; otherwise the HDR node reports 'missing model files'."

real_model() {  # real_model <path> - true only for a genuine payload, not an LFS stub
  [ -f "$1" ] || return 1
  [ "$(stat -f%z "$1" 2>/dev/null || echo 0)" -gt 1000000 ] || return 1
  ! head -c 32 "$1" 2>/dev/null | grep -q "version https://git-lfs"
}

TRANSFORMER=""
for cand in "$M/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
            "$M/_ltx25/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"; do
  real_model "$cand" && { TRANSFORMER="$cand"; break; }
done
VAE=""
for cand in "$M/ltx-2.5-video-vae-bf16.safetensors" \
            "$M/_ltx25/vae/ltx-2.5-video-vae-bf16.safetensors"; do
  real_model "$cand" && { VAE="$cand"; break; }
done
if [ -n "$TRANSFORMER" ]; then
  ok "22B transformer already present ($(du -h "$TRANSFORMER" | awk '{print $1}')) - it will be reused"
  MODEL_DIRS="$(dirname "$TRANSFORMER"):$(dirname "${VAE:-$TRANSFORMER}")"
else
  note "the 42 GB transformer is not in $M. If the Matte kit or ComfyUI already has it, set"
  note "model_dirs in $ROOT/hdr_settings.json to that folder; otherwise the node reports"
  note "'Missing model files'."
  MODEL_DIRS=""
fi

HF="$ROOT/venv/bin/hf"
if ! "$HF" auth whoami >/dev/null 2>&1; then
  cat <<TXT
  The HDR IC-LoRA is gated. Before continuing, open this page while logged in to
  Hugging Face and click "Agree and access repository":

      https://huggingface.co/${HDR_REPO}

  (LTX-2.5 itself only needs the same acceptance the Matte kit asked for.) Then create
  a READ token at https://huggingface.co/settings/tokens and paste it below.
TXT
  "$HF" auth login || fail "Hugging Face login failed."
fi
ok "logged in to Hugging Face as $("$HF" auth whoami 2>/dev/null | awk '/user:/{print $2}' | head -1)"

if [ -n "$MODEL_DIRS" ] || [ -n "${KEEP_SETTINGS:-}" ]; then
  # persist where the big models live; Flame does not inherit LTX_MODEL_DIRS, so the
  # engine reads this back from hdr_settings.json at startup
  SETTINGS="$ROOT/hdr_settings.json"
  if [ -f "$SETTINGS" ]; then
    "$ROOT/venv/bin/python" - "$SETTINGS" "$MODEL_DIRS" <<'PY'
import json, sys
p, dirs = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(p))
except Exception:
    d = {}
if dirs:
    d["model_dirs"] = dirs
json.dump(d, open(p, "w"), indent=2)
print(f"  wrote {p}: model_dirs={d.get('model_dirs','')}")
PY
  elif [ -n "$MODEL_DIRS" ]; then
    printf '{"model_dirs": "%s"}\n' "$MODEL_DIRS" > "$SETTINGS"
    echo "  wrote $SETTINGS"
  fi
fi

if [ -f "$M/hdr/ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors" ] \
   && real_model "$M/hdr/ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors" \
   && real_model "$M/hdr/ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors"; then
  ok "SDR-to-HDR IC-LoRA + scene-emb already downloaded"
else
  echo "  SDR-to-HDR IC-LoRA + scene embeddings..."
  "$HF" download "$HDR_REPO" --local-dir "$M/hdr" >"$ROOT/run/download.log" 2>&1 \
    || { tail -5 "$ROOT/run/download.log"; fail "Download failed. Most likely the repository needs approval: open https://huggingface.co/${HDR_REPO} and accept the licence, then re-run this installer."; }
  ok "IC-LoRA + scene-emb"
fi

# ---------------------------------------------------------------- plugin
bold "5/6  Flame plugin (needs your Mac password)"
pgrep -xq flame && note "Flame is running - quit it before using the plugin, then reopen it."
if [ ! -d "$KIT/payload/$PLUGIN" ]; then
  note "No prebuilt plugin in this kit - building it now (needs cmake and the OpenFX SDK)."
  [ -x "$KIT/build.sh" ] && OFX_SDK_DIR="${OFX_SDK_DIR:-$HOME/projets/openfx}" "$KIT/build.sh" \
    || fail "Could not build the plugin. Install cmake (brew install cmake) and re-run."
fi
TMPB="$(mktemp -d)/$PLUGIN"
cp -R "$KIT/payload/$PLUGIN" "$TMPB" || fail "Plugin bundle missing from the kit."
echo "$ROOT" > "$TMPB/Contents/Resources/engine_root.txt"
sudo mkdir -p "$PLUGIN_DIR" && sudo rm -rf "$PLUGIN_DIR/$PLUGIN" \
  && sudo cp -R "$TMPB" "$PLUGIN_DIR/" && sudo xattr -cr "$PLUGIN_DIR/$PLUGIN" \
  || fail "Could not install the plugin into $PLUGIN_DIR"
ok "installed $PLUGIN_DIR/$PLUGIN"

# ---------------------------------------------------------------- self test
bold "6/6  Quick check"
"$ROOT/venv/bin/python" - "$ROOT" <<'PY' || fail "Engine check failed - see above."
import os, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / "engine"))
import dnz_hdr_io, dnz_hdr_pipeline, dnz_hdr_engine   # noqa: imports must not raise
from dnz_hdr_pipeline import resolve_models
try:
    models = resolve_models(root)
    print("  models OK:", ", ".join(sorted(os.path.basename(str(v)) for k, v in models.items() if k != "root")))
except Exception as e:
    print("  models NOT ready:")
    print("   ", str(e).replace("\n", "\n    "))
    sys.exit(1)
print("  engine imports OK")
PY
ok "ready"

bold "Done."
cat <<TXT
  In Flame: add  DNZ > DNZ SDR to HDR  after your plate, set Input Color Space, press
  Analyze, then Refresh Status. First run loads the 22B model (about a minute).
  Engine + models:  $ROOT
  Engine log:       $ROOT/run/dnz_hdr_engine.log
  CLI (no Flame):   $ROOT/venv/bin/python $ROOT/engine/dnz_hdr_pipeline.py --help
TXT
echo; read -r -p "Press Return to close."
