#!/bin/bash
# Removes the DNZ SDR to HDR plugin and, if you ask for it, its engine files.
# The Matte LTX Native kit, the shared Python environment and the big models are
# never touched: those belong to the other kit.
B="/Library/OFX/Plugins/DNZSDRtoHDR.ofx.bundle"
ROOT="$(cat "$B/Contents/Resources/engine_root.txt" 2>/dev/null)"
ROOT="${ROOT:-/Users/Shared/MatteLTXNative}"

echo "This removes the DNZ SDR to HDR Flame plugin."
read -r -p "Continue? [y/N] " a; [ "$a" = "y" ] || exit 0

[ -S "$ROOT/run/dnz_hdr.sock" ] && "$ROOT/venv/bin/python" -c "
import socket; c=socket.socket(socket.AF_UNIX); c.connect('$ROOT/run/dnz_hdr.sock'); c.sendall(b'{\"cmd\":\"shutdown\"}\n')" 2>/dev/null

sudo rm -rf "$B" && echo "Plugin removed."

if [ -d "$ROOT" ]; then
  echo
  echo "Also remove from $ROOT:"
  echo "  engine/dnz_hdr_*.py      the HDR engine (a few KB)"
  echo "  cache/hdr_jobs/          stored HDR clips (can be several GB)"
  echo "  models/hdr/              the SDR-to-HDR IC-LoRA + scene-emb (~3 GB)"
  echo
  read -r -p "Remove the HDR engine and its stored clips? [y/N] " b
  if [ "$b" = "y" ]; then
    rm -rf "$ROOT/engine/dnz_hdr_io.py" "$ROOT/engine/dnz_hdr_pipeline.py" "$ROOT/engine/dnz_hdr_engine.py" \
           "$ROOT/engine/__pycache__/dnz_hdr"* "$ROOT/cache/hdr_jobs" "$ROOT/run/dnz_hdr_engine.log" \
           "$ROOT/run/dnz_hdr_engine_stdout.log" "$ROOT/run/dnz_hdr.sock" "$ROOT/hdr_settings.json"
    echo "Engine files and stored clips removed."
  fi
  read -r -p "Also delete the downloaded IC-LoRA (~3 GB)? [y/N] " c
  [ "$c" = "y" ] && rm -rf "$ROOT/models/hdr" && echo "IC-LoRA deleted."
fi
read -r -p "Done. Press Return to close."
