#!/bin/bash
# Run every test in the kit. Nothing here needs the real model or a GPU: the pipeline
# is faked, so this is safe to run any time and takes about a minute.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
KIT="$(cd "$HERE/.." && pwd)"
OFX="${OFX_SDK_DIR:-$HOME/projets/openfx}"
SDK="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
pass=0; fail=0
run() {  # name, command...
  local name="$1"; shift
  printf "\n\033[1m%s\033[0m\n" "$name"
  if "$@"; then pass=$((pass+1)); else fail=$((fail+1)); fi
}

printf "\033[1mBuilding the plugin\033[0m\n"
"$KIT/build.sh" >/dev/null || { echo "build FAILED"; exit 1; }

run "1/4 float16 conversion vs numpy" bash -c "
  xcrun clang++ -isysroot '$SDK' -std=c++17 -O1 -I'$OFX/include' -I'$KIT/src' \
      '$KIT/tests/half_roundtrip.cpp' '$KIT/src/render.cpp' '$KIT/src/effect.cpp' \
      -o '$KIT/tests/half_roundtrip' && '$KIT/tests/half_roundtrip'"

run "2/4 OFX describe / instance (mock host)" bash -c "
  xcrun clang++ -isysroot '$SDK' -std=c++17 -O1 -I'$OFX/include' -I'$KIT/src' \
      '$KIT/tests/ofx_host_smoke.cpp' -o '$KIT/tests/ofx_host_smoke' -ldl && '$KIT/tests/ofx_host_smoke'"

run "3/4 engine: job, 8k+1 padding, cache, colour" "$KIT/tests/engine_smoke.sh"

run "4/4 C++ client vs Python engine" "$KIT/tests/engine_client_it.sh"

printf "\n\033[1m%s passed, %s failed\033[0m\n" "$pass" "$fail"
[ "$fail" -eq 0 ]
