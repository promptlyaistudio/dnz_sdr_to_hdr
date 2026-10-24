# DNZ SDR to HDR

> **Turn an SDR plate into HDR with the LTX-2.5 *SDR-to-HDR* IC-LoRA — as a native Autodesk Flame OFX node.** Runs entirely on your Apple Silicon Mac: no ComfyUI, no shared folders, no cloud.

## What it does

Feed the node an SDR plate, get back:

1. **A scene-linear EXR clip** — ACEScg by default, ready for grading.
2. **An HLG master** you can look at right away.

The heavy lifting (the LTX-2.5 transformer, ~42 GB) runs locally and is downloaded and
managed by the install command on first run — after that it is cached and shared with
this author's sibling native kits, so installing one more OFX node costs ~3 GB, not 45.

## Architecture (short version)

```
Flame (OFX host, C++ plugin)              Local engine (Python)
┌────────────────────────┐  unix socket  ┌──────────────────────────┐
│  src/effect.cpp        │ ─────────────►│  payload/engine/          │
│  src/engine_client.cpp │  JSON lines   │   dnz_hdr_pipeline.py     │
│  • job folders, f16    │ ◄─────────────│   dnz_hdr_engine.py       │
│    streaming, cancel   │  status/cancel│   dnz_hdr_io.py           │
└────────────────────────┘               └──────────────────────────┘
```

- **C++ OFX side** — host UI (`group()/inGroup()`), job-folder based transport,
  `float16` frame streaming, robust cancel (socket *and* marker file).
- **Python engine side** — LTX-2.5 execution, color pipeline toward ACEScg/HLG,
  queue, progress and cancel handling.
- **Field-tested paths** — socket parsing edge cases (empty first tokens of absolute
  paths), `half`-precision round-trips and cancellation behaviour are all covered in
  `tests/`.

## Repo layout

| Path | What |
|---|---|
| `src/` | OFX plugin + engine client (C++) |
| `payload/engine/` | Python engine: `dnz_hdr_pipeline.py`, `dnz_hdr_engine.py`, `dnz_hdr_io.py` |
| `cmake/` | macOS bundle plumbing (Info.plist) |
| `tests/` | smoke tests: OFX host probe, half round-trip, engine client IT, cancel test, progress test |
| `Install DNZ SDR to HDR.command` / `Uninstall …` | one-click macOS install/uninstall |
| `ARCHITECTURE.md` | the deeper design notes |

## Build

```bash
git clone https://github.com/dnzyc/dnz_sdr_to_hdr.git
cd dnz_sdr_to_hdr
cmake -B build && cmake --build build        # or see build.sh
```

Or simply run `Install DNZ SDR to HDR.command` (macOS, Apple Silicon).
Requirements: macOS 13+, Apple Silicon, Python 3.12 environment,
a Hugging Face token with **READ** access for the one-time model pull.

Drop the produced `DNZSDRtoHDR.ofx.bundle` into your Flame OFX plug-in directory
and restart the host.

## Status

Built and actively used as part of the **Promptly AI Studio** toolkit for
professional post-production — integrating generative video/image models
(LTX-2 family) into node-based DCC pipelines.

## Sibling project

- [`dnz-relight-ofx`](https://github.com/dnzyc/dnz-relight-ofx) — IC-Light style
  AI relighting as a native Flame OFX node.

## License

MIT — see [LICENSE](LICENSE).
