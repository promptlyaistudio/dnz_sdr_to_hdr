// Integration test: the plugin's C++ engine client against the real Python engine
// server, using the fake pipeline. This is the only test that crosses the language
// boundary, so it is the one that catches a protocol drift (field names, socket
// framing, job folder layout) before Flame does.
//
//   ./tests/engine_client_it
#include "../src/engine_client.cpp"

#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static int failCount = 0;
static void check(bool ok, const char* what) {
    std::printf("  %-34s %s\n", what, ok ? "OK" : "FAILED");
    if (!ok) ++failCount;
}

int main(int argc, char** argv) {
    const char* root = (argc > 1) ? argv[1] : "/tmp/dnz_it_root";
    setenv("AIFX_NATIVE_ROOT", root, 1);

    std::printf("engine root   : %s\n", dnz::engineRoot().c_str());

    // ---- engine comes up on demand, the way the plugin starts it
    std::string err;
    check(dnz::ensureEngine(&err), "ensureEngine (spawn + ping)");
    if (!err.empty()) std::printf("      %s\n", err.c_str());

    std::string reply;
    // json.dumps writes "ok": true (with the space), so do not match a bare "ok":true
    check(dnz::engineCall("{\"cmd\":\"ping\"}", &reply, &err)
              && reply.find("\"ok\"") != std::string::npos
              && reply.find("true") != std::string::npos,
          "ping");
    std::printf("      reply: %s\n", reply.c_str());

    // ---- a job the way render() would build it
    const int n = 3, w = 64, h = 32;
    std::vector<dnz::Frame> frames(n);
    for (int i = 0; i < n; ++i) {
        frames[i].w = w;
        frames[i].h = h;
        frames[i].rgb.resize(static_cast<size_t>(w) * h * 3);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c)
                    frames[i].rgb[(static_cast<size_t>(y) * w + x) * 3 + c] = 0.5f + 0.01f * i;
    }
    std::string job = dnz::makeJobDir();
    check(!job.empty(), "makeJobDir");
    std::printf("      job: %s\n", job.c_str());

    dnz::EffectData p;
    p.inCS = dnz::IN_ACESCG;
    p.outCS = dnz::OUT_ACESCG;
    p.fps = 24.0;
    p.seed = 11;
    check(dnz::writeJobJson(job, p, n, w, h, false), "writeJobJson");
    check(dnz::writeInputF16(job, frames), "writeInputF16");

    check(dnz::engineCall("{\"cmd\":\"convert\",\"job\":\"" + job + "\"}", &reply, &err),
          "convert submitted");
    std::printf("      reply: %s\n", reply.c_str());

    std::string state;
    double prog = 0;
    std::string msg;
    bool done = false;
    for (int i = 0; i < 240; ++i) {
        dnz::engineStatus(job, state, prog, msg);
        if (state == "done" || state == "error" || state == "cancelled") {
            done = true;
            break;
        }
        ::usleep(500 * 1000);
    }
    std::printf("      final: %s %.0f%% - %s\n", state.c_str(), prog * 100, msg.c_str());
    check(done && state == "done", "job reached done");
    if (!done || state != "done") {
        std::printf("\nRESULT: %d failure(s)\n", failCount + 1);
        return 1;
    }

    // ---- read the HDR clip back the way render() does, and check the values
    dnz::Frame out;
    check(dnz::readHdrFrame(job, n, w, h, 1, out, -1), "readHdrFrame");
    check(out.w == w && out.h == h, "returned frame has the job's size");
    float mn = 1e30f, mx = -1e30f;
    for (float v : out.rgb) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    std::printf("      values: min %.4f max %.4f\n", mn, mx);
    // fake pipeline: x * 2.5 + 0.25 from a 0.51 plate -> ~1.525, i.e. above 1.0
    check(mx > 1.0f, "HDR values above 1.0 came back");
    check(mx < 10.0f && mn > 0.0f, "values are sane (no inf / NaN)");

    // ---- a second identical job should be served from the engine's cache
    std::string job2 = dnz::makeJobDir();
    dnz::writeJobJson(job2, p, n, w, h, false);
    dnz::writeInputF16(job2, frames);
    dnz::engineCall("{\"cmd\":\"convert\",\"job\":\"" + job2 + "\"}", &reply, &err);
    bool cached = false;
    for (int i = 0; i < 120; ++i) {
        dnz::engineStatus(job2, state, prog, msg);
        if (state == "done" || state == "error") break;
        ::usleep(250 * 1000);
    }
    cached = (msg.find("cached") != std::string::npos);
    std::printf("      second job: %s - %s\n", state.c_str(), msg.c_str());
    check(state == "done" && cached, "identical job reused the cache");

    // ---- colour-space names the plugin sends must be ones the engine accepts
    for (int cs = 0; cs <= 3; ++cs) {
        std::string nm = dnz::inCsName(cs);
        bool ok = nm == "srgb_linear" || nm == "acescg" || nm == "acescct" || nm == "display_srgb";
        check(ok, (std::string("inCsName ") + std::to_string(cs) + " = " + nm).c_str());
    }

    std::printf(failCount ? "\nRESULT: %d failure(s)\n" : "\nRESULT: all checks passed\n", failCount);
    return failCount ? 1 : 0;
}
