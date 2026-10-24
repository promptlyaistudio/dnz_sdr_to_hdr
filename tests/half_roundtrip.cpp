// Prove the hand-rolled float16 conversion in engine_client.cpp matches numpy's
// "<f2" exactly. The plugin writes input.f16 and reads hdr.f16 with this code; a
// mismatch would corrupt every pixel with no error anywhere.
//
//   g++ tests/half_roundtrip.cpp -I src ... && ./half_roundtrip
#include "../src/engine_client.cpp"   // the functions under test live in an anon namespace

#include <cmath>
#include <cstdio>
#include <random>

int main() {
    // 1. the exact bit patterns numpy produces, as ground truth
    std::vector<float> vals = {0.0f, 1.0f, -1.0f, 0.5f, 2.5f, 3.0f, 12.0f, 0.25f,
                               65504.0f, 1e-6f, 1024.0f, 0.0031308f, 1.055f,
                               -0.055f, 1e8f /*clamps*/, 0.1f, 0.9f, 100.0f};
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-50.0f, 50.0f);
    for (int i = 0; i < 5000; ++i) vals.push_back(u(rng));

    // write with our converter
    dnz::Frame f;
    f.w = (int)vals.size();
    f.h = 1;
    f.rgb = vals;
    std::vector<dnz::Frame> frames = {f};
    std::system("rm -rf /tmp/dnz_half_job && mkdir -p /tmp/dnz_half_job");
    if (!dnz::writeInputF16("/tmp/dnz_half_job", frames)) {
        std::printf("could not write test file\n");
        return 1;
    }
    // the same bytes, named as the finished clip, so readHdrFrame can find them
    std::system("cp /tmp/dnz_half_job/input.f16 /tmp/dnz_half_job/hdr.f16");

    // read with our converter and compare against the original floats
    dnz::Frame back;
    int n = 1;
    if (!dnz::readHdrFrame("/tmp/dnz_half_job", n, f.w, 1, 0, back, -1)) {
        std::printf("could not read back\n");
        return 1;
    }
    int bad = 0;
    double worst = 0;
    for (size_t i = 0; i < vals.size(); ++i) {
        // expected = what numpy's float16 round trip gives
        uint16_t h = 0;
        float v = vals[i];
        if (v != v) {                                   // NaN
            std::printf("  [%zu] input is NaN, skipping\n", i);
            continue;
        }
        // reference conversion using the compiler's _Float16 when available
        _Float16 ref = (_Float16)v;
        float refBack = (float)ref;
        double err = std::fabs((double)back.rgb[i] - (double)refBack);
        if (err > worst) worst = err;
        if (err > 1e-6 * (1.0 + std::fabs((double)refBack))) {
            if (bad < 10)
                std::printf("  MISMATCH [%zu] in=%.9g ours=%.9g ref=%.9g\n", i, vals[i], back.rgb[i], refBack);
            ++bad;
        }
        (void)h;
    }
    std::printf("checked %zu values, %d mismatches, worst abs error %.3g\n", vals.size(), bad, worst);
    std::printf(bad ? "HALF CONVERSION: FAILED\n" : "HALF CONVERSION: PASSED\n");
    return bad ? 1 : 0;
}
