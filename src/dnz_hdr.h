// DNZ SDR to HDR - shared declarations for the OFX plugin.
// 
// The plugin is a thin, patient shell around the engine: it never loads a model.
// On "Convert" it gathers the clip, writes it as float16 raw pixels into a job
// folder, and asks the engine (a Python server we spawn) to fill that folder with
// the HDR clip. Rendering just reads the finished clip back. Same shape as the
// Matte plugin, so the two sit side by side without surprises.
#pragma once

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxParam.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace dnz {

// ---------------------------------------------------------------- colour spaces
// Keep in sync with dnz_hdr_io.IN_SPACES / OUT_SPACES on the Python side.
// Order matches the node's Input Color Space / Output Color Space menus.
enum InCS { IN_SRGB_LINEAR = 0, IN_ACESCG = 1, IN_ACESCCT = 2, IN_DISPLAY_SRGB = 3 };
enum OutCS { OUT_ACESCG = 0, OUT_SRGB_LINEAR = 1, OUT_ACESCCT = 2 };
enum OutMode { MODE_HDR = 0, MODE_SDR = 1, MODE_SIDE_BY_SIDE = 2 };

const char* inCsName(int v);
const char* outCsName(int v);

// ---------------------------------------------------------------- effect state
struct EffectData {
    int inCS = IN_SRGB_LINEAR;
    int outCS = OUT_SRGB_LINEAR;   // default: the plate's own space - linear light, Rec.709 primaries
    double fps = 24.0;
    int seed = 42;
    int workRes = 1920;      // long side of the working resolution (0 = native)
    int previewRes = 768;
    int highQuality = 0;
    int keyframes = 1;       // seam keyframes (official pipeline default)
    int fp8 = 0;
    int offload = 0;         // 0 none, 1 cpu, 2 disk
    int rangeStart = 0;      // first frame to convert, 0 = clip start
    int rangeEnd = 0;        // last frame to convert,  0 = clip end
    int outputMode = MODE_HDR;
    double exposure = 0.0;   // stops, applied live in render()
    double highlightGain = 1.0;
    int clipPreview = 0;
    // Additive black pedestal removal. 0 = untouched.
    double blackLevel = 0.0;
    // Tiling (VAE decode) - 0 = auto
    int tileFrames = 0;
    int tileHeight = 0;
    int tileWidth = 0;
    int overlapFrames = 0;
    int overlapHeight = 0;
    int overlapWidth = 0;
    // 8-bit display knee: identity below kneeK, rolls to 1 above. 0 = hard clip.
    double kneeK = 0.8;         // knee point in linear AP1
    double kneeStrength = 1.0;  // 0=hard clip, 1=full knee
    // 8-bit gamut handling above 1.0: 0 = keep hue (scale by 1/max, darker),
    // 1 = clip per channel (brighter, hue shifts). See gamutClamp.
    double gamutClip = 0.0;
};

    // ---------------------------------------------------------------- job state
    struct JobState {
        std::mutex mutex;
        std::string folder;           // current job folder, "" = nothing converted yet
        std::string lastError;
        bool running = false;
        bool haveClip = false;        // hdr.f16 is finished and matches workRes/fps/colour choices
        bool metaValidated = false;   // metadata validation done
        bool metaMismatch = false;    // metadata out_cs != expected outCS
        int n = 0, w = 0, h = 0;
        int inCS = IN_SRGB_LINEAR, outCS = OUT_SRGB_LINEAR;
        double fps = 24.0;
        // live grade, editable without re-analyzing
        int outputMode = MODE_HDR;
        double exposure = 0.0;
        double highlightGain = 1.0;
        int clipPreview = 0;
        // additive black pedestal removal, editable without re-analyzing
        double blackLevel = 0.0;
        // 8-bit display knee, editable without re-analyzing
        double kneeK = 0.8;
        double kneeStrength = 1.0;
        // 8-bit gamut handling above 1.0, editable without re-analyzing
        double gamutClip = 0.0;
    };
    JobState& jobState();

    // ---------------------------------------------------------------- small helpers
    struct OfxGlobals {
        OfxHost* host = nullptr;
        OfxImageEffectSuiteV1* effectSuite = nullptr;
        OfxPropertySuiteV1* propertySuite = nullptr;
        OfxParameterSuiteV1* paramSuite = nullptr;
    };
    extern OfxGlobals g;
    extern OfxPlugin pluginStruct;

    // A frame pulled out of the host: float32 RGB, top row first, ready to hand over.
    struct Frame {
        int w = 0, h = 0;
        std::vector<float> rgb;  // w*h*3
    };

    // param access
    double getDouble(OfxParamSetHandle ps, const char* n, double t);
    int getInt(OfxParamSetHandle ps, const char* n, double t);
    int getBool(OfxParamSetHandle ps, const char* n, double t);
    std::string getString(OfxParamSetHandle ps, const char* n);
    void setStringParam(OfxParamSetHandle ps, const char* n, const std::string& v);

    // image access
    bool fetchFrame(OfxImageEffectHandle effect, const char* clipName, double time, Frame& out);
    void writeFrameToImage(OfxImageEffectHandle effect, double time, const Frame& src);
    Frame resized(const Frame& src, int w, int h);

    // Read HDR frame from job folder; expectedOutCS is optional for metadata validation
    bool readHdrFrame(const std::string& job, int n, int w, int h, int index, Frame& out, int expectedOutCS = -1);

    // work resolution for a clip
    void workSizeFor(int w, int h, int longSide, int& ow, int& oh);

    // ---------------------------------------------------------------- colour conversion (AP1/ACEScg <-> Rec.709)
    // ACEScg (AP1) primaries to Rec.709 primaries (both D65, no chromatic adaptation).
    inline void ap1ToRec709(float& r, float& g, float& b) {
        float nr =  1.45143932f * r - 0.23651075f * g - 0.21492857f * b;
        float ng = -0.07655377f * r + 1.17622970f * g - 0.09967593f * b;
        float nb =  0.00831615f * r - 0.00603245f * g + 0.99771630f * b;
        r = nr; g = ng; b = nb;
    }
    // ACEScct (log) -> ACEScg (linear AP1). Spec S-2016-001, Eq. 2.
    //   lin = (cct - B) / A            for cct <= Y_break   (the toe)
    //   lin = 2^(cct*17.52 - 9.72)     for Y_break < cct < (log2(65504)+9.72)/17.52
    //   lin = 65504                    above that (half-float ceiling)
    inline float acescctToAcescg(float y) {
        const float A = 10.5402377416545f;
        const float B = 0.0729055341958355f;
        const float Y_break = 0.155251141552511f;      // ACEScct at the toe break
        const float Y_max = (std::log2(65504.0f) + 9.72f) / 17.52f;
        if (y <= 0.0f) return 0.0f;
        if (y <= Y_break) return std::max(0.0f, (y - B) / A);   // toe maps y<B to black
        if (y >= Y_max) return 65504.0f;
        return std::exp2(y * 17.52f - 9.72f);
    }

    // ACEScg (linear AP1) -> ACEScct. Spec S-2016-001, Eq. 1. Exact inverse of the above.
    inline float acescgToAcescct(float x) {
        const float A = 10.5402377416545f;
        const float B = 0.0729055341958355f;
        const float X_break = 0.0078125f;
        if (x <= X_break) return A * x + B;
        return (std::log2(x) + 9.72f) / 17.52f;
    }
    // Linear Rec.709 -> sRGB gamma (piecewise)
    inline float linearToSrgbGamma(float x) {
        x = std::max(0.0f, x);
        return (x <= 0.0031308f) ? x * 12.92f : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
    }

// --------------------------------------------------------------------------- display knee
// For 8-bit output the HDR clip has to be brought into [0,1]. Every classic tonemap
// (Reinhard, Hable, ACES-approx, "log") reshapes the SDR range on the way: SDR white
// 1.0 lands below 1.0, so an SDR plate next to its own HDR render reads grey and milky.
// What an 8-bit output actually needs is a curve that is the exact identity up to a knee
// and only then rolls off to 1:
//
//     f(x) = x                                                  x <= k
//     f(x) = k + (1-k) * (1 - exp(-(x-k)/(1-k)))                x >  k
//
// Checked numerically: identity below k, f(0)=0, monotone, bounded to 1, slope
// continuous at k (1.0 on both sides), f(1)=0.9264, f(2)=0.9995, f(222.9)=1.0.
//
// Applied on LUMINANCE, not per channel. Per-channel curves compress the already-hot
// channels first, which kills saturation and walks the hue: a warm highlight
// [2.5,1.6,1.0] becomes [1.0,0.996,0.926] (sat 0.60 -> 0.07, chroma drift 0.53).
// Scaling all three channels by f(L)/L keeps hue and saturation exactly (drift 0.0000).
inline float displayKnee(float x, float k) {
    if (x <= k) return x;
    const float head = 1.0f - k;
    return k + head * (1.0f - std::exp(-(x - k) / head));
}

// 8-bit display transform on a frame already in linear AP1.
//   strength 0 -> hard clip (the old behaviour), 1 -> full knee, in between a lerp.
inline void kneeOnLuminance(float& r, float& g, float& b, float k, float strength) {
    const float L = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    if (!(L > 0.0f)) return;                               // pure black: nothing to do
    const float mapped = (1.0f - strength) * std::min(L, 1.0f) + strength * displayKnee(L, k);
    const float scale = mapped / L;
    r *= scale; g *= scale; b *= scale;
}

// Bring a linear colour above 1.0 back into [0,1] for an SDR container.
//
// Above 1.0 there is NO transparent choice - hue and brightness trade directly:
//   t = 0 : scale all channels by 1/max. Channel ratios are preserved EXACTLY, so
//           hue and saturation survive; the price is brightness: a saturated
//           highlight darkens (orange horizon luma 0.625 -> 0.357, 43% darker).
//   t = 1 : clamp each channel on its own. The brightest result, but the ratios
//           move and the hue walks toward white/yellow (18-25 deg on saturated
//           highlights - measured, knee_hue_test 2026-10-15).
// In between: a linear blend. The max channel stays exactly 1.0 along the whole
// blend, so there is no brightness step at either end.
//
// Runs AFTER ap1ToRec709: that matrix can itself push a channel above 1 (red row
// gain 1.4514), so clamping before it would not help. Colours already in range
// are exact identity for every t.
inline void gamutClamp(float& r, float& g, float& b, float t) {
    const float mx = std::max({r, g, b});
    if (!(mx > 1.0f)) return;                          // in range: nothing to do
    const float scale = 1.0f / mx;
    const float a = 1.0f - t;
    r = a * (r * scale) + t * std::min(r, 1.0f);
    g = a * (g * scale) + t * std::min(g, 1.0f);
    b = a * (b * scale) + t * std::min(b, 1.0f);
}

// Additive black pedestal removal: L' = L - bl * exp(-L / tau).
//
// The model emits a floor that does NOT reach 0 (measured: source black lands at
// 0.0037 linear ~= 15/255 after sRGB). It is also not constant -- 0.0032-0.0039 across
// frames, decaying to ~0.0007 by luma 0.07 -- so a flat subtraction would over-correct
// the mid-shadows. This is a manual control, default 0, meant to be dialled to taste.
//
//   full  (w = 1.00)  at L = 0
//   w = 0.19          at L = 0.05
//   w = 0.0025        at L = 0.18   -> midtones effectively untouched
//
// Applied on luminance and scaled back onto r,g,b for the same reason the knee is:
// subtracting per channel shifts hue. Monotone by construction -- dL'/dL = 1 + (bl/tau)e^(-L/tau)
// is always > 1 -- so pixel ordering is never reversed. L' clamps at 0, so over-dialling
// crushes to black instead of going negative.
inline void blackLevelToe(float& r, float& g, float& b, float bl) {
    if (!(bl > 0.0f)) return;                       // default: exact identity
    const float tau = 0.03f;
    const float L = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    if (!(L > 0.0f)) return;                        // pure black: nothing to remove
    const float w = std::exp(-L / tau);
    float L2 = L - bl * w;
    if (L2 < 0.0f) L2 = 0.0f;                       // over-correction clips to black
    const float scale = L2 / L;
    r *= scale; g *= scale; b *= scale;
}

    // Convert stored HDR frame (in storedOutCS space) to output color space for writing.
    // If host output is 8-bit, roll highlights into [0,1] with the display knee and
    // bring the result into gamut, then leave it LINEAR: writeFrameToImage (writeCh8,
    // render.cpp) does the ONE sRGB encode with Rec.709 primaries. Encoding here too
    // would double-encode. Float/half output never gets a tone curve.
    void convertHdrForOutput(Frame& frame, int storedOutCS, int targetOutCS, bool hostIsByte,
                             double kneeK, double kneeStrength, double gamutClip);

    // ---------------------------------------------------------------- ofx entry points
    OfxStatus ofxDescribe(OfxImageEffectHandle effect);
    OfxStatus ofxDescribeInContext(OfxImageEffectHandle effect, OfxPropertySetHandle args);
    OfxStatus ofxCreateInstance(OfxImageEffectHandle effect);
    OfxStatus ofxDestroyInstance(OfxImageEffectHandle effect);
    OfxStatus ofxRender(OfxImageEffectHandle effect, OfxPropertySetHandle args);
    OfxStatus ofxInstanceChanged(OfxImageEffectHandle effect, OfxPropertySetHandle args);
    OfxStatus ofxInvoke(OfxImageEffectHandle effect, OfxPropertySetHandle args);

    // ---------------------------------------------------------------- engine client
    // Root folder of the kit (where venv/, engine/, models/ live).
    std::string engineRoot();
    // Socket path for the engine server.
    inline std::string socketPath() { return engineRoot() + "/run/dnz_hdr.sock"; }
    // Make sure the engine server is up; returns false with *err set when it is not.
    bool ensureEngine(std::string* err);
    // One JSON object per line, exactly like the Matte engine.
    bool engineCall(const std::string& line, std::string* reply, std::string* err);
    // Job folder handling.
    std::string makeJobDir();                                  // <root>/cache/hdr_jobs/<stamp>-<rand>
    bool writeJobJson(const std::string& job, const EffectData& p, int n, int w, int h, bool preview);
    bool writeInputF16(const std::string& job, const std::vector<Frame>& frames);
    bool engineStatus(const std::string& job, std::string& state, double& progress, std::string& message);

    // Load the 22B transformer without a clip. Returns the engine's warmup folder to poll
    // with engineStatus(), or empty if the engine could not be reached.
    std::string engineWarmup(const std::string& offload, bool fp8);
    void removeJobDir(const std::string& job);

    // ---------------------------------------------------------------- shared image types
    struct ImageView {
        OfxPropertySetHandle img = nullptr;
        OfxImageClipHandle clip = nullptr;
        int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        int rowBytes = 0;
        const unsigned char* base = nullptr;
        int comps = 3;
        int depth = 0;    // 0=float, 1=byte, 2=half, 3=short
        int pxStride() const { return comps * (depth == 1 ? 1 : depth == 2 ? 2 : 4); }
        int w() const { return x2 - x1 + 1; }
        int h() const { return y2 - y1 + 1; }
        const unsigned char* row(int y) const {
            if (rowBytes >= 0) return base + static_cast<size_t>(y - y1) * rowBytes;
            return base + static_cast<size_t>(y2 - y) * (-rowBytes);
        }
    };

    // Half <-> float go through the hardware conversion, so they match numpy exactly.
    inline float fromHalfBits(uint16_t bits) {
        _Float16 h;
        std::memcpy(&h, &bits, 2);
        return static_cast<float>(h);
    }

    inline uint16_t toHalfBits(float f) {
        if (!(f > -65504.0f)) f = (f < 0.0f && f == f) ? -65504.0f : 0.0f;   // NaN -> 0
        if (f > 65504.0f) f = 65504.0f;
        _Float16 h = (_Float16)f;
        uint16_t bits;
        std::memcpy(&bits, &h, 2);
        return bits;
    }

    inline float readPix_global(const unsigned char* p, int depthCode, int c) {
        switch (depthCode) {
            case 1: {   // 8-bit: normalise to 0..1 then linearise sRGB.
                const float s = p[c] / 255.0f;
                return s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
            }
            case 2:     // 16-bit half
                return fromHalfBits(reinterpret_cast<const uint16_t*>(p)[c]);
            case 3: {   // 16-bit unsigned int (Short)
                const float s = reinterpret_cast<const uint16_t*>(p)[c] / 65535.0f;
                return s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
            }
            default:    // 32-bit float
                return reinterpret_cast<const float*>(p)[c];
        }
    }

    // ---------------------------------------------------------------- engine client functions (internal)
    bool acquire(OfxImageEffectHandle effect, const char* clipName, double time, ImageView& v);
    void release(ImageView& v);

}  // namespace dnz