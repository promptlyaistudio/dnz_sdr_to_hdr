// Effect definition (parameters, buttons) plus host image plumbing.
#include "dnz_hdr.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace dnz {

OfxGlobals g;

// fromHalfBits / toHalfBits / readPix_global are defined inline in dnz_hdr.h.

// --------------------------------------------------------------------------- math helpers
struct float3 { float x, y, z; };
inline float3 make_float3(float x, float y, float z) { return float3{x, y, z}; }
inline float3 lerp(float3 a, float3 b, float t) {
    return float3{
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t
    };
}
inline float3 clamp(float3 v, float lo, float hi) {
    return float3{
        std::max(lo, std::min(v.x, hi)),
        std::max(lo, std::min(v.y, hi)),
        std::max(lo, std::min(v.z, hi))
    };
}

// ---------------------------------------------------------------- colour conversion implementation
void convertHdrForOutput(Frame& frame, int storedOutCS, int targetOutCS, bool hostIsByte,
                         double kneeK, double kneeStrength, double gamutClip) {
    // storedOutCS: OUT_ACESCG=0, OUT_SRGB_LINEAR=1, OUT_ACESCCT=2
    // targetOutCS: same enum, what the UI currently requests
    // hostIsByte: true -> output clip is 8-bit. The byte branch leaves the frame
    //              LINEAR Rec.709; writeCh8 (render.cpp) does the single sRGB encode.
    // kneeK: knee point for the 8-bit display curve (identity below it). 0.8 is the default.
    // kneeStrength: 0 = hard clip, 1 = full knee.
    // gamutClip: how colours above 1.0 come back into gamut on the byte path.
    //            0 = scale by 1/max (hue preserved, darker), 1 = clip per channel
    //            (brighter, hue shifts). See gamutClamp in dnz_hdr.h.
    // No tone curve is ever applied to float/half output: HDR stays HDR.

    const size_t npix = static_cast<size_t>(frame.w) * frame.h;
    float* p = frame.rgb.data();

    // Step 1: Convert from stored color space to linear AP1 (ACEScg primaries)
    // If stored is ACEScg (0), already linear AP1.
    // If stored is sRGB linear (1), it's linear Rec.709 -> convert to linear AP1.
    // If stored is ACEScct (2), log -> linear AP1.
    if (storedOutCS == OUT_SRGB_LINEAR) {
        // Linear Rec.709 -> linear AP1. Exact inverse of ap1ToRec709 (verified
        // numerically: inv * fwd = I to 1e-8). The constants that were here before
        // were NOT the inverse - inv*fwd came out [1.0283, -0.0041, -0.0242; ...] -
        // but their row sums are 1, so neutrals round-tripped fine and only
        // saturated colours drifted (~2.8% on red). Found by the chain test,
        // 2026-10-15.
        // [  0.69545224, 0.14067870, 0.16386906 ]
        // [  0.04479456, 0.85967112, 0.09553432 ]
        // [ -0.00552588, 0.00402521, 1.00150067 ]
        for (size_t i = 0; i < npix; ++i) {
            float r = p[i*3+0], g = p[i*3+1], b = p[i*3+2];
            p[i*3+0] =  0.69545224f * r + 0.14067870f * g + 0.16386906f * b;
            p[i*3+1] =  0.04479456f * r + 0.85967112f * g + 0.09553432f * b;
            p[i*3+2] = -0.00552588f * r + 0.00402521f * g + 1.00150067f * b;
        }
    } else if (storedOutCS == OUT_ACESCCT) {
        // ACEScct log -> linear AP1
        for (size_t i = 0; i < npix; ++i) {
            p[i*3+0] = acescctToAcescg(p[i*3+0]);
            p[i*3+1] = acescctToAcescg(p[i*3+1]);
            p[i*3+2] = acescctToAcescg(p[i*3+2]);
        }
    }
    // else storedOutCS == OUT_ACESCG: already linear AP1, nothing to do.

    // Step 2: Convert from linear AP1 to target output space
    if (hostIsByte) {
        // 8-bit output: an SDR container cannot hold >1. Roll the HDR highlights into
        // [0,1] with the display knee (identity below the knee, so the SDR range is
        // untouched), on luminance so hue survives the knee itself, then bring the
        // result into gamut. Stay LINEAR: writeCh8 (render.cpp) does the ONE sRGB
        // encode. Encoding here as well used to double-encode everything - black
        // 3 -> 25, mid grey 0.18 -> 181 instead of 118 (measured, 2026-10-15).
        const float k = static_cast<float>(kneeK);
        const float s = static_cast<float>(kneeStrength);
        const float t = static_cast<float>(gamutClip);
        for (size_t i = 0; i < npix; ++i) {
            float r = p[i*3+0], g = p[i*3+1], b = p[i*3+2];
            kneeOnLuminance(r, g, b, k, s);
            ap1ToRec709(r, g, b);                    // AP1 -> Rec.709 primaries
            gamutClamp(r, g, b, t);                  // in-gamut for the 8-bit container
            p[i*3+0] = r; p[i*3+1] = g; p[i*3+2] = b;
        }
    } else {
        // Float/half output: convert to targetOutCS linear space
        if (targetOutCS == OUT_ACESCG) {
            // Already linear AP1, nothing to do
        } else if (targetOutCS == OUT_SRGB_LINEAR) {
            // Linear AP1 -> linear Rec.709
            for (size_t i = 0; i < npix; ++i) {
                float r = p[i*3+0], g = p[i*3+1], b = p[i*3+2];
                ap1ToRec709(r, g, b);
                p[i*3+0] = r; p[i*3+1] = g; p[i*3+2] = b;
            }
        } else if (targetOutCS == OUT_ACESCCT) {
            // Linear AP1 -> ACEScct (spec S-2016-001; see acescgToAcescct in dnz_hdr.h)
            for (size_t i = 0; i < npix; ++i) {
                p[i*3+0] = acescgToAcescct(p[i*3+0]);
                p[i*3+1] = acescgToAcescct(p[i*3+1]);
                p[i*3+2] = acescgToAcescct(p[i*3+2]);
            }
        }
    }
}

OfxStatus ofxDescribe(OfxImageEffectHandle effect) {
    OfxPropertySetHandle props;
    g.effectSuite->getPropertySet(effect, &props);
    g.propertySuite->propSetString(props, kOfxPropLabel, 0, "DNZ SDR to HDR");
    g.propertySuite->propSetString(props, kOfxImageEffectPluginPropGrouping, 0, "DNZ");
    g.propertySuite->propSetString(props, kOfxPropLongLabel, 0,
        "SDR plate to HDR (ACEScg / ACEScct / linear Rec.709 EXR + HLG) using the "
        "Lightricks LTX-2.5 SDR-to-HDR IC-LoRA, running locally. Press Convert, wait for "
        "the Status line to say done, then render. This is inferred HDR, not recovered "
        "highlight detail: grade it, do not light-match from it.");
    g.propertySuite->propSetString(props, kOfxImageEffectPropSupportedContexts, 0,
                                   kOfxImageEffectContextFilter);
    g.propertySuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 0,
                                   kOfxImageComponentRGBA);
    g.propertySuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 1,
                                   kOfxImageComponentRGB);
    // Advertise every bit depth the host may hand us. Without this the host has no idea
    // what we accept and refuses plates ("unsupported input ... 8 bit rgb"). The render
    // path normalises all of these to float internally.
    g.propertySuite->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 0,
                                   kOfxBitDepthFloat);
    g.propertySuite->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 1,
                                   kOfxBitDepthHalf);
    g.propertySuite->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 2,
                                   kOfxBitDepthByte);
    g.propertySuite->propSetInt(props, kOfxImageEffectPropSupportsTiles, 0, 0);
    g.propertySuite->propSetInt(props, kOfxImageEffectPluginPropHostFrameThreading, 0, 0);
    g.propertySuite->propSetInt(props, kOfxImageEffectPropSupportsMultiResolution, 0, 0);
    return kOfxStatOK;
}

OfxStatus ofxDescribeInContext(OfxImageEffectHandle effect, OfxPropertySetHandle /*args*/) {
    OfxPropertySetHandle srcClip, dstClip;
    g.effectSuite->clipDefine(effect, kOfxImageEffectSimpleSourceClipName, &srcClip);
    g.propertySuite->propSetString(srcClip, kOfxImageEffectPropSupportedComponents, 0,
                                   kOfxImageComponentRGBA);
    g.propertySuite->propSetString(srcClip, kOfxImageEffectPropSupportedComponents, 1,
                                   kOfxImageComponentRGB);
    g.propertySuite->propSetString(srcClip, kOfxImageEffectPropSupportedPixelDepths, 0,
                                   kOfxBitDepthFloat);
    g.propertySuite->propSetString(srcClip, kOfxImageEffectPropSupportedPixelDepths, 1,
                                   kOfxBitDepthHalf);
    g.propertySuite->propSetString(srcClip, kOfxImageEffectPropSupportedPixelDepths, 2,
                                   kOfxBitDepthByte);
    g.propertySuite->propSetInt(srcClip, kOfxImageEffectPropTemporalClipAccess, 0, 1);

    g.effectSuite->clipDefine(effect, kOfxImageEffectOutputClipName, &dstClip);
    g.propertySuite->propSetString(dstClip, kOfxImageEffectPropSupportedComponents, 0,
                                   kOfxImageComponentRGBA);
    g.propertySuite->propSetString(dstClip, kOfxImageEffectPropSupportedComponents, 1,
                                   kOfxImageComponentRGB);
    g.propertySuite->propSetString(dstClip, kOfxImageEffectPropSupportedPixelDepths, 0,
                                   kOfxBitDepthFloat);
    g.propertySuite->propSetString(dstClip, kOfxImageEffectPropSupportedPixelDepths, 1,
                                   kOfxBitDepthHalf);
    g.propertySuite->propSetString(dstClip, kOfxImageEffectPropSupportedPixelDepths, 2,
                                   kOfxBitDepthByte);
    g.propertySuite->propSetInt(srcClip, kOfxImageEffectPropTemporalClipAccess, 0, 1);

    g.effectSuite->clipDefine(effect, kOfxImageEffectOutputClipName, &dstClip);
    g.propertySuite->propSetString(dstClip, kOfxImageEffectPropSupportedComponents, 0,
                                   kOfxImageComponentRGBA);
    g.propertySuite->propSetString(dstClip, kOfxImageEffectPropSupportedComponents, 1,
                                   kOfxImageComponentRGB);
    g.propertySuite->propSetString(dstClip, kOfxImageEffectPropSupportedPixelDepths, 0,
                                   kOfxBitDepthFloat);
    g.propertySuite->propSetString(dstClip, kOfxImageEffectPropSupportedPixelDepths, 1,
                                   kOfxBitDepthHalf);
    g.propertySuite->propSetString(dstClip, kOfxImageEffectPropSupportedPixelDepths, 2,
                                   kOfxBitDepthByte);

    OfxParamSetHandle ps;
    g.effectSuite->getParamSet(effect, &ps);
    OfxPropertySetHandle pp;

    // Everything after a group() call lands inside that group in the node UI.
    std::string curGroup;
    auto inGroup = [&](OfxPropertySetHandle p) {
        if (!curGroup.empty())
            g.propertySuite->propSetString(p, kOfxParamPropParent, 0, curGroup.c_str());
    };
    auto group = [&](const char* n, const char* label) {
        g.paramSuite->paramDefine(ps, kOfxParamTypeGroup, n, &pp);
        g.propertySuite->propSetString(pp, kOfxPropLabel, 0, label);
        curGroup = n;
    };

    auto choice = [&](const char* n, const char* label, const char* hint, int dflt,
                      std::initializer_list<const char*> opts) {
        g.paramSuite->paramDefine(ps, kOfxParamTypeChoice, n, &pp);
        inGroup(pp);
        g.propertySuite->propSetString(pp, kOfxPropLabel, 0, label);
        g.propertySuite->propSetString(pp, kOfxParamPropHint, 0, hint);
        g.propertySuite->propSetInt(pp, kOfxParamPropDefault, 0, dflt);
        int i = 0;
        for (const char* o : opts)
            g.propertySuite->propSetString(pp, kOfxParamPropChoiceOption, i++, o);
    };
    auto ddef = [&](const char* n, const char* label, const char* hint, double d, double mn, double mx) {
        g.paramSuite->paramDefine(ps, kOfxParamTypeDouble, n, &pp);
        inGroup(pp);
        g.propertySuite->propSetString(pp, kOfxPropLabel, 0, label);
        g.propertySuite->propSetString(pp, kOfxParamPropHint, 0, hint);
        g.propertySuite->propSetDouble(pp, kOfxParamPropDefault, 0, d);
        g.propertySuite->propSetDouble(pp, kOfxParamPropMin, 0, mn);
        g.propertySuite->propSetDouble(pp, kOfxParamPropMax, 0, mx);
        g.propertySuite->propSetDouble(pp, kOfxParamPropDisplayMin, 0, mn);
        g.propertySuite->propSetDouble(pp, kOfxParamPropDisplayMax, 0, mx);
    };
    auto idef = [&](const char* n, const char* label, const char* hint, int d, int mn, int mx) {
        g.paramSuite->paramDefine(ps, kOfxParamTypeInteger, n, &pp);
        inGroup(pp);
        g.propertySuite->propSetString(pp, kOfxPropLabel, 0, label);
        g.propertySuite->propSetString(pp, kOfxParamPropHint, 0, hint);
        g.propertySuite->propSetInt(pp, kOfxParamPropDefault, 0, d);
        g.propertySuite->propSetInt(pp, kOfxParamPropMin, 0, mn);
        g.propertySuite->propSetInt(pp, kOfxParamPropMax, 0, mx);
    };
    auto bdef = [&](const char* n, const char* label, const char* hint) {
        g.paramSuite->paramDefine(ps, kOfxParamTypePushButton, n, &pp);
        inGroup(pp);
        g.propertySuite->propSetString(pp, kOfxPropLabel, 0, label);
        g.propertySuite->propSetString(pp, kOfxParamPropHint, 0, hint);
    };

    // ---- Analysis (same shape as the Matte node: run it, store it, read it back)
    group("analysisGroup", "Analysis");
    bdef("convertBtn", "Analyze",
         "Convert the whole clip to HDR and store it with this node. Minutes, not seconds: "
         "the 22B model loads on the first run.");
    bdef("previewBtn", "Preview (fast)",
         "Same at Preview Res - about a minute for 40 frames. Check the look before a full Analyze.");
    bdef("refreshBtn", "Refresh Status", "Redraw the viewer with whatever is ready and update Status.");
    bdef("warmupBtn", "Warm Up (load model)",
         "Load the 22B model without a clip, so the next Analyze skips the 10-20 min load. "
         "Run it once per Flame session, whenever suits you.");
    bdef("clearBtn", "Clear Analysis", "Cancel a running job / forget the stored HDR clip.");
    idef("rangeStart", "Range Start", "First frame to convert. 0 = clip start.", 0, 0, 4000);
    idef("rangeEnd", "Range End", "Last frame to convert. 0 = clip end.", 0, 0, 4000);
    idef("workRes", "Work Res",
         "Long side the clip is analyzed at (LTX max 1920). A UHD plate is analyzed at "
         "1920x1088 and scaled back up.", 1920, 256, 4096);
    idef("previewRes", "Preview Res", "Resolution used by Preview (fast).", 768, 256, 4096);
    g.paramSuite->paramDefine(ps, kOfxParamTypeString, "status", &pp);
    inGroup(pp);
    g.propertySuite->propSetString(pp, kOfxPropLabel, 0, "Status");
    g.propertySuite->propSetString(pp, kOfxParamPropHint, 0, "Engine state for this node.");
    g.propertySuite->propSetString(pp, kOfxParamPropStringMode, 0, kOfxParamStringIsSingleLine);
    g.paramSuite->paramDefine(ps, kOfxParamTypeString, "jobFolder", &pp);
    inGroup(pp);
    g.propertySuite->propSetString(pp, kOfxPropLabel, 0, "Job Folder");
    g.propertySuite->propSetString(pp, kOfxParamPropStringMode, 0, kOfxParamStringIsSingleLine);

    // ---- Input
    group("inputGroup", "Input");
    choice("inCS", "Input Color Space",
           "Colour space of the incoming plate. ACEScg = scene-linear EXR from an ACES Flame. "
           "Display sRGB = video/PNG with gamma baked in. Wrong choice = wrong HDR; it is never guessed.",
           0, {"sRGB linear", "ACEScg", "ACEScct (log)", "sRGB gamma (video/PNG)"});
    idef("seed", "Seed", "Same plate + same seed = same HDR.", 42, 0, 999999);
    ddef("frameRate", "Frame Rate", "Frames per second of the clip (the EXR pipeline needs it).",
         24.0, 1.0, 120.0);

    // ---- Inference
    group("infGroup", "Inference");
    g.paramSuite->paramDefine(ps, kOfxParamTypeBoolean, "highQuality", &pp);
    inGroup(pp);
    g.propertySuite->propSetString(pp, kOfxPropLabel, 0, "High Quality");
    g.propertySuite->propSetString(pp, kOfxParamPropHint, 0,
        "2x internal frames, about 2x longer (official pipeline option).");
    g.paramSuite->paramDefine(ps, kOfxParamTypeBoolean, "keyframes", &pp);
    inGroup(pp);
    g.propertySuite->propSetString(pp, kOfxPropLabel, 0, "Seam Keyframes");
    g.propertySuite->propSetString(pp, kOfxParamPropHint, 0,
        "Official DFR-style seam guides between 8-frame segments. Turn off if seams show.");
    g.propertySuite->propSetInt(pp, kOfxParamPropDefault, 0, 1);
    g.paramSuite->paramDefine(ps, kOfxParamTypeBoolean, "fp8", &pp);
    inGroup(pp);
    g.propertySuite->propSetString(pp, kOfxPropLabel, 0, "FP8");
    g.propertySuite->propSetString(pp, kOfxParamPropHint, 0,
        "Keep the pipeline's FP8 cast. Off by default: Apple GPU has no FP8.");
    choice("offload", "Offload", "Where the transformer lives when it does not fit.",
           0, {"No offload", "CPU offload", "Disk offload"});

    // ---- Tiling (VAE decode tile size & overlap in pixels/frames)
    group("tilingGroup", "Tiling (VAE Decode)");
    idef("tileFrames", "Tile Frames", "Temporal tile size in frames. 0 = auto (default 80).",
         0, 0, 240);
    idef("tileHeight", "Tile Height", "Spatial tile height in pixels. 0 = auto (default 768).",
         0, 0, 4096);
    idef("tileWidth", "Tile Width", "Spatial tile width in pixels. 0 = auto (default 768).",
         0, 0, 4096);
    idef("overlapFrames", "Overlap Frames", "Temporal overlap between tiles in frames. 0 = auto (default 24).",
         0, 0, 60);
    idef("overlapHeight", "Overlap Height", "Vertical overlap between tiles in pixels. 0 = auto (default 64).",
         0, 0, 512);
    idef("overlapWidth", "Overlap Width", "Horizontal overlap between tiles in pixels. 0 = auto (default 64).",
         0, 0, 512);

    // ---- Output
    group("outGroup", "Output");
    choice("outCS", "Output Color Space",
           "Colour space of the HDR frames this node writes. sRGB linear (default) keeps the "
           "plate's own colour: linear light, Rec.709 primaries - the same space as the source "
           "after decoding, so the HDR clip is NOT log and NOT re-primaried; it drops next to "
           "the SDR plate and matches below the highlights. ACEScg only for an ACES batch; "
           "ACEScct only if you explicitly want log.", 1, {"ACEScg", "sRGB linear", "ACEScct (log)"});
    choice("outputMode", "Output Mode",
           "HDR = the converted clip. Original SDR = the plate untouched (for comparison or "
           "bypass). Side by Side = SDR on the left half, HDR on the right half.",
           0, {"HDR", "Original SDR", "Side by Side"});

    // ---- HDR Adjust (live, applied to the stored clip: no re-analysis)
    group("adjGroup", "HDR Adjust");
    ddef("exposure", "Exposure", "Stops, applied to the HDR clip before it leaves the node. "
         "Keyframeable.", 0.0, -8.0, 8.0);
    ddef("highlightGain", "Highlight Gain", "Gain applied only above 1.0 (the HDR part). "
         "1.0 = untouched.", 1.0, 0.0, 8.0);
    ddef("blackLevel", "Black Level", "Removes the additive black pedestal the model leaves "
         "in the shadows (source black lands around 0.0037 linear instead of 0). Subtracts "
         "this much from luminance with a toe that fades as brightness rises, so only the "
         "deepest shadows move: full at black, 19% at luma 0.05, 0.25% at 0.18 - midtones are "
         "effectively untouched. 0 = untouched. The floor is not constant across frames, so "
         "dial this by eye rather than to a number. Applied on luminance, not per channel, "
         "so hue and saturation are preserved. Too much crushes the blacks.",
         0.0, 0.0, 0.02);
    ddef("kneeK", "Display Knee", "Knee point for the 8-bit display curve, in linear light. "
         "Below this value the curve is the exact identity, so the SDR range is untouched "
         "and the SDR plate next to its HDR render still matches. Above it the curve rolls "
         "smoothly to 1.0 (highlight at 1.0 -> 0.926, at 2.0 -> 0.9995). Applied on "
         "luminance, not per channel, so hue and saturation are preserved.",
         0.8, 0.1, 1.0);
    ddef("kneeStrength", "Knee Strength", "0 = hard clip at 1.0 (every highlight above 1 is "
         "cut), 1 = full knee roll-off. In between the two curves are mixed linearly. "
         "Only affects 8-bit output; float/HDR output is never toned.",
         1.0, 0.0, 1.0);
    ddef("gamutClip", "Gamut Clip", "How highlights above 1.0 are brought back into the 8-bit "
         "container. 0 = keep hue: all channels are scaled by 1/max, so the colour of a "
         "saturated highlight survives exactly, at the cost of brightness (an orange "
         "sunset highlight darkens by up to ~43%). 1 = clip per channel: the brightest "
         "result, but the hue walks toward white/yellow (18-25 degrees on saturated "
         "colours). In between the two are blended. Only affects 8-bit output.",
         0.0, 0.0, 1.0);
    g.paramSuite->paramDefine(ps, kOfxParamTypeBoolean, "clipPreview", &pp);
    inGroup(pp);
    g.propertySuite->propSetString(pp, kOfxPropLabel, 0, "Clip Preview");
    g.propertySuite->propSetString(pp, kOfxParamPropHint, 0,
        "Clamp the result to 0..1 so you can see exactly what clips.");

    return kOfxStatOK;
}

OfxStatus ofxCreateInstance(OfxImageEffectHandle effect) {
    OfxPropertySetHandle props;
    g.effectSuite->getPropertySet(effect, &props);
    g.propertySuite->propSetPointer(props, kOfxPropInstanceData, 0, new EffectData());
    return kOfxStatOK;
}

OfxStatus ofxDestroyInstance(OfxImageEffectHandle effect) {
    OfxPropertySetHandle props;
    g.effectSuite->getPropertySet(effect, &props);
    void* raw = nullptr;
    g.propertySuite->propGetPointer(props, kOfxPropInstanceData, 0, &raw);
    delete static_cast<EffectData*>(raw);
    g.propertySuite->propSetPointer(props, kOfxPropInstanceData, 0, nullptr);
    return kOfxStatOK;
}

}  // namespace dnz
