// Render + button handling: gather the clip, hand it to the engine, read the HDR
// clip back. Nothing here loads a model; the worst case is a slow pass-through.
#include "dnz_hdr.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace dnz {

JobState& jobState() {
    static JobState s;
    return s;
}

namespace {

// Live HDR grade: black-pedestal removal, exposure in stops, highlight-only gain, and an
// optional clamp so the user can see what clips. None of this re-runs the model.
//
// Order matters. The pedestal is an additive error sitting in the source, so it comes off
// FIRST, before exposure multiplies it back up. Exposure then scales linear light, and the
// gain is confined to the part already above 1.0.
void adjustHdr(Frame& f, double exposure, double gain, bool clipPreview, double blackLevel) {
    if (blackLevel > 0.0) {
        const float bl = static_cast<float>(blackLevel);
        for (size_t i = 0; i + 2 < f.rgb.size(); i += 3)     // luminance-based, per pixel
            blackLevelToe(f.rgb[i], f.rgb[i + 1], f.rgb[i + 2], bl);
    }
    const double stops = std::pow(2.0, exposure);
    for (float& v : f.rgb) {
        double x = static_cast<double>(v) * stops;
        if (x > 1.0) x = 1.0 + (x - 1.0) * gain;           // gain touches only the HDR part
        if (clipPreview) x = std::max(0.0, std::min(1.0, x));
        v = static_cast<float>(x);
    }
}

// Validate job metadata (hdr_meta.json) against expected outCS.
// Called once when job completes. Returns true if valid or no metadata file.
bool validateJobMetadata(const std::string& folder, int expectedOutCS) {
    std::string metaPath = folder + "/hdr_meta.json";
    if (FILE* mf = std::fopen(metaPath.c_str(), "r")) {
        char buf[1024];
        size_t len = std::fread(buf, 1, sizeof(buf) - 1, mf);
        std::fclose(mf);
        if (len > 0) {
            buf[len] = '\0';
            const char* key = "\"out_cs\"";
            char* p = std::strstr(buf, key);
            if (p) {
                p = std::strchr(p + std::strlen(key), '"');
                if (p) {
                    char* end = std::strchr(p + 1, '"');
                    if (end) {
                        std::string metaOutCS(p + 1, end - p - 1);
                        int metaOutCSEnum = -1;
                        if (metaOutCS == "acescg") metaOutCSEnum = OUT_ACESCG;
                        else if (metaOutCS == "srgb_linear") metaOutCSEnum = OUT_SRGB_LINEAR;
                        else if (metaOutCS == "acescct") metaOutCSEnum = OUT_ACESCCT;
                        if (expectedOutCS >= 0 && metaOutCSEnum >= 0 && metaOutCSEnum != expectedOutCS) {
                            return false; // mismatch
                        }
                    }
                }
            }
        }
    }
    return true; // valid or no metadata
}

void readParams(OfxParamSetHandle ps, EffectData& p, double t) {
    p.inCS = getInt(ps, "inCS", t);
    p.outCS = getInt(ps, "outCS", t);
    p.fps = getDouble(ps, "frameRate", t);
    p.seed = getInt(ps, "seed", t);
    p.workRes = getInt(ps, "workRes", t);
    p.previewRes = getInt(ps, "previewRes", t);
    p.highQuality = getBool(ps, "highQuality", t);
    p.keyframes = getBool(ps, "keyframes", t);
    p.fp8 = getBool(ps, "fp8", t);
    p.offload = getInt(ps, "offload", t);
    p.rangeStart = getInt(ps, "rangeStart", t);
    p.rangeEnd = getInt(ps, "rangeEnd", t);
    p.outputMode = getInt(ps, "outputMode", t);
    p.exposure = getDouble(ps, "exposure", t);
    p.highlightGain = getDouble(ps, "highlightGain", t);
    p.blackLevel = getDouble(ps, "blackLevel", t);
    p.clipPreview = getBool(ps, "clipPreview", t);
    p.tileFrames = getInt(ps, "tileFrames", t);
    p.tileHeight = getInt(ps, "tileHeight", t);
    p.tileWidth = getInt(ps, "tileWidth", t);
    p.overlapFrames = getInt(ps, "overlapFrames", t);
    p.overlapHeight = getInt(ps, "overlapHeight", t);
    p.overlapWidth = getInt(ps, "overlapWidth", t);
    p.kneeK = getDouble(ps, "kneeK", t);
    p.kneeStrength = getDouble(ps, "kneeStrength", t);
    p.gamutClip = getDouble(ps, "gamutClip", t);
}

void setStatus(OfxParamSetHandle ps, const std::string& s) { setStringParam(ps, "status", s); }

// Ask the host which frames this effect can produce. Several sources, most reliable first:
//  1. kOfxImageEffectPropFrameRange on the effect props (present during a render action).
//  2. The kOfxImageEffectActionGetFrameRange action, which the host answers on outArgs.
//  3. OfxImageClipPropFrameRange_<clipName> on the Source clip's own props.
// Returns false only when the host genuinely reports nothing from any source.
bool clipRange(OfxImageEffectHandle effect, int& first, int& last) {
    OfxPropertySetHandle props;
    double f0 = 0, f1 = 0;

    // 1. effect prop set
    g.effectSuite->getPropertySet(effect, &props);
    if (g.propertySuite->propGetDouble(props, kOfxImageEffectPropFrameRange, 0, &f0) == kOfxStatOK &&
        g.propertySuite->propGetDouble(props, kOfxImageEffectPropFrameRange, 1, &f1) == kOfxStatOK) {
        first = static_cast<int>(std::lround(f0));
        last = static_cast<int>(std::lround(f1));
        return last >= first;
    }

    // 2. getFrameRange action: OFX has the host call this action on us and pass the range
    //    in outArgs; a plugin cannot invoke it itself, so there is nothing to call here.
    //    The reliable in-plugin source is the clip prop (3) below.

    // 3. Source clip's own frame-range prop.
    OfxImageClipHandle srcClip = nullptr;
    if (g.effectSuite->clipGetHandle(effect, kOfxImageEffectSimpleSourceClipName, &srcClip, nullptr) == kOfxStatOK) {
        OfxPropertySetHandle clipProps = nullptr;
        if (g.effectSuite->clipGetPropertySet(srcClip, &clipProps) == kOfxStatOK && clipProps) {
            if (g.propertySuite->propGetDouble(clipProps, "OfxImageClipPropFrameRange_Source", 0, &f0) == kOfxStatOK &&
                g.propertySuite->propGetDouble(clipProps, "OfxImageClipPropFrameRange_Source", 1, &f1) == kOfxStatOK) {
                first = static_cast<int>(std::lround(f0));
                last = static_cast<int>(std::lround(f1));
                return last >= first;
            }
        }
    }
    return false;
}

// Pull the whole clip at work resolution and queue it with the engine.
void startJob(OfxImageEffectHandle effect, OfxParamSetHandle ps, bool preview) {
    EffectData p;
    readParams(ps, p, 0.0);

    int first = 0, last = 0;
    bool haveRange = clipRange(effect, first, last);
    if (!haveRange) {
        // The host did not report a frame range (common when Analyze is pressed outside a
        // render action). Do not fail: start at Range Start (0 = clip start) and, unless
        // the node pins an end, read frames until the plate runs dry in the fetch loop.
        first = (p.rangeStart > 0) ? p.rangeStart : 0;
        last = (p.rangeEnd > 0) ? p.rangeEnd : -1;      // -1 = unknown end, read to end
    } else {
        if (p.rangeStart > first) first = p.rangeStart;
        if (p.rangeEnd >= first && p.rangeEnd < last) last = p.rangeEnd;
    }
    int n = (last >= 0) ? (last - first + 1) : 0;       // 0 = discover by reading
    if (last >= 0 && n < 1) {
        setStatus(ps, "ERROR: empty frame range");
        return;
    }

    std::string err;
    setStatus(ps, "starting the engine...");
    if (!ensureEngine(&err)) {
        setStatus(ps, "ERROR: " + err);
        return;
    }

    // one conversion at a time, and only one thread may gather frames.
    // The flag is test-and-set under the mutex and NOTHING else happens under it.
    // A function-scope guard here used to (a) hold the lock across fetchFrame /
    // setStatus / file writes and (b) NEST with the scoped guards further down -
    // and std::mutex is not recursive, so every successful Analyze deadlocked the
    // thread on itself and froze Flame from the button press on (found in a live
    // Flame sample, 2026-10-15).
    JobState& js = jobState();
    bool alreadyRunning = false;
    {
        std::lock_guard<std::mutex> lk(js.mutex);
        alreadyRunning = js.running;
        if (!alreadyRunning) {
            js.running = true;
            js.metaValidated = false;
            js.metaMismatch = false;
        }
    }
    if (alreadyRunning) {
        setStatus(ps, "a job is already running (press Clear Result to cancel)");
        return;
    }

    setStatus(ps, "gathering frames...");
    std::vector<Frame> frames;
    int nw = 0, nh = 0;
    const bool unknownEnd = (last < 0);
    for (int f = first; unknownEnd || f <= last; ++f) {
        Frame fr;
        if (!fetchFrame(effect, kOfxImageEffectSimpleSourceClipName, static_cast<double>(f), fr)) {
            if (unknownEnd && f > first) {
                break;                     // ran past the end of the plate: stop cleanly
            }
            { std::lock_guard<std::mutex> lk(js.mutex); js.running = false; }
            setStatus(ps, "ERROR: could not read the plate at frame " + std::to_string(f));
            return;
        }
        if (nw == 0) {
            nw = fr.w;
            nh = fr.h;
        }
        frames.push_back(std::move(fr));
        if (unknownEnd && static_cast<int>(frames.size()) >= 100000) {
            break;                          // safety stop, do not loop forever
        }
    }
    n = static_cast<int>(frames.size());

    const int longSide = preview ? p.previewRes : p.workRes;
    int ww = 0, wh = 0;
    workSizeFor(nw, nh, longSide, ww, wh);
    if (ww != nw || wh != nh) {
        for (auto& fr : frames) fr = resized(fr, ww, wh);
    }

    setStatus(ps, "writing the job...");
    std::string job = makeJobDir();
    if (!writeJobJson(job, p, n, ww, wh, preview) || !writeInputF16(job, frames)) {
        { std::lock_guard<std::mutex> lk(js.mutex); js.running = false; }
        setStatus(ps, "ERROR: could not write the job folder (disk full?)");
        return;
    }

    std::ostringstream req;
    req << "{\"cmd\":\"convert\",\"job\":\"" << job << "\"}";
    std::string reply, cerr;
    if (!engineCall(req.str(), &reply, &cerr)) {
        { std::lock_guard<std::mutex> lk(js.mutex); js.running = false; }
        setStatus(ps, "ERROR: " + cerr);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(js.mutex);
        js.folder = job;
        js.haveClip = false;
        js.n = n;
        js.w = ww;
        js.h = wh;
        js.inCS = p.inCS;
        js.outCS = p.outCS;
        js.fps = p.fps;
        js.outputMode = p.outputMode;
        js.exposure = p.exposure;
        js.highlightGain = p.highlightGain;
        js.blackLevel = p.blackLevel;
        js.clipPreview = p.clipPreview;
        js.kneeK = p.kneeK;
        js.kneeStrength = p.kneeStrength;
        js.gamutClip = p.gamutClip;
    }
    setStringParam(ps, "jobFolder", job);
    setStatus(ps, (preview ? "PREVIEW queued" : "queued") + std::string(" (") + std::to_string(n) +
                      " frames at " + std::to_string(ww) + "x" + std::to_string(wh) +
                      "). Press Refresh Status.");
}

}  // namespace

OfxStatus ofxInstanceChanged(OfxImageEffectHandle effect, OfxPropertySetHandle args) {
    char* name = nullptr;
    if (g.propertySuite->propGetString(args, kOfxPropName, 0, &name) != kOfxStatOK || !name)
        return kOfxStatReplyDefault;
    OfxParamSetHandle ps;
    g.effectSuite->getParamSet(effect, &ps);
    const std::string n(name);

    if (n == "convertBtn") {
        startJob(effect, ps, false);
        return kOfxStatOK;
    }
    if (n == "previewBtn") {
        startJob(effect, ps, true);
        return kOfxStatOK;
    }
    if (n == "refreshBtn") {
        JobState& js = jobState();
        std::string folder;
        {
            std::lock_guard<std::mutex> lk(js.mutex);
            folder = js.folder;
        }
        if (folder.empty()) {
            setStatus(ps, "nothing converted yet - press Convert");
        } else {
            std::string state, msg;
            double prog = 0;
            engineStatus(folder, state, prog, msg);      // socket: outside the lock
            if (state == "done") {
                // A guard nested inside the outer one used to live here and
                // deadlocked the thread on itself the moment a job finished.
                std::string warn;
                bool needValidate = false;
                int expectedOutCS = OUT_ACESCG;
                {
                    std::lock_guard<std::mutex> lk(js.mutex);
                    js.haveClip = true;
                    js.running = false;
                    needValidate = !js.metaValidated;
                    expectedOutCS = js.outCS;
                }
                if (needValidate) {
                    bool mismatch = !validateJobMetadata(folder, expectedOutCS);  // file read: outside
                    {
                        std::lock_guard<std::mutex> lk(js.mutex);
                        js.metaValidated = true;
                        js.metaMismatch = mismatch;
                    }
                    if (mismatch) warn = " [WARNING: metadata color space mismatch]";
                }
                msg += warn;
            }
            setStatus(ps, state + " " + std::to_string(static_cast<int>(prog * 100)) + "% - " + msg);
        }
        return kOfxStatOK;
    }
    if (n == "warmupBtn") {
        const int off_i = getInt(ps, "offload", 0);
        const char* offload = (off_i == 1) ? "cpu" : (off_i == 2) ? "disk" : "none";
        const bool fp8 = getInt(ps, "fp8", 0) != 0;
        std::string folder = engineWarmup(offload, fp8);
        JobState& js = jobState();
        if (folder.empty()) {
            setStatus(ps, "warmup: could not reach the engine");
            return kOfxStatOK;
        }
        {
            std::lock_guard<std::mutex> lk(js.mutex);
            js.folder = folder;                 // Refresh Status now polls the warmup job
            js.haveClip = false;
            js.metaValidated = false;
            js.metaMismatch = false;
        }
        std::string state, msg;
        double prog = 0;
        engineStatus(folder, state, prog, msg);              // socket: outside the lock
        setStatus(ps, "warmup: " + msg);                      // host call: outside
        return kOfxStatOK;
    }
    if (n == "clearBtn") {
        JobState& js = jobState();
        std::string folder;
        {
            std::lock_guard<std::mutex> lk(js.mutex);
            folder = js.folder;
            js.folder.clear();
            js.haveClip = false;
            js.running = false;
            js.metaValidated = false;
            js.metaMismatch = false;
        }
        if (!folder.empty()) removeJobDir(folder);           // file deletion: outside the lock
        setStringParam(ps, "jobFolder", "");
        setStatus(ps, "cleared");
        return kOfxStatOK;
    }
    return kOfxStatReplyDefault;
}

// A few hosts report a push button through the (newer) Invoke action instead.
OfxStatus ofxInvoke(OfxImageEffectHandle effect, OfxPropertySetHandle args) {
    return ofxInstanceChanged(effect, args);
}

OfxStatus ofxRender(OfxImageEffectHandle effect, OfxPropertySetHandle args) {
    double time = 0;
    g.propertySuite->propGetDouble(args, kOfxPropTime, 0, &time);
    const int frame = static_cast<int>(std::lround(time));

    OfxParamSetHandle ps;
    g.effectSuite->getParamSet(effect, &ps);

    // always fetch the plate: it is the pass-through and it carries our alpha
    Frame src;
    if (!fetchFrame(effect, kOfxImageEffectSimpleSourceClipName, time, src))
        return kOfxStatFailed;

    JobState& js = jobState();

    // The HDR Adjust group is advertised as a LIVE grade -- "applied to the stored clip,
    // no re-analysis" -- but readParams() is otherwise only called from startJob(), so the
    // grade would stay frozen at whatever the sliders said when Convert was pressed and a
    // slider move would silently do nothing. Re-read just the live fields here, on every
    // render. The job-defining fields (colour space, resolution, offload, ...) are NOT
    // copied: changing those still requires a re-conversion.
    {
        EffectData live;
        readParams(ps, live, time);
        std::lock_guard<std::mutex> lk(js.mutex);
        js.outputMode    = live.outputMode;
        js.exposure      = live.exposure;
        js.highlightGain = live.highlightGain;
        js.blackLevel    = live.blackLevel;
        js.clipPreview   = live.clipPreview;
        js.kneeK         = live.kneeK;
        js.kneeStrength  = live.kneeStrength;
        js.gamutClip     = live.gamutClip;
    }

    std::string folder;
    bool have = false;
    int n = 0, w = 0, h = 0;
    {
        std::lock_guard<std::mutex> lk(js.mutex);
        folder = js.folder;
        have = js.haveClip;
        n = js.n;
        w = js.w;
        h = js.h;
    }

    Frame hdr;
    bool haveHdr = false;
    if (have && !folder.empty() && frame >= 0 && frame < n) {
        Frame raw;
        if (readHdrFrame(folder, n, w, h, frame, raw, js.outCS)) {
            hdr = (w == src.w && h == src.h) ? raw : resized(raw, src.w, src.h);
            haveHdr = true;
        }
    }

    int mode = MODE_HDR;
    double exposure = 0.0, gain = 1.0, blackLevel = 0.0;
    int clipPrev = 0;
    {
        std::lock_guard<std::mutex> lk(js.mutex);
        mode = js.outputMode;
        exposure = js.exposure;
        gain = js.highlightGain;
        blackLevel = js.blackLevel;
        clipPrev = js.clipPreview;
    }

    if (!haveHdr) {
        writeFrameToImage(effect, time, src);              // nothing stored: pass the plate through
        return kOfxStatOK;
    }
    adjustHdr(hdr, exposure, gain, clipPrev != 0, blackLevel);

    // Query host output clip depth (8-bit / half / float) to know target encoding.
    OfxImageClipHandle outClip = nullptr;
    bool hostIsByte = false;
    if (g.effectSuite->clipGetHandle(effect, kOfxImageEffectOutputClipName, &outClip, nullptr) == kOfxStatOK) {
        OfxPropertySetHandle imgProps = nullptr;
        if (g.effectSuite->clipGetPropertySet(outClip, &imgProps) == kOfxStatOK && imgProps) {
            char* depth = nullptr;
            if (g.propertySuite->propGetString(imgProps, kOfxImageEffectPropPixelDepth, 0, &depth) == kOfxStatOK && depth) {
                hostIsByte = (std::strcmp(depth, kOfxBitDepthByte) == 0);
            }
        }
    }

    // Get current UI output color space (target) and stored job output color space.
    int targetOutCS = getInt(ps, "outCS", time);
    int storedOutCS = js.outCS;

    Frame out;
    if (mode == MODE_SDR) {
        out = src;                                          // the untouched plate, for comparison
    } else if (mode == MODE_SIDE_BY_SIDE) {
        out = src;                                          // left half SDR ...
        // Right half: HDR converted to output space
        Frame hdrConv = hdr;
        convertHdrForOutput(hdrConv, storedOutCS, targetOutCS, hostIsByte, js.kneeK, js.kneeStrength, js.gamutClip);
        const int half = src.w / 2;
        for (int y = 0; y < src.h; ++y)
            for (int x = half; x < src.w; ++x) {
                const float* q = hdrConv.rgb.data() + (static_cast<size_t>(y) * src.w + x) * 3;
                float* d = out.rgb.data() + (static_cast<size_t>(y) * src.w + x) * 3;
                d[0] = q[0];
                d[1] = q[1];
                d[2] = q[2];
            }
    } else {
        // HDR mode: convert stored clip to output space
        out = hdr;
        convertHdrForOutput(out, storedOutCS, targetOutCS, hostIsByte, js.kneeK, js.kneeStrength, js.gamutClip);
    }
    writeFrameToImage(effect, time, out);
    return kOfxStatOK;
}

// --------------------------------------------------------------------------- parameter helpers
double getDouble(OfxParamSetHandle ps, const char* n, double t) {
    OfxParamHandle p;
    double v = 0;
    if (g.paramSuite->paramGetHandle(ps, n, &p, 0) == kOfxStatOK)
        g.paramSuite->paramGetValueAtTime(p, t, &v);
    return v;
}

int getInt(OfxParamSetHandle ps, const char* n, double t) {
    OfxParamHandle p;
    int v = 0;
    if (g.paramSuite->paramGetHandle(ps, n, &p, 0) == kOfxStatOK)
        g.paramSuite->paramGetValueAtTime(p, t, &v);
    return v;
}

int getBool(OfxParamSetHandle ps, const char* n, double t) {
    OfxParamHandle p;
    int v = 0;
    if (g.paramSuite->paramGetHandle(ps, n, &p, 0) == kOfxStatOK)
        g.paramSuite->paramGetValueAtTime(p, t, &v);
    return v;
}

std::string getString(OfxParamSetHandle ps, const char* n) {
    OfxParamHandle p;
    char* v = nullptr;
    if (g.paramSuite->paramGetHandle(ps, n, &p, 0) == kOfxStatOK)
        g.paramSuite->paramGetValue(p, &v);
    return v ? std::string(v) : std::string();
}

void setStringParam(OfxParamSetHandle ps, const char* n, const std::string& v) {
    OfxParamHandle p;
    if (g.paramSuite->paramGetHandle(ps, n, &p, 0) == kOfxStatOK)
        g.paramSuite->paramSetValue(p, v.c_str());
}

// --------------------------------------------------------------------------- images
bool fetchFrame(OfxImageEffectHandle effect, const char* clipName, double time, Frame& out) {
    ImageView v;
    if (!acquire(effect, clipName, time, v)) return false;
    const int w = v.w(), h = v.h();
    out.w = w;
    out.h = h;
    out.rgb.assign(static_cast<size_t>(w) * h * 3, 0.0f);
    for (int y = 0; y < h; ++y) {
        // pxStride() is ALREADY the full pixel size (comps * bytes-per-channel).
        // Multiplying by comps again advanced the pointer x*comps pixels instead of
        // x: with Flame's RGBA (comps=4) clips the plate was read at every 4th
        // pixel, which squeezed the whole image to 1/4 width and tiled it four
        // times across the frame - every job's input and every render was a 4-up
        // mosaic (measured: period 480px on 1920, 328px on 1312, strip
        // correlation 0.997; found 2026-10-16).
        const unsigned char* src = v.row(y) + static_cast<size_t>(v.x1) * v.pxStride();
        float* dst = out.rgb.data() + static_cast<size_t>(y) * w * 3;
        for (int x = 0; x < w; ++x) {
            const unsigned char* q = src + static_cast<size_t>(x) * v.pxStride();
            dst[x * 3 + 0] = readPix_global(q, v.depth, 0);
            dst[x * 3 + 1] = readPix_global(q, v.depth, 1);
            dst[x * 3 + 2] = readPix_global(q, v.depth, 2);
        }
    }
    release(v);
    return true;
}

void writeFrameToImage(OfxImageEffectHandle effect, double time, const Frame& src) {
    OfxImageClipHandle clip;
    if (g.effectSuite->clipGetHandle(effect, kOfxImageEffectOutputClipName, &clip, 0) != kOfxStatOK) return;
    OfxPropertySetHandle img;
    if (g.effectSuite->clipGetImage(clip, time, 0, &img) != kOfxStatOK) return;
    ImageView v;
    v.img = img;
    v.clip = clip;
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    g.propertySuite->propGetInt(img, kOfxImagePropBounds, 0, &x1);
    g.propertySuite->propGetInt(img, kOfxImagePropBounds, 1, &y1);
    g.propertySuite->propGetInt(img, kOfxImagePropBounds, 2, &x2);
    g.propertySuite->propGetInt(img, kOfxImagePropBounds, 3, &y2);
    int rowBytes = 0;
    g.propertySuite->propGetInt(img, kOfxImagePropRowBytes, 0, &rowBytes);
    void* data = nullptr;
    g.propertySuite->propGetPointer(img, kOfxImagePropData, 0, &data);
    if (!data) {
        g.effectSuite->clipReleaseImage(img);
        return;
    }
    bool half = false;
    bool isByte = false;
    bool isShort = false;
    char* depth = nullptr;
    if (g.propertySuite->propGetString(img, kOfxImageEffectPropPixelDepth, 0, &depth) == kOfxStatOK && depth) {
        half = (std::strcmp(depth, kOfxBitDepthHalf) == 0);
        isByte = (std::strcmp(depth, kOfxBitDepthByte) == 0);
        isShort = (std::strcmp(depth, kOfxBitDepthShort) == 0);
    }
    int comps = 3;
    char* cs = nullptr;
    if (g.propertySuite->propGetString(img, kOfxImageEffectPropComponents, 0, &cs) == kOfxStatOK && cs)
        comps = (std::strcmp(cs, kOfxImageComponentRGB) == 0) ? 3 : 4;

    const int dw = x2 - x1 + 1, dh = y2 - y1 + 1;
    const size_t stride = isByte ? 1u : (half || isShort) ? 2u : 4u;
    const int copyW = std::min(dw, src.w), copyH = std::min(dh, src.h);
    unsigned char* base = static_cast<unsigned char*>(data);
    // The ONE sRGB encode lives in linearToSrgbGamma (dnz_hdr.h) so the byte path and
    // the math tests cannot drift apart. convertHdrForOutput leaves the frame LINEAR;
    // when it encoded as well, everything was double-encoded (black 3 -> 25, mid grey
    // 0.18 -> 181 instead of 118).
    auto writeCh8 = [&](float v) -> unsigned char {         // linear 0..1 -> gamma-encoded 0..255
        return static_cast<unsigned char>(std::lround(
            std::clamp(linearToSrgbGamma(v), 0.0f, 1.0f) * 255.0f));
    };
    auto writeCh16 = [&](float v) -> uint16_t {             // linear 0..1 -> gamma-encoded 0..65535
        return static_cast<uint16_t>(std::lround(
            std::clamp(linearToSrgbGamma(v), 0.0f, 1.0f) * 65535.0f));
    };
    for (int y = 0; y < dh; ++y) {
        unsigned char* row = (rowBytes >= 0) ? base + static_cast<size_t>(y) * rowBytes
                                             : base + static_cast<size_t>(dh - 1 - y) * (-rowBytes);
        for (int x = 0; x < dw; ++x) {
            unsigned char* q = row + (static_cast<size_t>(x) + x1) * comps * stride;
            float rgb[3] = {0, 0, 0};
            if (y < copyH && x < copyW) {
                const float* s = src.rgb.data() + (static_cast<size_t>(y) * src.w + x) * 3;
                rgb[0] = s[0];
                rgb[1] = s[1];
                rgb[2] = s[2];
            }
            for (int c = 0; c < comps && c < 3; ++c) {
                if (isByte)
                    q[c] = writeCh8(rgb[c]);
                else if (isShort)
                    reinterpret_cast<uint16_t*>(q)[c] = writeCh16(rgb[c]);
                else if (half)
                    reinterpret_cast<uint16_t*>(q)[c] = toHalfBits(rgb[c]);
                else
                    reinterpret_cast<float*>(q)[c] = rgb[c];
            }
            if (comps == 4) {                          // keep a solid alpha: the plate is opaque HDR
                if (isByte)
                    q[3] = 255;
                else if (isShort)
                    reinterpret_cast<uint16_t*>(q)[3] = 65535;
                else if (half)
                    reinterpret_cast<uint16_t*>(q)[3] = toHalfBits(1.0f);
                else
                    reinterpret_cast<float*>(q)[3] = 1.0f;
            }
        }
    }
    g.effectSuite->clipReleaseImage(img);
}

Frame resized(const Frame& src, int w, int h) {
    Frame out;
    out.w = w;
    out.h = h;
    out.rgb.assign(static_cast<size_t>(w) * h * 3, 0.0f);
    if (src.w <= 0 || src.h <= 0 || w <= 0 || h <= 0) return out;
    const double sx = static_cast<double>(src.w) / w;
    const double sy = static_cast<double>(src.h) / h;
    for (int y = 0; y < h; ++y) {
        double fy = (y + 0.5) * sy - 0.5;
        int y0 = std::max(0, std::min(src.h - 1, static_cast<int>(std::floor(fy))));
        int y1 = std::min(src.h - 1, y0 + 1);
        double wy = std::max(0.0, std::min(1.0, fy - y0));
        for (int x = 0; x < w; ++x) {
            double fx = (x + 0.5) * sx - 0.5;
            int x0 = std::max(0, std::min(src.w - 1, static_cast<int>(std::floor(fx))));
            int x1 = std::min(src.w - 1, x0 + 1);
            double wx = std::max(0.0, std::min(1.0, fx - x0));
            float* d = out.rgb.data() + (static_cast<size_t>(y) * w + x) * 3;
            for (int c = 0; c < 3; ++c) {
                float a = src.rgb[(static_cast<size_t>(y0) * src.w + x0) * 3 + c];
                float b = src.rgb[(static_cast<size_t>(y0) * src.w + x1) * 3 + c];
                float cc = src.rgb[(static_cast<size_t>(y1) * src.w + x0) * 3 + c];
                float dd = src.rgb[(static_cast<size_t>(y1) * src.w + x1) * 3 + c];
                double top = a + (b - a) * wx;
                double bot = cc + (dd - cc) * wx;
                d[c] = static_cast<float>(top + (bot - top) * wy);
            }
        }
    }
    return out;
}

void workSizeFor(int w, int h, int longSide, int& ow, int& oh) {
    auto snap = [](int n) { return std::max(32, static_cast<int>(std::lround(n / 32.0)) * 32); };
    if (longSide <= 0 || (w <= longSide && h <= longSide)) {
        ow = snap(w);
        oh = snap(h);
        return;
    }
    double s = static_cast<double>(longSide) / std::max(w, h);
    ow = snap(static_cast<int>(std::lround(w * s)));
    oh = snap(static_cast<int>(std::lround(h * s)));
}

// --------------------------------------------------------------------------- image helpers (used by fetchFrame/writeFrameToImage)
// ImageView, readPix_global, fromHalfBits, toHalfBits are defined inline in dnz_hdr.h.
bool acquire(OfxImageEffectHandle effect, const char* clipName, double time, ImageView& v) {
    OfxStatus s = g.effectSuite->clipGetHandle(effect, clipName, &v.clip, 0);
    if (s != kOfxStatOK) return false;
    s = g.effectSuite->clipGetImage(v.clip, time, 0, &v.img);
    if (s != kOfxStatOK) return false;
    OfxPropertySetHandle p = v.img;
    if (g.propertySuite->propGetInt(p, kOfxImagePropBounds, 0, &v.x1) != kOfxStatOK) return false;
    g.propertySuite->propGetInt(p, kOfxImagePropBounds, 1, &v.y1);
    g.propertySuite->propGetInt(p, kOfxImagePropBounds, 2, &v.x2);
    g.propertySuite->propGetInt(p, kOfxImagePropBounds, 3, &v.y2);
    g.propertySuite->propGetInt(p, kOfxImagePropRowBytes, 0, &v.rowBytes);
    void* raw = nullptr;
    g.propertySuite->propGetPointer(p, kOfxImagePropData, 0, &raw);
    v.base = static_cast<const unsigned char*>(raw);
    char* comps = nullptr;
    if (g.propertySuite->propGetString(p, kOfxImageEffectPropComponents, 0, &comps) == kOfxStatOK && comps) {
        v.comps = (std::strcmp(comps, kOfxImageComponentRGB) == 0) ? 3 : 4;
    }
    char* depth = nullptr;
    if (g.propertySuite->propGetString(p, kOfxImageEffectPropPixelDepth, 0, &depth) == kOfxStatOK && depth) {
        if (std::strcmp(depth, kOfxBitDepthByte) == 0) v.depth = 1;
        else if (std::strcmp(depth, kOfxBitDepthHalf) == 0) v.depth = 2;
        else if (std::strcmp(depth, kOfxBitDepthShort) == 0) v.depth = 3;
        else v.depth = 0;
    }
    return v.base != nullptr;
}

void release(ImageView& v) {
    if (v.img) {
        g.effectSuite->clipReleaseImage(v.img);
        v.img = nullptr;
    }
}

// --------------------------------------------------------------------------- colour space names (for job.json)
const char* inCsName(int v) {
    switch (v) {
        case IN_SRGB_LINEAR: return "srgb_linear";
        case IN_ACESCG: return "acescg";
        case IN_ACESCCT: return "acescct";
        case IN_DISPLAY_SRGB: return "display_srgb";
        default: return "srgb_linear";
    }
}

const char* outCsName(int v) {
    switch (v) {
        case OUT_ACESCG: return "acescg";
        case OUT_SRGB_LINEAR: return "srgb_linear";
        case OUT_ACESCCT: return "acescct";
        default: return "acescg";
    }
}

// --------------------------------------------------------------------------- engine client (readHdrFrame)
bool readHdrFrame(const std::string& job, int n, int w, int h, int index, Frame& out, int expectedOutCS) {
    std::string path = job + "/hdr.f16";
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    long want = static_cast<long>(w) * h * 3;
    if (std::fseek(f, static_cast<long>(index) * want * 2, SEEK_SET) != 0) {
        std::fclose(f);
        return false;
    }
    std::vector<uint16_t> raw(static_cast<size_t>(want));
    if (std::fread(raw.data(), sizeof(uint16_t), raw.size(), f) != raw.size()) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    out.w = w;
    out.h = h;
    out.rgb.resize(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {          // half -> float32, hardware conversion
        _Float16 h16;
        uint16_t bits = raw[i];
        std::memcpy(&h16, &bits, 2);
        out.rgb[i] = static_cast<float>(h16);
    }
    return true;
}

}  // namespace dnz
