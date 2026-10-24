// Load the built .ofx exactly the way a host does: dlopen -> OfxGetPlugin ->
// check the descriptor -> drive Describe / DescribeInContext with a mock suite.
//
// This catches the mistakes that only show up inside Flame: wrong property types,
// missing entry points, parameters that never get defined, group parents that do
// not stick. It does not prove the engine works - that is tests/engine_smoke.sh.
#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxParam.h"

#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <string>
#include <vector>

// ------------------------------------------------------------------ mock host
using Props = std::map<std::string, std::vector<std::string>>;
static std::map<const void*, Props> g_props;      // string-valued properties
static std::map<const void*, Props> g_intProps;   // int-valued, stored as text
static std::vector<std::string> g_paramTypes;     // in define order
static std::vector<std::string> g_paramNames;
static std::map<std::string, std::string> g_paramParent;   // param -> parent group
static std::map<std::string, std::string> g_paramLabel;    // param -> label
static std::map<const void*, std::string> g_handleToParam; // paramDefine property handle -> param
static std::vector<std::string> g_clips;

static intptr_t g_nextHandle = 0x10000;

static void putStr(OfxPropertySetHandle p, const char* n, int i, const char* v) {
    auto& vec = g_props[p][n];
    vec.resize(i + 1);
    vec[i] = v ? v : "";
}
static const char* getStr(OfxPropertySetHandle p, const char* n, int i) {
    auto it = g_props.find(p);
    if (it == g_props.end()) return nullptr;
    auto jt = it->second.find(n);
    if (jt == it->second.end() || i < 0 || i >= (int)jt->second.size()) return nullptr;
    return jt->second[i].c_str();
}

static OfxStatus propSetString(OfxPropertySetHandle p, const char* n, int i, const char* v) {
    putStr(p, n, i, v);
    auto it = g_handleToParam.find(p);                  // remember labels/parents of params
    if (it != g_handleToParam.end() && i == 0 && v) {
        if (!std::strcmp(n, kOfxPropLabel)) g_paramLabel[it->second] = v;
        if (!std::strcmp(n, kOfxParamPropParent)) g_paramParent[it->second] = v;
    }
    return kOfxStatOK;
}
static OfxStatus propSetInt(OfxPropertySetHandle p, const char* n, int i, int v) {
    g_intProps[p][n].resize(i + 1);
    char b[32];
    std::snprintf(b, sizeof(b), "%d", v);
    g_intProps[p][n][i] = b;
    return kOfxStatOK;
}
static OfxStatus propSetDouble(OfxPropertySetHandle p, const char* n, int i, double v) {
    g_intProps[p][n].resize(i + 1);
    char b[48];
    std::snprintf(b, sizeof(b), "%g", v);
    g_intProps[p][n][i] = b;
    return kOfxStatOK;
}
static OfxStatus propSetPointer(OfxPropertySetHandle p, const char* n, int i, void* v) {
    g_intProps[p][n].resize(i + 1);
    g_intProps[p][n][i] = v ? "1" : "0";
    return kOfxStatOK;
}
static OfxStatus propGetString(OfxPropertySetHandle p, const char* n, int i, char** v) {
    static std::string store;
    const char* s = getStr(p, n, i);
    if (!s) return kOfxStatErrUnknown;
    store = s;
    *v = const_cast<char*>(store.c_str());
    return kOfxStatOK;
}
static OfxStatus propGetInt(OfxPropertySetHandle p, const char* n, int i, int* v) {
    auto it = g_intProps.find(p);
    if (it == g_intProps.end()) return kOfxStatErrUnknown;
    auto jt = it->second.find(n);
    if (jt == it->second.end() || i < 0 || i >= (int)jt->second.size()) return kOfxStatErrUnknown;
    *v = std::atoi(jt->second[i].c_str());
    return kOfxStatOK;
}
static OfxStatus propGetDouble(OfxPropertySetHandle p, const char* n, int i, double* v) {
    int iv = 0;
    if (propGetInt(p, n, i, &iv) != kOfxStatOK) return kOfxStatErrUnknown;
    *v = iv;
    return kOfxStatOK;
}
static OfxStatus propGetPointer(OfxPropertySetHandle p, const char* n, int i, void** v) {
    int iv = 0;
    if (propGetInt(p, n, i, &iv) != kOfxStatOK) return kOfxStatErrUnknown;
    *v = reinterpret_cast<void*>(static_cast<intptr_t>(iv));
    return kOfxStatOK;
}

static OfxStatus clipDefine(OfxImageEffectHandle, const char* n, OfxPropertySetHandle* h) {
    g_clips.push_back(n);
    *h = reinterpret_cast<OfxPropertySetHandle>(++g_nextHandle);
    return kOfxStatOK;
}
static OfxStatus getPropertySet(OfxImageEffectHandle, OfxPropertySetHandle* h) {
    // one stable handle for the effect itself, so describe's property sets can be read back
    static intptr_t effectProps = ++g_nextHandle;
    *h = reinterpret_cast<OfxPropertySetHandle>(effectProps);
    return kOfxStatOK;
}
static OfxStatus getParamSet(OfxImageEffectHandle, OfxParamSetHandle* h) {
    *h = reinterpret_cast<OfxParamSetHandle>(++g_nextHandle);
    return kOfxStatOK;
}
static std::string g_lastParamName;
static OfxStatus paramDefine(OfxParamSetHandle, const char* type, const char* n,
                             OfxPropertySetHandle* propHandle) {
    g_paramTypes.push_back(type);
    g_paramNames.push_back(n);
    OfxPropertySetHandle ph = reinterpret_cast<OfxPropertySetHandle>(++g_nextHandle);
    g_handleToParam[ph] = n;
    *propHandle = ph;
    g_lastParamName = n;
    return kOfxStatOK;
}

static OfxPropertySuiteV1 g_propSuite;
static OfxImageEffectSuiteV1 g_effectSuite;
static OfxParameterSuiteV1 g_paramSuite;
static OfxHost g_host;

static const void* fetchSuite(OfxPropertySetHandle, const char* name, int) {
    if (!std::strcmp(name, kOfxPropertySuite)) return &g_propSuite;
    if (!std::strcmp(name, kOfxImageEffectSuite)) return &g_effectSuite;
    if (!std::strcmp(name, kOfxParameterSuite)) return &g_paramSuite;
    return nullptr;
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1]
                                  : "build/DNZSDRtoHDR.ofx.bundle/Contents/MacOS/DNZSDRtoHDR.ofx";
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        std::printf("dlopen FAILED: %s\n", dlerror());
        return 1;
    }
    auto numberOfPlugins = (int (*)())dlsym(h, "OfxGetNumberOfPlugins");
    auto getPlugin = (OfxPlugin * (*)(int)) dlsym(h, "OfxGetPlugin");
    if (!numberOfPlugins || !getPlugin) {
        std::printf("missing Ofx entry points\n");
        return 1;
    }
    int n = numberOfPlugins();
    OfxPlugin* p = getPlugin(0);
    std::printf("plugins      : %d\n", n);
    std::printf("api          : %s (v%d)\n", p->pluginApi, p->apiVersion);
    std::printf("identifier   : %s  version %d.%d\n", p->pluginIdentifier,
                p->pluginVersionMajor, p->pluginVersionMinor);
    std::printf("setHost      : %s\n", p->setHost ? "OK" : "NULL (host cannot talk to it)");
    std::printf("mainEntry    : %s\n", p->mainEntry ? "OK" : "NULL (host cannot talk to it)");
    if (!p->setHost || !p->mainEntry) return 1;

    g_propSuite.propSetString = propSetString;
    g_propSuite.propSetInt = propSetInt;
    g_propSuite.propSetDouble = propSetDouble;
    g_propSuite.propSetPointer = propSetPointer;
    g_propSuite.propGetString = propGetString;
    g_propSuite.propGetInt = propGetInt;
    g_propSuite.propGetDouble = propGetDouble;
    g_propSuite.propGetPointer = propGetPointer;
    g_effectSuite.clipDefine = clipDefine;
    g_effectSuite.getPropertySet = getPropertySet;
    g_effectSuite.getParamSet = getParamSet;
    g_paramSuite.paramDefine = paramDefine;

    g_host.host = reinterpret_cast<OfxPropertySetHandle>(1);
    g_host.fetchSuite = fetchSuite;
    p->setHost(&g_host);

    void* effect = reinterpret_cast<void*>(0x10);
    int failures = 0;

    OfxStatus s = p->mainEntry(kOfxActionDescribe, effect, nullptr, nullptr);
    OfxPropertySetHandle ep = reinterpret_cast<OfxPropertySetHandle>(0);
    g_effectSuite.getPropertySet(reinterpret_cast<OfxImageEffectHandle>(effect), &ep);
    const char* label = getStr(ep, kOfxPropLabel, 0);
    const char* grouping = getStr(ep, kOfxImageEffectPluginPropGrouping, 0);
    const char* contexts = getStr(ep, kOfxImageEffectPropSupportedContexts, 0);
    std::printf("\ndescribe       : %s\n", s == kOfxStatOK ? "OK" : "FAILED");
    std::printf("  label        : %s\n", label ? label : "(unset)");
    std::printf("  grouping     : %s\n", grouping ? grouping : "(unset)");
    std::printf("  contexts     : %s\n", contexts ? contexts : "(unset)");
    if (s != kOfxStatOK || !label || !grouping) ++failures;

    s = p->mainEntry(kOfxImageEffectActionDescribeInContext, effect, nullptr, nullptr);
    std::printf("describeInCtx  : %s\n", s == kOfxStatOK ? "OK" : "FAILED");
    std::printf("  clips        : %zu\n", g_clips.size());
    std::printf("  params       : %zu\n", g_paramNames.size());
    for (size_t i = 0; i < g_paramNames.size(); ++i) {
        const std::string& nm = g_paramNames[i];
        std::string parent = g_paramParent.count(nm) ? g_paramParent[nm] : "root";
        std::string lab = g_paramLabel.count(nm) ? g_paramLabel[nm] : "";
        std::printf("    %-20s %-24s in %-14s %s\n", nm.c_str(), g_paramTypes[i].c_str(),
                    parent.c_str(), lab.c_str());
    }
    if (s != kOfxStatOK || g_paramNames.empty()) ++failures;

    // instance lifecycle
    s = p->mainEntry(kOfxActionCreateInstance, effect, nullptr, nullptr);
    std::printf("createInstance : %s\n", s == kOfxStatOK ? "OK" : "FAILED");
    if (s != kOfxStatOK) ++failures;
    void* raw = nullptr;
    propGetPointer(ep, kOfxPropInstanceData, 0, &raw);
    std::printf("  instanceData : %s\n", raw ? "set" : "MISSING");
    if (!raw) ++failures;

    std::printf(failures ? "\nRESULT: %d failure(s)\n" : "\nRESULT: all checks passed\n", failures);
    return failures ? 1 : 0;
}
