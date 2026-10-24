// OFX entry points for DNZ SDR to HDR.
#include "dnz_hdr.h"

#include <cstring>

namespace dnz {
namespace {

void setHost(OfxHost* host) {
    g.host = host;
    g.effectSuite = (OfxImageEffectSuiteV1*)host->fetchSuite(host->host, kOfxImageEffectSuite, 1);
    g.propertySuite = (OfxPropertySuiteV1*)host->fetchSuite(host->host, kOfxPropertySuite, 1);
    g.paramSuite = (OfxParameterSuiteV1*)host->fetchSuite(host->host, kOfxParameterSuite, 1);
}

OfxStatus mainEntry(const char* action, const void* handle, OfxPropertySetHandle inArgs,
                    OfxPropertySetHandle outArgs) {
    OfxImageEffectHandle effect = (OfxImageEffectHandle)handle;
    if (std::strcmp(action, kOfxActionLoad) == 0) return kOfxStatOK;
    if (std::strcmp(action, kOfxActionUnload) == 0) return kOfxStatOK;
    if (std::strcmp(action, kOfxActionDescribe) == 0) return ofxDescribe(effect);
    if (std::strcmp(action, kOfxImageEffectActionDescribeInContext) == 0)
        return ofxDescribeInContext(effect, inArgs);
    if (std::strcmp(action, kOfxActionCreateInstance) == 0) return ofxCreateInstance(effect);
    if (std::strcmp(action, kOfxActionDestroyInstance) == 0) return ofxDestroyInstance(effect);
    if (std::strcmp(action, kOfxImageEffectActionRender) == 0) return ofxRender(effect, inArgs);
    if (std::strcmp(action, kOfxActionInstanceChanged) == 0) return ofxInstanceChanged(effect, inArgs);
    (void)outArgs;
    return kOfxStatReplyDefault;
}

}  // namespace

OfxPlugin pluginStruct = {
    kOfxImageEffectPluginApi,
    1,
    "com.dnz.sdrtohdr",          // plugin identifier
    1,                           // version major
    0,                           // version minor
    setHost,
    mainEntry,
};

}  // namespace dnz

extern "C" {

OfxExport int OfxGetNumberOfPlugins(void) { return 1; }

OfxExport OfxPlugin* OfxGetPlugin(int nth) {
    if (nth == 0) return &dnz::pluginStruct;
    return nullptr;
}

}  // extern "C"
