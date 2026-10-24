// Engine client: spawn the Python server, talk JSON-lines over a Unix socket,
// and manage the job folder (float16 pixels in, float16 pixels out).
#include "dnz_hdr.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <netinet/in.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>

extern char** environ;

namespace dnz {
namespace {

std::mutex g_rootMutex;
std::string g_root;

// ---- minimal JSON extraction (our own payloads are flat; engine replies are flat too)
bool jsonFindKey(const std::string& s, const std::string& key, size_t& pos) {
    std::string needle = "\"" + key + "\"";
    size_t p = s.find(needle);
    if (p == std::string::npos) return false;
    p = s.find(':', p + needle.size());
    if (p == std::string::npos) return false;
    pos = p + 1;
    return true;
}

std::string jsonRaw(const std::string& s, const std::string& key) {
    size_t p = 0;
    if (!jsonFindKey(s, key, p)) return "";
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) ++p;
    if (p >= s.size()) return "";
    if (s[p] == '"') {                                 // string, with \" unescaping
        std::string out;
        for (size_t i = p + 1; i < s.size(); ++i) {
            if (s[i] == '\\' && i + 1 < s.size()) {
                char c = s[i + 1];
                out += (c == 'n') ? '\n' : (c == 't') ? '\t' : c;
                ++i;
                continue;
            }
            if (s[i] == '"') break;
            out += s[i];
        }
        return out;
    }
    size_t e = p;
    while (e < s.size() && s[e] != ',' && s[e] != '}' && s[e] != ']' && s[e] != ' ') ++e;
    return s.substr(p, e - p);
}

std::string jsonStr(const std::string& s, const std::string& key) { return jsonRaw(s, key); }
double jsonNum(const std::string& s, const std::string& key, double dflt = 0.0) {
    std::string v = jsonRaw(s, key);
    if (v.empty() || v == "null") return dflt;
    try {
        return std::stod(v);
    } catch (...) {
        return dflt;
    }
}
bool jsonBool(const std::string& s, const std::string& key, bool dflt = false) {
    std::string v = jsonRaw(s, key);
    if (v == "true") return true;
    if (v == "false") return false;
    return dflt;
}

std::string jescape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o;
}

// ---- socket helpers
inline std::string socketPath() { return engineRoot() + "/run/dnz_hdr.sock"; }

int connectSocket(int timeoutMs) {
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", socketPath().c_str());
    struct timeval tv;
    tv.tv_sec = timeoutMs / 1000;
    tv.tv_usec = (timeoutMs % 1000) * 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    if (::connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

bool mkdirp(const std::string& p) {
    // ONE stream, consumed progressively. Constructing stringstream(p) inside the
    // loop condition re-read the FIRST token on every iteration: an absolute path
    // starts with '/', so that token is always EMPTY, the empty-token `continue`
    // fired every time and the loop never reached mkdir - an infinite, syscall-free
    // spin. This hung the plugin's Warm Up/Analyze thread whenever the engine had
    // to be spawned (engine not running -> spawnEngine -> mkdirp), with nothing in
    // any log because posix_spawn was never reached. Proven with a 4-iteration repro,
    // 2026-10-15.
    std::stringstream ss(p);
    std::string cur, part;
    const bool abs = !p.empty() && p[0] == '/';
    while (std::getline(ss, part, '/')) {
        if (part.empty()) continue;                       // leading '/' yields an empty first token
        if (cur.empty()) cur = abs ? "/" + part : part;
        else            cur += "/" + part;
        if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
    }
    return true;
}

pid_t spawnEngine(std::string* err) {
    std::string root = engineRoot();
    std::string py = root + "/venv/bin/python";
    std::string script = root + "/engine/dnz_hdr_engine.py";
    if (::access(py.c_str(), X_OK) != 0) {
        if (err) *err = "engine not installed at " + root + " (re-run the installer)";
        return 0;
    }
    if (::access(script.c_str(), R_OK) != 0) {
        if (err) *err = "missing " + script + " (re-run the installer)";
        return 0;
    }
    mkdirp(root + "/run");
    mkdirp(root + "/cache/hdr_jobs");
    std::string logPath = root + "/run/dnz_hdr_engine_stdout.log";

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    int fd = ::open(logPath.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd >= 0) {
        posix_spawn_file_actions_addopen(&actions, 1, logPath.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        posix_spawn_file_actions_addopen(&actions, 2, logPath.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    }
    std::vector<std::string> args = {py, script};
    std::vector<char*> cargs;
    for (auto& a : args) cargs.push_back(const_cast<char*>(a.c_str()));
    cargs.push_back(nullptr);

    std::string rootEnv = "AIFX_NATIVE_ROOT=" + root;
    std::vector<std::string> envStore = {rootEnv};
    std::vector<char*> cenv;
    for (char** e = environ; *e; ++e) cenv.push_back(*e);
    cenv.push_back(const_cast<char*>(envStore[0].c_str()));
    cenv.push_back(nullptr);

    pid_t pid = 0;
    int rc = posix_spawn(&pid, py.c_str(), &actions, nullptr, cargs.data(), cenv.data());
    posix_spawn_file_actions_destroy(&actions);
    if (fd >= 0) ::close(fd);
    if (rc != 0) {
        if (err) *err = std::string("could not start engine (posix_spawn ") + std::strerror(rc) + ")";
        return 0;
    }
    return pid;
}

}  // namespace

std::string engineRoot() {
    std::lock_guard<std::mutex> lk(g_rootMutex);
    if (!g_root.empty()) return g_root;

    if (const char* env = std::getenv("AIFX_NATIVE_ROOT")) {
        if (*env) {
            g_root = env;
            return g_root;
        }
    }

    char path[PATH_MAX];
    uint32_t size = sizeof(path);
    std::string root;
    if (_NSGetExecutablePath(path, &size) == 0) {
        char real[PATH_MAX];
        if (::realpath(path, real)) {
            std::string exe = real;
            size_t p = exe.find("/Contents/MacOS/");
            if (p != std::string::npos) {
                std::string res = exe.substr(0, p) + "/Contents/Resources/engine_root.txt";
                std::ifstream f(res);
                if (f) {
                    std::getline(f, root);
                    root = root.substr(0, root.find_last_not_of(" \n\r\t") + 1);
                }
            }
        }
    }
    if (root.empty()) root = "/Users/Shared/MatteLTXNative";
    g_root = root;
    return g_root;
}

bool ensureEngine(std::string* err) {
    std::string reply;
    if (engineCall("{\"cmd\":\"ping\"}", &reply, nullptr) && jsonBool(reply, "ok")) return true;
    if (!spawnEngine(err)) return false;
    for (int i = 0; i < 60; ++i) {
        ::usleep(500 * 1000);
        if (engineCall("{\"cmd\":\"ping\"}", &reply, nullptr) && jsonBool(reply, "ok")) return true;
    }
    if (err) *err = "engine did not start; see " + engineRoot() + "/run/dnz_hdr_engine_stdout.log";
    return false;
}

bool engineCall(const std::string& line, std::string* reply, std::string* err) {
    int fd = connectSocket(2000);
    if (fd < 0) {
        if (err) *err = "engine is not running";
        return false;
    }
    std::string msg = line + "\n";
    if (::send(fd, msg.data(), msg.size(), 0) < 0) {
        ::close(fd);
        if (err) *err = std::string("could not send to engine: ") + std::strerror(errno);
        return false;
    }
    std::string buf;
    char tmp[4096];
    ssize_t n;
    while ((n = ::recv(fd, tmp, sizeof(tmp), 0)) > 0) {
        buf.append(tmp, static_cast<size_t>(n));
        if (buf.find('\n') != std::string::npos) break;
    }
    ::close(fd);
    if (buf.empty()) {
        if (err) *err = "engine did not answer";
        return false;
    }
    if (reply) *reply = buf.substr(0, buf.find('\n'));
    return true;
}

std::string makeJobDir() {
    mkdirp(engineRoot() + "/cache/hdr_jobs");
    auto now = std::chrono::system_clock::now().time_since_epoch();
    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    std::random_device rd;
    std::ostringstream s;
    s << engineRoot() << "/cache/hdr_jobs/" << ms << "-" << std::hex << (rd() & 0xffff);
    std::string dir = s.str();
    mkdirp(dir);
    return dir;
}

bool writeJobJson(const std::string& job, const EffectData& p, int n, int w, int h, bool preview) {
    const char* offload = (p.offload == 1) ? "cpu" : (p.offload == 2) ? "disk" : "none";
    std::ostringstream s;
    s << "{"
      << "\"job\":\"" << jescape(job) << "\","
      << "\"frames\":" << n << ","
      << "\"width\":" << w << ","
      << "\"height\":" << h << ","
      << "\"fps\":" << p.fps << ","
      << "\"in_cs\":\"" << inCsName(p.inCS) << "\","
      << "\"out_cs\":\"" << outCsName(p.outCS) << "\","
      << "\"seed\":" << p.seed << ","
      << "\"high_quality\":" << (p.highQuality ? "true" : "false") << ","
      << "\"keyframes\":" << (p.keyframes ? "true" : "false") << ","
      << "\"fp8\":" << (p.fp8 ? "true" : "false") << ","
      << "\"offload\":\"" << offload << "\","
      << "\"preview\":" << (preview ? "true" : "false")
      << "}";
    std::ofstream f(job + "/job.json");
    if (!f) return false;
    f << s.str();
    return true;
}

bool writeInputF16(const std::string& job, const std::vector<Frame>& frames) {
    if (frames.empty()) return false;
    std::string tmp = job + "/input.f16.tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    std::vector<uint16_t> row;
    for (const Frame& fr : frames) {
        size_t count = static_cast<size_t>(fr.w) * fr.h * 3;
        row.resize(count);
        for (size_t i = 0; i < count; ++i) {           // float -> half, via the hardware conversion
            float v = fr.rgb[i];
            if (!(v > -65504.0f))                      // NaN -> 0 (and -inf, which a plate never has)
                v = (v < 0.0f && v == v) ? -65504.0f : 0.0f;
            if (v > 65504.0f) v = 65504.0f;            // half's largest finite value
            _Float16 h = (_Float16)v;                  // round-to-nearest-even, subnormals included
            uint16_t bits;
            std::memcpy(&bits, &h, 2);
            row[i] = bits;
        }
        if (std::fwrite(row.data(), sizeof(uint16_t), count, f) != count) {
            std::fclose(f);
            return false;
        }
    }
    std::fclose(f);
    return ::rename(tmp.c_str(), (job + "/input.f16").c_str()) == 0;
}

bool engineStatus(const std::string& job, std::string& state, double& progress, std::string& message) {
    std::string reply, err;
    if (!engineCall("{\"cmd\":\"status\",\"job\":\"" + jescape(job) + "\"}", &reply, &err)) {
        state = "error";
        message = err;
        return false;
    }
    state = jsonStr(reply, "state");
    progress = jsonNum(reply, "progress", 0.0);
    message = jsonStr(reply, "message");
    return true;
}

// Load the 22B transformer without a clip, so the first Analyze is fast. Returns the
// engine's warmup folder to poll with engineStatus() like any job,
// or an empty string if the engine could not be reached.
std::string engineWarmup(const std::string& offload, bool fp8) {
    if (!ensureEngine(nullptr)) return std::string();
    std::string body = "{\"cmd\":\"warmup\",\"offload\":\"" + jescape(offload) + "\"";
    if (fp8) body += ",\"fp8\":true";
    body += "}";
    std::string reply, err;
    if (!engineCall(body, &reply, &err)) return std::string();
    return jsonStr(reply, "job");                  // the engine assigns <root>/run/warmup
}

// readHdrFrame lives in render.cpp (7-param, validates expectedOutCS).

void removeJobDir(const std::string& job) {
    if (job.empty()) return;
    engineCall("{\"cmd\":\"cancel\",\"job\":\"" + jescape(job) + "\"}", nullptr, nullptr);
    std::string cmd = "rm -rf '" + job + "'";
    if (std::system(cmd.c_str()) != 0) { /* best effort */ }
}

}  // namespace dnz