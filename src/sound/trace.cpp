// DSOUND call trace, bounded: the last loading steps before a guest crash
#include "sound/sound.h"

using namespace kernel;
using namespace sound;

namespace {
std::atomic<int> g_budget{4000};
}

void sound::Trace(const char* what, uint32_t obj, uint32_t caller, uint32_t result) {
    if (g_budget.fetch_sub(1) <= 0) return;
    LOGW("dsound-call %s obj=%08X caller=%08X result=%08X", what, obj, caller, result);
}
