// BinkGetSummary divides bytes read * 1000 by the read time in ms + 1. Host reads take under a millisecond,
// the time stays 0 and the quotient of a large movie overflows the 32 bit div. Raise the time just enough.
#include <mutex>
#include <unordered_set>

#include "common/options.h"
#include "xdk/hle.h"

using namespace kernel;

X86_EXTERN_FUNC(GUEST_ORIG(BinkGetSummary));
X86_FUNC(GUEST(BinkGetSummary)) {
    static std::atomic<uint32_t> diag_calls{0};
    uint32_t diag_n = diag_calls.fetch_add(1, std::memory_order_relaxed);
    uint32_t bink = R32(ctx.esp + 4);
    if (diag_n < 8) LOGR("diag: bink-summary[%u] object=%08X", diag_n, bink);
    if (bink) {
        // 0x134 bytes read, 0x13C read time in ms
        uint64_t scaled = uint64_t(R32(bink + 0x134)) * 1000;
        uint32_t time = R32(bink + 0x13C);
        if (scaled >> 32 > time) W32(bink + 0x13C, uint32_t(scaled >> 32));
    }
    GUEST_ORIG(BinkGetSummary)(ctx, base);
}

// Movies step one frame per game frame, timed by the console's 30 Hz present. Above 30 fps a frame is decoded
// and advanced only once BinkWait, which follows the movie's own rate and audio, says it is due.
namespace {

constexpr uint32_t kBinkWait = SYM(BinkWait);
std::mutex g_held_lock;
std::unordered_set<uint32_t> g_held;  // movies whose frame was not decoded this game frame
std::unordered_set<uint32_t> g_skipped;  // movies advanced to their last frame by skip_intros

}  // namespace

X86_EXTERN_FUNC(GUEST_ORIG(BinkDoFrame));
X86_FUNC(GUEST(BinkDoFrame)) {
    static std::atomic<uint32_t> diag_calls{0};
    uint32_t diag_n = diag_calls.fetch_add(1, std::memory_order_relaxed);
    uint32_t bink = R32(ctx.esp + 4);
    if (game::Opt().skip_intros && bink) {
        const uint32_t frames = R32(bink + 8);
        if (frames && frames < 100000) {
            // The game's video manager checks current == total immediately
            // after BinkDoFrame. Setting that guest-visible state lets its
            // normal close/transition path run without decoding the movie.
            W32(bink + 0x10, frames);
            {
                std::lock_guard<std::mutex> lock(g_held_lock);
                g_skipped.insert(bink);
            }
            if (diag_n < 16)
                LOGR("diag: bink-skip-intro[%u] object=%08X frame=%u/%u", diag_n, bink, frames, frames);
            ctx.eax = 0;
            ctx.esp += 8;
            return;
        }
    }
    bool waiting = bink && CallGuest(ctx, kBinkWait, {bink});
    if (diag_n < 16) LOGR("diag: bink-do-frame[%u] object=%08X wait=%u", diag_n, bink, waiting);
    if (waiting) {
        std::lock_guard<std::mutex> lock(g_held_lock);
        g_held.insert(bink);
        ctx.eax = 0;
        ctx.esp += 8;
        return;
    }
    GUEST_ORIG(BinkDoFrame)(ctx, base);
}

X86_EXTERN_FUNC(GUEST_ORIG(BinkNextFrame));
X86_FUNC(GUEST(BinkNextFrame)) {
    static std::atomic<uint32_t> diag_calls{0};
    uint32_t diag_n = diag_calls.fetch_add(1, std::memory_order_relaxed);
    uint32_t bink = R32(ctx.esp + 4);
    {
        std::lock_guard<std::mutex> lock(g_held_lock);
        if (g_skipped.erase(bink)) {
            if (diag_n < 16) LOGR("diag: bink-next-frame[%u] skipped=1 object=%08X", diag_n, bink);
            ctx.esp += 8;
            return;
        }
        if (g_held.erase(bink)) {
            if (diag_n < 16) LOGR("diag: bink-next-frame[%u] object=%08X held=1", diag_n, bink);
            ctx.esp += 8;
            return;
        }
    }
    if (diag_n < 16) LOGR("diag: bink-next-frame[%u] object=%08X held=0", diag_n, bink);
    GUEST_ORIG(BinkNextFrame)(ctx, base);
}
