// The title's Direct3D 8 calls drawn with Vulkan 1.1, dynamic rendering and a timeline semaphore. The SDK's
// translated shaders are HLSL, which glslang turns into SPIR-V (vk_shader.cpp).
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include <dxgiformat.h>
#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>

#include "app/app.h"
#include "common/aspect.h"
#include "common/options.h"
#include "core/log.h"
#include "d3d/gpu_frame.h"
#include "gpu/backend.h"
#include "gpu/shaders.h"
#include "gpu/texture.h"
#include "gpu/vertex.h"
#include "host.h"
#include "kernel/kernel.h"
#include "kernel/memory.h"
#include "vk_shader.h"

namespace gpu {

namespace {

constexpr int kFrames = 3;
constexpr VkDeviceSize kUploadSize = 128ull << 20;  // per frame: vertices, indices, constants, texture staging
constexpr VkDeviceSize kUboRange = 3072;           // b0 and b1: 192 vertex constants, DrawConstants under it
constexpr VkDeviceSize kUboAlign = 256;
constexpr VkFormat kBgra = VK_FORMAT_B8G8R8A8_UNORM;
constexpr uint32_t kSetsPerPool = 1024;
static_assert(sizeof(DrawConstants) <= kUboRange);

const char* kBlitHlsl = R"(
cbuffer B : register(b1) { float4 uvrect; float4 opts; };
Texture2D img : register(t0);
Texture2D lut : register(t1);
SamplerState ls : register(s0);
SamplerState ps : register(s1);
struct O { float4 p : SV_Position; float2 uv : TEXCOORD0; };
O vs(uint id : SV_VertexID) {
    O o;
    float2 t = float2((id << 1) & 2, id & 2);
    o.p = float4(t * float2(2, -2) + float2(-1, 1), 0, 1);
    o.uv = uvrect.xy + t * uvrect.zw;
    return o;
}
float4 psmain(O i) : SV_Target {
    float4 c = 0;
    int n = max((int)opts.z, 1);
    float2 px = fwidth(i.uv);
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) c += img.SampleLevel(ls, i.uv + px * ((float2(x, y) + 0.5) / n - 0.5), 0);
    c /= n * n;
    c.rgb *= opts.w;
    if (opts.x > 0) c.rgb = float3(lut.Sample(ps, float2(c.r, 0.5)).r, lut.Sample(ps, float2(c.g, 0.5)).g, lut.Sample(ps, float2(c.b, 0.5)).b);
    return float4(c.rgb, opts.y > 0 ? c.a : 1);
}
)";

// the screen filters, the frame onto the screen through one of them. src.xy a texel of the frame, src.z seconds for
// what moves (grain, VHS), src.w the colour look. pscrt: crt.x the lines, a whole number of pixels each so they never
// beat against the screen's; crt.y how dark the gaps between them are, crt.z the tube's curve, crt.w the phosphor
// mask's dimming; look.x the corners' shade, look.y the brightness the lines and mask take back. post.x HDR on,
// post.y FXAA on; both go before the filter, psplain is them alone
const char* kFilterHlsl = R"(
cbuffer B : register(b1) { float4 uvrect; float4 opts; float4 crt; float4 look; float4 src; float4 post; };
Texture2D img : register(t0);
Texture2D lut : register(t1);
SamplerState ls : register(s0);
SamplerState ps : register(s1);
struct O { float4 p : SV_Position; float2 uv : TEXCOORD0; };
O vs(uint id : SV_VertexID) {
    O o;
    float2 t = float2((id << 1) & 2, id & 2);
    o.p = float4(t * float2(2, -2) + float2(-1, 1), 0, 1);
    o.uv = uvrect.xy + t * uvrect.zw;
    return o;
}
float3 Fetch(float2 uv) {
    float3 c = img.SampleLevel(ls, uv, 0).rgb;
    if (opts.x > 0) c = float3(lut.Sample(ps, float2(c.r, 0.5)).r, lut.Sample(ps, float2(c.g, 0.5)).g, lut.Sample(ps, float2(c.b, 0.5)).b);
    return c;
}
float Luma(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }
float Noise(float2 p) { return frac(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453); }
float Corners(float2 uv, float k) { return pow(saturate(16 * uv.x * uv.y * (1 - uv.x) * (1 - uv.y)), k); }

// a Reinhard tone map against the console's overbright look, with the saturation and the shadows it loses given back
float3 Tone(float3 c) {
    if (post.x <= 0) return c;
    float l = dot(c, float3(0.2126, 0.7152, 0.0722));
    float shadow = 1 - smoothstep(0, 0.8, l);
    c = lerp(l, c, 1 + shadow);
    float3 mapped = pow(max(c / (c + 1), 0), 1 / 2.2);
    c = lerp(c, mapped, 0.8);
    return saturate(c * (1 - shadow * 0.5));
}

// FXAA: across an edge the luma's direction says, two taps or four, the four unless they reach past the corners
float3 Fxaa(float2 uv) {
    float2 px = src.xy;
    float3 m = Fetch(uv);
    float nw = Luma(Fetch(uv + float2(-1, -1) * px)), ne = Luma(Fetch(uv + float2(1, -1) * px));
    float sw = Luma(Fetch(uv + float2(-1, 1) * px)), se = Luma(Fetch(uv + float2(1, 1) * px));
    float lm = Luma(m);
    float lo = min(lm, min(min(nw, ne), min(sw, se))), hi = max(lm, max(max(nw, ne), max(sw, se)));
    if (hi - lo < max(0.0312, hi * 0.125)) return m;
    float2 dir = float2(-((nw + ne) - (sw + se)), (nw + sw) - (ne + se));
    float reduce = max((nw + ne + sw + se) * (0.25 / 8), 1.0 / 128);
    dir = clamp(dir / (min(abs(dir.x), abs(dir.y)) + reduce), -8, 8) * px;
    float3 a = 0.5 * (Fetch(uv + dir * (1.0 / 3 - 0.5)) + Fetch(uv + dir * (2.0 / 3 - 0.5)));
    float3 b = a * 0.5 + 0.25 * (Fetch(uv - dir * 0.5) + Fetch(uv + dir * 0.5));
    float lb = Luma(b);
    return lb < lo || lb > hi ? a : b;
}

// the frame at a point with what the graphics settings add before a filter
float3 Frame(float2 uv) { return Tone(post.y > 0 ? Fxaa(uv) : Fetch(uv)); }

float4 psplain(O i) : SV_Target { return float4(Frame(i.uv), 1); }

float4 pscrt(O i) : SV_Target {
    float2 cc = i.uv * 2 - 1;
    cc *= 1 + crt.z * cc.yx * cc.yx;
    if (any(abs(cc) > 1)) return float4(0, 0, 0, 1);
    float2 uv = cc * 0.5 + 0.5;
    float3 c = Frame(uv);
    // the beam is brightest at a line's middle, and a bright one spreads over the gap
    float f = frac(uv.y * crt.x);
    float gap = cos(3.14159265 * f);
    float scan = 1 - crt.y * gap * gap;
    scan = lerp(scan, 1, saturate(Luma(c)) * 0.4);
    // an aperture grille: each column of pixels one of the phosphors
    float col = fmod(floor(i.p.x), 3);
    float3 grille = 1 - crt.w * float3(col != 0, col != 1, col != 2);
    return float4(saturate(c * scan * grille * Corners(uv, look.x) * look.y), 1);
}

// contrast adaptive: an edge gains what its neighbours lack, less where it is near black or white already
float4 pssharpen(O i) : SV_Target {
    float3 c = Fetch(i.uv);
    float3 n = Fetch(i.uv - float2(0, src.y)), s = Fetch(i.uv + float2(0, src.y));
    float3 w = Fetch(i.uv - float2(src.x, 0)), e = Fetch(i.uv + float2(src.x, 0));
    float3 lo = min(c, min(min(n, s), min(w, e))), hi = max(c, max(max(n, s), max(w, e)));
    float3 amp = sqrt(saturate(min(lo, 1 - hi) / max(hi, 1e-4)));
    return float4(Tone(saturate(c + (4 * c - n - s - w - e) * amp * 0.35)), 1);
}

float4 pscolor(O i) : SV_Target {
    float2 uv = i.uv;
    float t = src.z;
    int mode = (int)src.w;
    if (mode == 4) {
        // VHS: lines torn sideways a little, a band of worse tracking rolling down
        float row = floor(uv.y * 240);
        float band = saturate(1 - abs(frac(uv.y - t * 0.07) - 0.5) * 12);
        uv.x += (Noise(float2(row, floor(t * 30))) - 0.5) * (0.0015 + band * 0.006);
    }
    float3 c = Frame(uv);
    if (mode == 0) {  // vibrant
        c = lerp(Luma(c), c, 1.35);
        c = (c - 0.5) * 1.08 + 0.5;
    } else if (mode == 1) {  // noir: black and white, hard, grain
        float l = saturate((Luma(c) - 0.5) * 1.35 + 0.5);
        l += (Noise(i.p.xy + frac(t) * 97) - 0.5) * 0.05;
        c = l * Corners(uv, 0.35);
    } else if (mode == 2) {  // sepia
        c = float3(dot(c, float3(0.393, 0.769, 0.189)), dot(c, float3(0.349, 0.686, 0.168)),
                   dot(c, float3(0.272, 0.534, 0.131)));
        c *= Corners(uv, 0.2);
    } else if (mode == 3) {  // night vision: brightened green, grain, faint lines, a dark rim
        float l = pow(saturate(Luma(c)), 0.7) * 1.5;
        l += (Noise(i.p.xy + frac(t) * 131) - 0.5) * 0.12;
        l *= 1 - 0.12 * fmod(floor(i.p.y), 2);
        c = float3(0.25, 1.0, 0.35) * l * Corners(uv, 0.6);
    } else if (mode == 4) {  // VHS: colour bleeding sideways, washed out, grain
        float off = src.x * 2.5;
        c.r = Fetch(uv + float2(off, 0)).r;
        c.b = Fetch(uv - float2(off, 0)).b;
        c = lerp(Luma(c), c, 0.8) * 0.95 + 0.03;
        c += (Noise(i.p.xy + frac(t) * 173) - 0.5) * 0.06;
    }
    return float4(saturate(c), 1);
}
)";

const char* kDepthHlsl = R"(
Texture2D<float> img : register(t0);
SamplerState ps : register(s1);
struct O { float4 p : SV_Position; float2 uv : TEXCOORD0; };
O vs(uint id : SV_VertexID) {
    O o;
    o.uv = float2((id << 1) & 2, id & 2);
    o.p = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}
float psmain(O i) : SV_Depth { return img.SampleLevel(ps, i.uv, 0); }
)";

// ---- the console's values as Vulkan's

float Bits(uint32_t v) {
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}

void Color(uint32_t argb, float* o) {
    o[0] = ((argb >> 16) & 255) / 255.0f;
    o[1] = ((argb >> 8) & 255) / 255.0f;
    o[2] = (argb & 255) / 255.0f;
    o[3] = (argb >> 24) / 255.0f;
}

VkBlendFactor Blend(uint32_t v, bool alpha) {
    switch (v) {
        case 0: return VK_BLEND_FACTOR_ZERO;
        case 1: return VK_BLEND_FACTOR_ONE;
        case 0x300: return alpha ? VK_BLEND_FACTOR_SRC_ALPHA : VK_BLEND_FACTOR_SRC_COLOR;
        case 0x301: return alpha ? VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA : VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 0x302: return VK_BLEND_FACTOR_SRC_ALPHA;
        case 0x303: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case 0x304: return VK_BLEND_FACTOR_DST_ALPHA;
        case 0x305: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 0x306: return alpha ? VK_BLEND_FACTOR_DST_ALPHA : VK_BLEND_FACTOR_DST_COLOR;
        case 0x307: return alpha ? VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA : VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case 0x308: return alpha ? VK_BLEND_FACTOR_ONE : VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case 0x8001: return VK_BLEND_FACTOR_CONSTANT_COLOR;
        case 0x8002: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        case 0x8003: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
        case 0x8004: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
        default: return VK_BLEND_FACTOR_ONE;
    }
}

VkBlendOp BlendOp(uint32_t v) {
    switch (v) {
        case 0x800A: return VK_BLEND_OP_SUBTRACT;
        case 0x800B: return VK_BLEND_OP_REVERSE_SUBTRACT;
        case 0x8007: return VK_BLEND_OP_MIN;
        case 0x8008: return VK_BLEND_OP_MAX;
        default: return VK_BLEND_OP_ADD;
    }
}

// D3DCMP_NEVER 0x200 to ALWAYS 0x207: VkCompareOp has the same order, and the shaders take the same numbers
uint32_t CmpIndex(uint32_t v) { return v >= 0x200 && v <= 0x207 ? v - 0x200 : 7; }
VkCompareOp Cmp(uint32_t v) { return VkCompareOp(CmpIndex(v)); }

VkStencilOp StencilOp(uint32_t v) {
    switch (v) {
        case 0: return VK_STENCIL_OP_ZERO;
        case 0x1E01: return VK_STENCIL_OP_REPLACE;
        case 0x1E02: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
        case 0x1E03: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
        case 0x150A: return VK_STENCIL_OP_INVERT;
        case 0x8507: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
        case 0x8508: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
        default: return VK_STENCIL_OP_KEEP;
    }
}

VkSamplerAddressMode Address(uint32_t v) {
    switch (v) {
        case 2: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case 3: case 5: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case 4: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        default: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
}

// the input assembler's formats the SDK's vertex.cpp names
VkFormat VertexFormat(uint32_t dxgi) {
    switch (dxgi) {
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case DXGI_FORMAT_R32G32B32_FLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
        case DXGI_FORMAT_R32G32_FLOAT: return VK_FORMAT_R32G32_SFLOAT;
        case DXGI_FORMAT_R32_FLOAT: return VK_FORMAT_R32_SFLOAT;
        case DXGI_FORMAT_R32_UINT: return VK_FORMAT_R32_UINT;
        case DXGI_FORMAT_R16G16B16A16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
        case DXGI_FORMAT_R16G16B16A16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
        case DXGI_FORMAT_R16G16_SNORM: return VK_FORMAT_R16G16_SNORM;
        case DXGI_FORMAT_R16G16_SINT: return VK_FORMAT_R16G16_SINT;
        case DXGI_FORMAT_R16_SNORM: return VK_FORMAT_R16_SNORM;
        case DXGI_FORMAT_R16_SINT: return VK_FORMAT_R16_SINT;
        case DXGI_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_R8G8_UNORM: return VK_FORMAT_R8G8_UNORM;
        case DXGI_FORMAT_R8_UNORM: return VK_FORMAT_R8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
        default: return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
}

// ---- DXT on the CPU, for GPUs without BC textures (most phones)

void Rgb565(uint16_t c, uint8_t* bgr) {
    bgr[2] = uint8_t((c >> 11) * 255 / 31);
    bgr[1] = uint8_t(((c >> 5) & 63) * 255 / 63);
    bgr[0] = uint8_t((c & 31) * 255 / 31);
}

// one 4x4 block's colours, BGRA, out 16 x 4 bytes; dxt1 alpha for the 3 colour mode
void ColourBlock(const uint8_t* b, bool dxt1, uint8_t* out) {
    uint16_t c0 = uint16_t(b[0] | b[1] << 8), c1 = uint16_t(b[2] | b[3] << 8);
    uint8_t pal[4][4] = {};
    Rgb565(c0, pal[0]);
    Rgb565(c1, pal[1]);
    for (int i = 0; i < 4; i++) pal[i][3] = 255;
    for (int k = 0; k < 3; k++) {
        if (c0 > c1 || !dxt1) {
            pal[2][k] = uint8_t((2 * pal[0][k] + pal[1][k]) / 3);
            pal[3][k] = uint8_t((pal[0][k] + 2 * pal[1][k]) / 3);
        } else {
            pal[2][k] = uint8_t((pal[0][k] + pal[1][k]) / 2);
            pal[3][k] = 0;
        }
    }
    if (dxt1 && c0 <= c1) pal[3][3] = 0;
    uint32_t bits = uint32_t(b[4] | b[5] << 8 | b[6] << 16 | uint32_t(b[7]) << 24);
    for (int i = 0; i < 16; i++) std::memcpy(out + i * 4, pal[(bits >> (i * 2)) & 3], 4);
}

// a level of DXT1, 3 or 5 blocks (bw x bh per slice, row_pitch bytes a block row) into w x h BGRA texels
void DecodeDxt(Host host, const std::vector<uint8_t>& in, uint32_t row_pitch, uint32_t w, uint32_t h, uint32_t z,
               std::vector<uint8_t>& out) {
    uint32_t block = host == Host::BC1 ? 8 : 16;
    uint32_t bw = std::max(1u, (w + 3) / 4), bh = std::max(1u, (h + 3) / 4);
    out.assign(size_t(w) * h * z * 4, 0);
    uint8_t px[64];
    // Xbox volume DXT data stores up to four z-slice blocks together for each
    // 4x4 XY block (xemu's s3tc_decompress_3d layout), not whole 2D slices in
    // sequence. 2D textures are unchanged because their block depth is one.
    for (uint32_t bz = 0; bz < (z + 3) / 4; bz++) {
        const uint32_t block_depth = std::min(4u, z - bz * 4);
        const size_t z_group = size_t(bz) * bh * row_pitch * 4;
        for (uint32_t by = 0; by < bh; by++)
            for (uint32_t bx = 0; bx < bw; bx++)
                for (uint32_t zs = 0; zs < block_depth; zs++) {
                    const uint32_t s = bz * 4 + zs;
                    const size_t at = z_group + (size_t(by) * bw + bx) * block_depth * block + size_t(zs) * block;
                    if (at + block > in.size()) continue;
                    const uint8_t* b = &in[at];
                    ColourBlock(host == Host::BC1 ? b : b + 8, host == Host::BC1, px);
                    if (host == Host::BC2) {
                        for (int i = 0; i < 16; i++) px[i * 4 + 3] = uint8_t(((b[i / 2] >> ((i & 1) * 4)) & 15) * 17);
                    } else if (host == Host::BC3) {
                        uint8_t a[8] = {b[0], b[1]};
                        if (a[0] > a[1]) {
                            for (int i = 1; i < 7; i++) a[i + 1] = uint8_t(((7 - i) * a[0] + i * a[1]) / 7);
                        } else {
                            for (int i = 1; i < 5; i++) a[i + 1] = uint8_t(((5 - i) * a[0] + i * a[1]) / 5);
                            a[6] = 0;
                            a[7] = 255;
                        }
                        uint64_t bits = 0;
                        for (int i = 0; i < 6; i++) bits |= uint64_t(b[2 + i]) << (8 * i);
                        for (int i = 0; i < 16; i++) px[i * 4 + 3] = a[(bits >> (3 * i)) & 7];
                    }
                    for (uint32_t y = 0; y < 4 && by * 4 + y < h; y++)
                        for (uint32_t x = 0; x < 4 && bx * 4 + x < w; x++)
                            std::memcpy(&out[((size_t(s) * h + by * 4 + y) * w + bx * 4 + x) * 4], px + (y * 4 + x) * 4, 4);
                }
    }
}

// ---- device memory: 64 MB blocks per memory type, first fit inside them; big allocations get their own

class Allocator {
public:
    struct Alloc {
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkDeviceSize off = 0, size = 0;
        uint8_t* map = nullptr;
        int block = -1;  // -1: memory of its own
    };

    void Init(VkPhysicalDevice pd, VkDevice dev) {
        dev_ = dev;
        vkGetPhysicalDeviceMemoryProperties(pd, &props_);
    }

    bool Get(const VkMemoryRequirements& r, VkMemoryPropertyFlags want, VkMemoryPropertyFlags nice, bool linear, Alloc& out) {
        int type = Type(r.memoryTypeBits, want | nice);
        if (type < 0) type = Type(r.memoryTypeBits, want);
        if (type < 0) return false;
        bool host = props_.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        std::lock_guard lock(m_);
        if (r.size > kBlock / 4) {
            out = {};
            if (!Allocate(uint32_t(type), r.size, out.mem)) return false;
            out.size = r.size;
            if (host) vkMapMemory(dev_, out.mem, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&out.map));
            return true;
        }
        for (size_t i = 0; i < blocks_.size(); i++) {
            Block& b = blocks_[i];
            if (b.type != uint32_t(type) || b.linear != linear) continue;
            if (Carve(b, r, out)) {
                out.block = int(i);
                return true;
            }
        }
        Block b;
        b.type = uint32_t(type);
        b.linear = linear;
        if (!Allocate(b.type, kBlock, b.mem)) return false;
        if (host) vkMapMemory(dev_, b.mem, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&b.map));
        b.free[0] = kBlock;
        blocks_.push_back(std::move(b));
        Carve(blocks_.back(), r, out);
        out.block = int(blocks_.size() - 1);
        return true;
    }

    void Free(Alloc& a) {
        if (!a.mem) return;
        std::lock_guard lock(m_);
        if (a.block < 0) {
            vkFreeMemory(dev_, a.mem, nullptr);
        } else {
            auto& free = blocks_[size_t(a.block)].free;
            auto it = free.emplace(a.off, a.size).first;
            if (auto next = std::next(it); next != free.end() && it->first + it->second == next->first) {
                it->second += next->second;
                free.erase(next);
            }
            if (it != free.begin()) {
                auto prev = std::prev(it);
                if (prev->first + prev->second == it->first) {
                    prev->second += it->second;
                    free.erase(it);
                }
            }
        }
        a = {};
    }

    bool HostCoherent(const Alloc& a) const {
        uint32_t type = a.block < 0 ? own_type_ : blocks_[size_t(a.block)].type;
        return props_.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }

private:
    static constexpr VkDeviceSize kBlock = 64ull << 20;
    struct Block {
        VkDeviceMemory mem = VK_NULL_HANDLE;
        uint32_t type = 0;
        bool linear = false;
        uint8_t* map = nullptr;
        std::map<VkDeviceSize, VkDeviceSize> free;  // offset to size
    };

    int Type(uint32_t bits, VkMemoryPropertyFlags want) const {
        for (uint32_t i = 0; i < props_.memoryTypeCount; i++)
            if ((bits >> i & 1) && (props_.memoryTypes[i].propertyFlags & want) == want) return int(i);
        return -1;
    }

    bool Allocate(uint32_t type, VkDeviceSize size, VkDeviceMemory& mem) {
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = size;
        ai.memoryTypeIndex = type;
        own_type_ = type;
        if (vkAllocateMemory(dev_, &ai, nullptr, &mem) == VK_SUCCESS) return true;
        LOGE("vulkan: out of device memory (%llu bytes)", (unsigned long long)size);
        return false;
    }

    bool Carve(Block& b, const VkMemoryRequirements& r, Alloc& out) {
        for (auto it = b.free.begin(); it != b.free.end(); ++it) {
            VkDeviceSize o = it->first, s = it->second;
            VkDeviceSize at = (o + r.alignment - 1) / r.alignment * r.alignment;
            if (at + r.size > o + s) continue;
            b.free.erase(it);
            if (at > o) b.free[o] = at - o;
            if (at + r.size < o + s) b.free[at + r.size] = o + s - at - r.size;
            out.mem = b.mem;
            out.off = at;
            out.size = r.size;
            out.map = b.map ? b.map + at : nullptr;
            return true;
        }
        return false;
    }

    VkDevice dev_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties props_{};
    std::vector<Block> blocks_;
    uint32_t own_type_ = 0;
    std::mutex m_;
};

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;    // sampled: every level, a depth image's depth alone
    VkImageView attach = VK_NULL_HANDLE;  // as a render target: level 0, depth and stencil
    Allocator::Alloc mem;
    VkFormat fmt = VK_FORMAT_UNDEFINED;
    uint32_t w = 1, h = 1, d = 1, layers = 1, levels = 1;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    bool depth = false, stencil = false;
    explicit operator bool() const { return image != VK_NULL_HANDLE; }
};

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    Allocator::Alloc mem;
};

// a console surface the renderer draws into
struct Target {
    Image img;
    bool depth = false;
    uint32_t gw = 0, gh = 0, w = 0, h = 0;
    uint32_t data = 0, bytes = 0;
    uint64_t epoch = 0, sig = 0;  // last lock seen, guest memory as last matched
    uint64_t content_serial = 0;  // advances when the guest writes this surface
    uint64_t use_serial = 0;      // most recently bound descriptor for data-only readbacks
    bool drawn = false;           // the GPU wrote it since the last readback
    std::vector<uint8_t> read;    // what Sync left in guest memory
    Image fb;                     // copy read by draws that sample the target they render to
    uint32_t perf2_draws = 0;     // draws into it in the current perf2 window
};

struct Tex {
    Image img;
    TextureDesc d;
    VkFormat fmt = VK_FORMAT_UNDEFINED;
    bool decode = false;  // DXT decoded on the CPU
    uint32_t levels = 1;
    uint64_t sig = 0;
    uint64_t frame = 0;
    uint64_t palette = 0;
    // per face: the target serial last copied in, and the image it was copied into. A frame's reflection faces
    // change once per frame, so the copy is done once instead of on every draw that samples the cube.
    uint64_t cube_face_serial[6] = {};
    VkImage cube_img = VK_NULL_HANDLE;
};

struct PsoKey {
    uint64_t vs;
    uint64_t ps;
    uint32_t regs;
    uint32_t topo;  // Topology, strip at bit 4
    uint32_t has_rt;
    uint32_t has_ds;
    uint32_t elements[16];  // per register: DXGI format, buffer << 8, offset << 16
    uint32_t strides[16];
    uint32_t buffers;
    uint32_t blend, src, dst, op, wmask;
    uint32_t zen, zwrite, zfunc;
    uint32_t sten, sfunc, smask, swmask, sfail, szfail, spass;
    uint32_t cull, ccw, fill;
    int32_t zbias;
    float zslope;
    uint32_t rtfmt;
};

struct BlitOptions {
    bool gamma = false;
    bool alpha = false;
    uint32_t taps = 1;
    float mul = 1;
    bool over = false;  // premultiplied over the target, which keeps its alpha
};

struct DrawTextures {
    TexInfo info[4];
    VkImageView view[4] = {};
    VkFormat fmt[4] = {};
    uint32_t levels[4] = {1, 1, 1, 1};
    bool rt[4] = {};
    float k[4] = {1, 1, 1, 1};
    float wide[4] = {};
    bool blur[4] = {};
    uint32_t gw[4] = {};  // a render target's guest width, 0 for any other texture
};

struct DrawShaders {
    bool prog = false;
    bool fixed_transform = false;
    FixedKey fk;
    uint64_t vkey = 0, vh = 0, pkey = 0, ph = 0;
    VkShaderModule vs = VK_NULL_HANDLE, ps = VK_NULL_HANDLE;
};

struct GuestExtent {
    uint32_t w, h;
};

// ---- what the console's state asks of a draw

// the game's 2D layer leaves oPos.w at 0: screen space, drawn with w = 1 as XYZRHW is
std::string VertexProgram(const VertexShader& vs, uint16_t regs, uint64_t decode) {
    const std::string clamp = "    p.w = magclamp(p.w);\n";
    const std::string screen_space = "    p.w = p.w == 0 ? 1 : magclamp(p.w);\n";
    std::string src = VertexProgramHlsl(vs.program, regs, decode);
    size_t at = src.find(clamp);
    if (at != std::string::npos) src.replace(at, clamp.size(), screen_space);
    return src;
}

// a point list writes its size: Vulkan leaves it undefined otherwise; empty when the source has no main to wrap
std::string PointsHlsl(const std::string& src) {
    size_t at = src.find("VOut main(");
    if (at == std::string::npos) return {};
    size_t p = at + 10, sp = src.find(' ', p);
    std::string type = src.substr(p, sp - p);
    // the module's entry point is named main: the program's own main goes by another name
    std::string out = src;
    out.replace(at, 10, "VOut vs_body(");
    return out + "\nVOut main_points(" + type + " i, out float psize : PSIZE) { psize = 1; return vs_body(i); }\n";
}

bool UnitTexcoords(const Assembled& a) {
    constexpr float kLo = -1e-4f, kHi = 1.0001f;
    constexpr uint32_t kSmallDrawVertices = 4096;
    if (!a.vertices.empty() || !a.vertex_count || a.vertex_count > kSmallDrawVertices) return false;
    bool any = false;
    for (int r = 0; r < 16; r++) {
        const VertexElement& e = a.elements[size_t(r)];
        if (!(a.regs >> r & 1) || e.format != DXGI_FORMAT_R32G32_FLOAT) continue;
        const VertexBuffer& vb = a.buffers[e.buffer];
        if (!vb.src || !vb.stride || uint64_t(a.vertex_count) * vb.stride > vb.bytes) return false;
        for (uint32_t v = 0; v < a.vertex_count; v++) {
            float uv[2];
            std::memcpy(uv, vb.src + size_t(v) * vb.stride + e.offset, sizeof(uv));
            if (!(uv[0] >= kLo && uv[0] <= kHi && uv[1] >= kLo && uv[1] <= kHi)) return false;
        }
        any = true;
    }
    return any;
}

constexpr uint32_t kTaddressWrap = 1;
constexpr uint32_t kTaddressClamp = 3;

bool AnyStageWraps(const State& s) {
    for (const auto& t : s.tss)
        if (t[kTssAddressU] == kTaddressWrap || t[kTssAddressV] == kTaddressWrap) return true;
    return false;
}

constexpr uint32_t kFmtD16 = 0x2C;
constexpr uint32_t kFmtLinD16 = 0x30;
uint8_t ShadowDepthKind(uint32_t format) { return format == kFmtD16 || format == kFmtLinD16 ? 2 : 1; }

uint32_t GuestPitch(uint32_t width) { return (width * 4 + 63) & ~63u; }

constexpr uint32_t kSigWholeMax = 65536;
constexpr uint32_t kSigSamples = 64;
constexpr uint32_t kSigSampleBytes = 64;

uint64_t MemSig(uint32_t data, uint32_t total) {
    const uint8_t* p = static_cast<const uint8_t*>(kernel::HostPtr(data));
    uint64_t sig = Hash(&total, sizeof(total));
    if (total <= kSigWholeMax) return Hash(p, total, sig);
    for (uint32_t i = 0; i < kSigSamples; i++) {
        const uint8_t* sample = p + uint64_t(total - kSigSampleBytes) * i / (kSigSamples - 1);
        sig = Hash(sample, kSigSampleBytes, sig);
    }
    return sig;
}

bool Changed(const Target& t) {
    if (t.drawn || t.read.empty()) return false;
    uint32_t row = t.gw * 4, pitch = GuestPitch(t.gw);
    const uint8_t* p = static_cast<const uint8_t*>(kernel::HostPtr(t.data));
    for (uint32_t y = 0; y < t.gh; y++)
        if (std::memcmp(p + size_t(y) * pitch, &t.read[size_t(y) * row], row)) return true;
    return false;
}

bool LatchCpuWrite(Target& t) {
    uint64_t e = DirtyEpoch(t.data);
    if (e <= t.epoch) return false;
    t.epoch = e;
    uint64_t sig = MemSig(t.data, t.bytes);
    if (sig == t.sig && !Changed(t)) return false;
    t.sig = sig;
    return true;
}

std::vector<uint8_t> LatchCpuChanges(Target& t) {
    std::vector<uint8_t> px(t.read.size(), 0);
    uint32_t guest_pitch = GuestPitch(t.gw);
    const uint8_t* guest = static_cast<const uint8_t*>(kernel::HostPtr(t.data));
    for (uint32_t y = 0; y < t.gh; y++)
        for (uint32_t x = 0; x < t.gw; x++) {
            const uint8_t* c = guest + size_t(y) * guest_pitch + x * 4;
            size_t i = (size_t(y) * t.gw + x) * 4;
            if (!std::memcmp(c, &t.read[i], 4)) continue;
            std::memcpy(&t.read[i], c, 4);
            std::memcpy(&px[i], c, 3);
            px[i + 3] = 255;
        }
    return px;
}

TexKind TextureKind(const TextureDesc& d) {
    if (d.cube) return kTexCube;
    if (d.depth > 1) return kTex3D;
    return kTex2D;
}

bool SameTexture(const Tex& tx, const TextureDesc& d) {
    return tx.img && tx.d.format == d.format && tx.d.width == d.width && tx.d.height == d.height &&
           tx.d.depth == d.depth && tx.d.levels == d.levels && tx.d.cube == d.cube && tx.d.pitch == d.pitch;
}

struct StagePalette {
    uint32_t colours[256] = {};
    uint32_t va = 0;
    uint64_t hash = 0;
};

void ReadStagePalette(const State& s, int stage, const TextureDesc& d, StagePalette& palette) {
    constexpr uint32_t kFmtP8 = 0x0B, kPaletteCommon = 0, kPaletteData = 4, kPaletteSizeShift = 30;
    if (d.format != kFmtP8 || !s.palettes[size_t(stage)]) return;
    uint32_t header = s.palettes[size_t(stage)];
    palette.va = mem::VirtualFromPhysical(kernel::R32(header + kPaletteData));
    uint32_t size_code = kernel::R32(header + kPaletteCommon) >> kPaletteSizeShift;
    size_t bytes = (std::size(palette.colours) >> size_code) * sizeof(uint32_t);
    std::memcpy(palette.colours, kernel::HostPtr(palette.va), bytes);
    palette.hash = Hash(palette.colours, sizeof(palette.colours));
}

constexpr uint32_t kTexfLinear = 2;
constexpr uint32_t kTexfAnisotropic = 3;

uint32_t SimpleState(const State& s, uint32_t method) { return s.simple[method >> 2]; }

DrawShaders ClassifyShaders(const State& s, const Assembled& a) {
    DrawShaders sh;
    sh.prog = (s.vs_handle & 1) && s.vs && !s.vs->program.empty();
    sh.fixed_transform = !sh.prog && !a.pretransformed;
    if (!sh.prog) sh.fk = MakeFixedKey(s);
    return sh;
}

void EnsureProgramHash(const State& s) {
    if (!s.vs->hash) s.vs->hash = Hash(s.vs->program.data(), s.vs->program.size() * 4) | 1;
}

// The guest allocates shader handles dynamically. Identify this Forza program
// by its bytecode, not by an address observed in one execution.
bool IsForzaFrontendShader(const State& s) {
    if (!s.vs || s.vs->program_type != 0x2078 || s.vs->program.size() != 40) return false;
    uint64_t h = 14695981039346656037ull;
    for (uint32_t word : s.vs->program) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            h ^= (word >> shift) & 0xffu;
            h *= 1099511628211ull;
        }
    }
    return h == 0x95B0B937E21846C2ull;
}

constexpr uint64_t kDeclarationSalt = 0x10000;
constexpr uint64_t kPretransformedSalt = 0x5A5A;
constexpr uint64_t kPointsSalt = 0x9017;
constexpr uint64_t kPixelKeySalt = 0x9E3779B97F4A7C15ull;

uint64_t VertexKey(const State& s, const Assembled& a, const DrawShaders& sh) {
    uint64_t seed = sh.prog ? s.vs->hash ^ a.regs : Hash(&s.vs_handle, 4, a.regs | kDeclarationSalt);
    uint64_t vkey = Hash(&a.decode, 8, seed);
    if (!sh.prog && a.pretransformed) vkey ^= kPretransformedSalt;
    if (!sh.prog) vkey = Hash(&sh.fk, sizeof(sh.fk), vkey);
    if (a.topology == Topology::Points) vkey ^= kPointsSalt;
    return vkey;
}

std::string VertexHlsl(const State& s, const Assembled& a, const DrawShaders& sh) {
    if (sh.prog) return VertexProgram(*s.vs, a.regs, a.decode);
    return FixedVertexHlsl(a.regs, a.pretransformed, a.decode, sh.fk);
}

void KeyBlend(const State& s, PsoKey& k) {
    k.blend = SimpleState(s, kMBlendEnable) != 0;
    if (k.blend) {
        k.src = SimpleState(s, kMSrcBlend);
        k.dst = SimpleState(s, kMDestBlend);
        k.op = SimpleState(s, kMBlendOp);
    }
    uint32_t cw = SimpleState(s, kMColorWriteEnable);
    // console ARGB bytes to Vulkan's RGBA bits
    k.wmask = ((cw >> 16) & 1) | ((cw >> 8) & 1) << 1 | (cw & 1) << 2 | ((cw >> 24) & 1) << 3;
}

// diagnostic switches, set from marker files in Present(): they exist only to find out whether a
// missing object is being culled or depth-rejected, and change nothing unless a marker is present
bool g_diag_nocull = false;
bool g_diag_nodepth = false;
bool g_diag_skycheck = false;  // clear the scene to magenta: whatever stays magenta was never drawn
bool g_diag_skystrip = false;  // read the top of the scene back after every draw of the frame
bool g_diag_noclear = false;   // drop the scene's colour clear: what was drawn before it stays visible

void KeyRasterizer(const State& s, const Assembled& a, PsoKey& k) {
    constexpr uint32_t kCullCw = 0x900;
    uint32_t cull = s.rs[kRsCullMode];
    k.cull = cull != 0 && a.topology == Topology::Triangles;
    k.ccw = cull == kCullCw;
    if (g_diag_nocull) k.cull = 0;
    k.fill = s.rs[kRsFillMode];
}

constexpr float kAlphaTestAlways = 7.0f;

// D3DRS_FOGTABLEMODE values
constexpr uint32_t kFogTableNone = 0;
constexpr float kFogTableLinear = 3.0f;

// FOG off leaves the constants as fog disabled: the shaders' fog factor is then 0, the colour as drawn
void FillFogConstants(const State& s, bool vertex_program, DrawConstants& c) {
    if (!s.rs[kRsFogEnable] || !game::Opt().fog) return;
    const uint32_t table_mode = s.rs[kRsFogTableMode];
    if (vertex_program && table_mode == kFogTableNone) {
        c.fog[0] = 1.0f;
        c.fog[1] = 0.0f;
        c.fog[2] = 0.0f;
        c.fog[3] = kFogTableLinear;
        return;
    }
    c.fog[0] = Bits(s.rs[kRsFogStart]);
    c.fog[1] = Bits(s.rs[kRsFogEnd]);
    c.fog[2] = Bits(s.rs[kRsFogDensity]);
    c.fog[3] = float(table_mode);
}

void FillSurfaceConstants(const State& s, GuestExtent size, DrawConstants& c) {
    c.surf[0] = float(size.w);
    c.surf[1] = float(size.h);
    c.surf[2] = 1.0f / s.zscale;
    Color(s.rs[kRsFogColor], c.fog_color);
    c.misc[0] = float(SimpleState(s, kMAlphaRef) & 0xFF);
    c.misc[1] = SimpleState(s, kMAlphaTestEnable) ? float(CmpIndex(SimpleState(s, kMAlphaFunc))) : kAlphaTestAlways;
    c.misc[2] = float(CmpIndex(s.rs[kRsShadowFunc]));
    c.misc[3] = s.rs[kRsSpecularEnable] ? 1.0f : 0.0f;
}

void FillCombinerConstants(const State& s, DrawConstants& c) {
    for (int i = 0; i < 8; i++) {
        Color(SimpleState(s, kMPsConstant0 + i * 4), c.pc[i * 2]);
        Color(SimpleState(s, kMPsConstant1 + i * 4), c.pc[i * 2 + 1]);
    }
    Color(SimpleState(s, kMPsFinalConstant0), c.pc[16]);
    Color(SimpleState(s, kMPsFinalConstant1), c.pc[17]);
    Color(s.rs[kRsTextureFactor], c.tfactor);
}

constexpr float kGlowBoxTexels = 3.0f;
constexpr float kBlurTapTexels = 7.0f / 6;

void FillTexelConstants(int i, const TextureDesc& d, const DrawTextures& stages, DrawConstants& c) {
    c.texsize[i][0] = float(d.width);
    c.texsize[i][1] = float(d.height);
    // the console samples pixels at integer positions, the host half a pixel on: targets are read half a texel back
    if (!stages.rt[i]) return;
    float k = stages.k[i];
    c.texsize[i][2] = -0.5f / (float(d.width) * k);
    c.texsize[i][3] = -0.5f / (float(d.height) * k);
    if (stages.wide[i] > 0) {
        bool blur = stages.blur[i];
        float fw = blur ? kBlurTapTexels : kGlowBoxTexels;
        float per_texel = blur ? 1.0f : 0.5f;
        float n = std::ceil(fw * stages.wide[i] * per_texel);
        c.spread[i][0] = fw / (n * float(d.width) * k);
        c.spread[i][1] = fw / (n * float(d.height) * k);
        c.spread[i][2] = n;
    }
}

void FillTextureConstants(const State& s, const DrawTextures& stages, DrawConstants& c) {
    for (int i = 0; i < 4; i++) {
        TextureDesc d;
        if (s.textures[size_t(i)] && Describe(s.textures[size_t(i)], d)) FillTexelConstants(i, d, stages, c);
        const auto& t = s.tss[size_t(i)];
        c.bump[i][0] = Bits(t[kTssBumpEnvMat00]);
        c.bump[i][1] = Bits(t[kTssBumpEnvMat01]);
        c.bump[i][2] = Bits(t[kTssBumpEnvMat10]);
        c.bump[i][3] = Bits(t[kTssBumpEnvMat11]);
        c.bumpl[i][0] = Bits(t[kTssBumpEnvLScale]);
        c.bumpl[i][1] = Bits(t[kTssBumpEnvLOffset]);
    }
}

void MatMul(const float* x, const float* y, float* o) {
    for (int r = 0; r < 4; r++)
        for (int col = 0; col < 4; col++)
            o[r * 4 + col] = x[r * 4] * y[col] + x[r * 4 + 1] * y[4 + col] + x[r * 4 + 2] * y[8 + col] +
                             x[r * 4 + 3] * y[12 + col];
}

void Transpose(const float* m, float* o) {
    for (int r = 0; r < 4; r++)
        for (int col = 0; col < 4; col++) o[col * 4 + r] = m[r * 4 + col];
}

void InverseTranspose3x3(const float* wv, float* n) {
    float det = wv[0] * (wv[5] * wv[10] - wv[6] * wv[9]) - wv[1] * (wv[4] * wv[10] - wv[6] * wv[8]) +
                wv[2] * (wv[4] * wv[9] - wv[5] * wv[8]);
    float id = det != 0 ? 1.0f / det : 0.0f;
    for (int r = 0; r < 3; r++)
        for (int col = 0; col < 3; col++) {
            int r1 = (r + 1) % 3, r2 = (r + 2) % 3, c1 = (col + 1) % 3, c2 = (col + 2) % 3;
            n[r * 4 + col] = (wv[r1 * 4 + c1] * wv[r2 * 4 + c2] - wv[r1 * 4 + c2] * wv[r2 * 4 + c1]) * id;
        }
}

constexpr size_t kTsView = 2;
constexpr size_t kTsProjection = 3;
constexpr size_t kTsTexture0 = 16;
constexpr size_t kTsWorld = 256;

void FillTransformConstants(const State& s, GuestExtent size, DrawConstants& c) {
    float wv[16], wvp[16];
    MatMul(s.transforms[kTsWorld].data(), s.transforms[kTsView].data(), wv);
    MatMul(wv, s.transforms[kTsProjection].data(), wvp);
    const auto& v = s.viewport;
    float sx = float(v.w) / size.w, sy = float(v.h) / size.h;
    float ox = (2.0f * v.x + v.w) / size.w - 1, oy = 1 - (2.0f * v.y + v.h) / size.h;
    float vpm[16] = {sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, v.max_z - v.min_z, 0, ox, oy, v.min_z, 1};
    float full[16];
    MatMul(wvp, vpm, full);
    Transpose(full, c.wvp);
    Transpose(wv, c.wv);
    float n[16] = {};
    InverseTranspose3x3(wv, n);
    Transpose(n, c.wvn);
    for (int t = 0; t < 4; t++) Transpose(s.transforms[kTsTexture0 + size_t(t)].data(), c.tm[t]);
}

void LightToView(const std::array<uint32_t, 26>& l, const float* vm, float (&o)[7][4]) {
    float f[26];
    std::memcpy(f, l.data(), sizeof(f));
    for (int j = 0; j < 4; j++) {
        o[0][j] = f[1 + j];
        o[1][j] = f[5 + j];
        o[2][j] = f[9 + j];
    }
    const float* pos = f + 13;
    const float* dir = f + 16;
    for (int j = 0; j < 3; j++) {
        o[3][j] = pos[0] * vm[j] + pos[1] * vm[4 + j] + pos[2] * vm[8 + j] + vm[12 + j];
        o[4][j] = dir[0] * vm[j] + dir[1] * vm[4 + j] + dir[2] * vm[8 + j];
    }
    o[3][3] = f[19];
    o[4][3] = f[20];
    o[5][0] = f[21];
    o[5][1] = f[22];
    o[5][2] = f[23];
    o[6][0] = std::cos(f[24] * 0.5f);
    o[6][1] = std::cos(f[25] * 0.5f);
}

void FillLightConstants(const State& s, const FixedKey& fk, DrawConstants& c) {
    std::memcpy(c.mat, s.material.data(), sizeof(float) * 16);
    c.mat[4][0] = s.material[16];
    Color(s.rs[kRsAmbient], c.amb);
    const float* vm = s.transforms[kTsView].data();
    for (int i = 0; i < 8; i++)
        if ((fk.types >> (i * 2)) & 3) LightToView(s.lights[size_t(i)], vm, c.lt[i]);
}

struct GuestRect {
    int32_t left, top, right, bottom;
};
GuestRect ReadGuestRect(uint32_t rects, uint32_t i) {
    uint32_t r = rects + i * 16;
    return {int32_t(kernel::R32(r)), int32_t(kernel::R32(r + 4)), int32_t(kernel::R32(r + 8)),
            int32_t(kernel::R32(r + 12))};
}
void ReadGuestPoint(uint32_t points, uint32_t i, int32_t& x, int32_t& y) {
    x = int32_t(kernel::R32(points + i * 8));
    y = int32_t(kernel::R32(points + i * 8 + 4));
}
uint32_t ScaleUnsigned(int32_t v, float sc) { return uint32_t(uint32_t(v) * sc); }

// A guest allocation can back multiple XDK surface descriptors at different extents. Keep each
// view independently; keying only by the data address makes render-target sizes evict each other.
uint64_t TargetMapKey(uint32_t data, bool depth, uint32_t w, uint32_t h) {
    return (uint64_t(data) << 32) | (uint64_t(depth) << 31) | (uint64_t(w & 0x7FFF) << 16) | (h & 0xFFFF);
}

// ---- layouts: the stages and accesses an image's layout stands for

void LayoutMasks(VkImageLayout l, VkPipelineStageFlags& st, VkAccessFlags& ac) {
    switch (l) {
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            st = VK_PIPELINE_STAGE_TRANSFER_BIT;
            ac = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            st = VK_PIPELINE_STAGE_TRANSFER_BIT;
            ac = VK_ACCESS_TRANSFER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            st = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
            ac = VK_ACCESS_SHADER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            st = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            ac = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            st = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            ac = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            st = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            ac = 0;
            break;
        default:
            st = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            ac = 0;
            break;
    }
}

bool HasStencil(VkFormat f) {
    return f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_D16_UNORM_S8_UINT;
}

class VulkanBackend final : public Backend {
public:
    ~VulkanBackend() override = default;
    bool Init(uint32_t width, uint32_t height) override;
    void Clear(const State& s, uint32_t count, const uint8_t* rects, uint32_t flags, uint32_t color, float z,
               uint32_t stencil) override;
    void Draw(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) override;
    void Copy(const Surface& src, const Surface& dst, uint32_t count, uint32_t src_rects, uint32_t dst_points) override;
    void Present(const Surface& front) override;
    void Sync(uint32_t data, uint32_t size) override;
    void SyncLate(uint32_t data) override;
    bool DeliverLate(uint32_t data) override;
    void RecordLate(uint32_t data) override;
    void Flush() override { Submit(true, false); }
    void SetGamma(const uint8_t* ramp) override {
        std::memcpy(ramp_, ramp, sizeof(ramp_));
        ramp_dirty_ = true;
    }
    void WaitGpuFrame(uint64_t n);

private:
    struct Frame {
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        Buffer upload;
        VkDeviceSize used = 0;
        uint64_t value = 0;  // the timeline's value once the GPU is past this frame
        std::vector<Image> images;
        std::vector<Buffer> buffers;
        std::vector<VkDescriptorPool> pools;
        size_t pool_at = 0;  // the descriptor pool sets come from
        std::unordered_map<uint64_t, VkDescriptorSet> sets;
        VkSemaphore acquire = VK_NULL_HANDLE;
        bool acquire_pending = false;
    };
    struct Alloc {
        VkBuffer buf;
        VkDeviceSize off;
        uint8_t* cpu;
    };
    struct Late {
        Buffer rb[kFrames];
        Image down;
        uint64_t fence[kFrames] = {};
        uint64_t frame[kFrames] = {};
        uint64_t delivered = 0;
        uint32_t gw = 0, gh = 0, next = 0;
    };

    Frame& CurrentFrame() { return frames_[frame_ % kFrames]; }
    VkCommandBuffer Cb() { return CurrentFrame().cb; }

    // Init, in order
    bool CreateInstance();
    bool CreateSurface();
    bool PickDevice();
    bool CreateDevice();
    bool CreateSwapchain();
    void DestroySwapchain();
    bool CreateLayouts();
    bool CreateFrames();
    bool CreateFixedImages();
    bool InitUi();

    // memory and objects
    bool CreateImage(Image& im, VkFormat fmt, uint32_t w, uint32_t h, uint32_t d, uint32_t layers, uint32_t levels,
                     VkImageUsageFlags usage, VkImageViewType vt, bool cube = false);
    bool CreateBuffer(Buffer& b, VkDeviceSize size, VkBufferUsageFlags usage, bool host);
    void Destroy(Image& im);
    void Destroy(Buffer& b);
    void Retire(Image& im) {
        if (rend_color_ == &im || rend_depth_ == &im) EndRender();
        if (im) CurrentFrame().images.push_back(im);
        im = {};
    }
    void Retire(Buffer& b) {
        if (b.buf) CurrentFrame().buffers.push_back(b);
        b = {};
    }
    Alloc Upload(VkDeviceSize size, VkDeviceSize align = 16);
    Alloc Constants(const void* data, size_t bytes);

    // command recording
    void Begin();
    void Submit(bool wait, bool present);
    void WaitValue(uint64_t v);
    uint64_t Completed();
    VkFence SubmitFence(uint64_t value);
    void PollFences();
    void ResetFrame(Frame& f);
    void Transition(Image& im, VkImageLayout to);
    void BeginRender(Image* color, Image* depth);
    void EndRender();
    // a clear that covers a whole attachment becomes the load op of the pass it opens, so a tile GPU starts
    // from the clear value instead of loading the stale tile from DRAM
    uint32_t clear_on_begin_ = 0;
    VkClearColorValue clear_color_{};
    VkClearDepthStencilValue clear_ds_{};
    // without dynamic rendering: render passes that only load and store, and framebuffers per pair of views.
    // clear is a mask: 1 clears the colour, 2 the depth and stencil; only the load op differs, so a framebuffer
    // made for one mask still fits the others.
    VkRenderPass Pass(VkFormat color, VkFormat depth, uint32_t clear = 0);
    VkFramebuffer Framebuffer(VkFormat color_fmt, VkFormat depth_fmt, VkImageView color, VkImageView depth,
                              uint32_t w, uint32_t h);
    void DropFramebuffers(VkImageView v);
    void SetViewport(float x, float y, float w, float h);
    void SetScissor(int32_t x0, int32_t y0, int32_t x1, int32_t y1);
    VkDescriptorSet Set(const VkImageView views[4], const VkSampler samplers[4]);
    void BindSet(VkDescriptorSet set, VkDeviceSize vc, VkDeviceSize dc);

    // shaders and pipelines
    VkShaderModule Module(const std::string& src, const char* entry, vkshader::Stage stage, uint64_t& hash);
    VkPipeline FullScreenPipeline(const char* hlsl, const char* ps_entry, VkFormat color, VkFormat depth, bool over);
    VkPipeline EnsurePipeline(const PsoKey& k, VkShaderModule vs, VkShaderModule ps);
    VkSampler Sampler(const VkSamplerCreateInfo& ci);
    VkSampler StageSampler(const State& s, int i, const DrawTextures& stages, bool clamp);

    // targets
    Target* FindTarget(const Surface& s, bool depth);
    Target* FindTargetByData(uint32_t data, bool depth);
    Target* EnsureTarget(const Surface& s, bool depth);
    void ReleaseTarget(std::unordered_map<uint64_t, Target>::iterator it);
    void UploadTarget(Target& t, const Surface& s);
    void UploadTarget(Target& t, const TextureDesc& d);
    bool PixelImage(Image& im, const std::vector<uint8_t>& px, uint32_t pitch, uint32_t w, uint32_t h);
    void Blit(Image& dst, const VkRect2D& to, Image& src, const float uvrect[4], const BlitOptions& o);
    Image* Feedback(Target& t);
    bool DepthCopy(const State& s);
    bool Shrink(Image& src, uint32_t w, uint32_t h, Image& down, uint32_t taps = 1);
    Image* CopySource(Target& t, Image& down);
    bool ReadBack(Image& im, uint32_t w, uint32_t h, std::vector<uint8_t>& out);
    bool CaptureIndexedColor(Target& target, const char* filename, std::vector<uint8_t>& pixels);
    void TraceImage(Image& image, const std::string& name);
    void TraceSkyBand(Image& image, const std::string& name, uint32_t relative);
    void TraceTopStrip(Image& image, uint32_t relative, uint32_t rt);
    bool TraceDepth(Target& target, const std::string& name);

    // draws
    void EnsureVertexShader(const State& s, const Assembled& a, DrawShaders& sh);
    void EnsurePixelShader(const PixelKey& pk, DrawShaders& sh);
    bool BindStageTextures(const State& s, DrawTextures& stages);
    VkImageView BindTexture(const State& s, int stage, DrawTextures& stages);
    bool GlowComposite(const State& s, const DrawTextures& stages) const;
    VkImageView BindTargetTexture(const State& s, int stage, const TextureDesc& d, Target& t, bool box,
                                  DrawTextures& stages);
    VkImageView BindGuestTexture(const State& s, int stage, const TextureDesc& d, const FormatInfo& fi,
                                 DrawTextures& stages);
    bool CreateTexture(Tex& tx, const TextureDesc& d, const FormatInfo& fi, uint64_t key);
    void UploadTexture(Tex& tx, const TextureDesc& d, const FormatInfo& fi, const uint8_t* p, const uint32_t* colours);
    void BuildPipelineKey(const State& s, const Assembled& a, const DrawShaders& sh, PsoKey& k);
    void KeyDepthStencil(const State& s, const Assembled& a, PsoKey& k);
    void ClearTarget(Target& t, const State& s, uint32_t count, const uint8_t* rects, uint32_t flags, uint32_t color,
                     float z, uint32_t stencil);

    // present
    bool AcquireImage(uint32_t& index);
    void DrawFrame(Target* t, uint32_t index);
    bool UpdateVideoOverlay();
    void UpdatePerf();
    void RenderUi();
    void UploadGammaRamp();
    void UploadSkin();
    void ReleaseOldTextures();
    void ApplySettings();
    void SavePipelineCache();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    uint32_t surface_epoch_ = 0;
    VkPhysicalDevice pd_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props_{};
    VkPhysicalDeviceFeatures feats_{};
    uint32_t api_ = VK_API_VERSION_1_1;
    uint32_t qfam_ = 0;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::vector<const char*> dev_exts_;
    bool bc_ = false;
    VkFormat depth_fmt_ = VK_FORMAT_D24_UNORM_S8_UINT;
    bool depth_linear_ = false;
    bool r32f_linear_ = false;
    Allocator alloc_;
    PFN_vkCmdBeginRenderingKHR begin_rendering_ = nullptr;
    PFN_vkCmdEndRenderingKHR end_rendering_ = nullptr;
    PFN_vkWaitSemaphoresKHR wait_semaphores_ = nullptr;
    PFN_vkGetSemaphoreCounterValueKHR counter_value_ = nullptr;
    bool dynamic_ = true;  // dynamic rendering; Vulkan 1.1 drivers (older Adrenos) only have render passes
    bool timeline_ok_ = true;  // without it a fence per submit stands in for the timeline's values
    struct Pending {
        uint64_t value;
        VkFence fence;
    };
    std::mutex fence_lock_;  // the game thread waits on frames too
    std::vector<Pending> pending_;
    std::vector<VkFence> spare_fences_;
    uint64_t fenced_ = 0;  // the highest value whose fence signalled
    std::unordered_map<uint64_t, VkRenderPass> passes_;
    struct Fb {
        VkImageView color, depth;
        VkFramebuffer fb;
    };
    std::unordered_map<uint64_t, Fb> framebuffers_;

    VkSwapchainKHR swap_ = VK_NULL_HANDLE;
    VkFormat swap_fmt_ = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D swap_extent_{};
    std::vector<VkImage> swap_images_;
    std::vector<VkImageView> swap_views_;
    std::vector<VkSemaphore> swap_done_;
    bool vsync_ = true;
    bool swap_stale_ = false;
    bool surface_lost_ = false;  // the driver said the window is gone: the surface is made again too
    bool swap_transfer_src_ = false;
    uint32_t surface_failures_ = 0;  // presents in a row without a surface, for the log
    uint32_t present_index_ = 0;  // the image the frame's submit signals the present of

    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipelineCache cache_ = VK_NULL_HANDLE;
    size_t cache_saved_ = 0;
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    std::atomic<uint64_t> value_{0};
    Frame frames_[kFrames];
    uint64_t frame_ = 0;

    // the rendering under way: its attachments
    bool rendering_ = false;
    Image* rend_color_ = nullptr;
    Image* rend_depth_ = nullptr;

    std::unordered_map<uint64_t, VkShaderModule> modules_;
    std::unordered_map<uint64_t, uint64_t> shader_src_;
    std::unordered_map<uint64_t, VkPipeline> pipelines_;
    std::unordered_map<uint64_t, VkSampler> samplers_;
    std::unordered_map<uint64_t, Target> targets_;
    std::unordered_map<uint64_t, uint64_t> presented_serial_;
    uint64_t last_present_key_ = 0;
    uint64_t target_use_serial_ = 0;
    std::unordered_map<uint64_t, Tex> textures_;
    Image lut_;
    Image video_overlay_;
    uint32_t video_overlay_w_ = 0, video_overlay_h_ = 0;
    Image dummy_[3];  // 2D, 3D, cube
    VkSampler linear_ = VK_NULL_HANDLE, point_ = VK_NULL_HANDLE;
    VkDescriptorPool ui_pool_ = VK_NULL_HANDLE;
    bool ui_ = false;
    uint8_t ramp_[768] = {};
    bool ramp_dirty_ = false;
    Image skin_;  // the touch controls' button skins (touch.cpp), made once
    bool skin_tried_ = false;
    bool gamma_ = false;
    Target* draw_rt_ = nullptr;
    Target* draw_ds_ = nullptr;
    uint32_t guest_w_ = 640, guest_h_ = 480;
    float scale_ = 1.0f;
    Assembled asm_;
    std::unordered_map<uint32_t, Late> late_;
    std::mutex late_lock_;
    std::atomic<uint64_t> presents_seen_{0};
    std::atomic<uint64_t> present_value_[kFrames + 1] = {};
    uint64_t presents_ = 0;
    std::filesystem::path trace_dir_;
    uint32_t trace_draw_ = 0, trace_samples_ = 0, trace_mesh_samples_ = 0;
    uint32_t trace_scene_draw_ = 0;
    uint32_t trace_relative_ = 0;  // draw index within the main target, for the sky probe
    bool trace_sky_window_ = false;
    bool trace_strip_ = false;      // sample the top of the scene after every draw (sky-search probe)
    uint32_t trace_strip_count_ = 0;
    uint32_t trace_strip_rt_ = 0;
    std::vector<uint8_t> trace_prev_pixels_;  // last scene sample, to count a draw's actual coverage
    uint32_t trace_prev_rt_ = 0;
    uint32_t trace_sky_saved_ = 0;
    double trace_sky_mean_ = -1.0;
    uint32_t trace_sky_on_ = 0;
    Target* trace_last_target_ = nullptr;
    std::set<uint32_t> trace_depth_captured_;
    double fps_ = 0;
    // diagnostic overlay (CPU/GPU MHz, battery, temperature, FPS); only when /sdcard/forza_diag_perf exists
    bool diag_perf_ = false;
    int diag_perf_freq_ = 0;
    uint32_t diag_perf_cpu_[8] = {0};
    uint32_t diag_perf_gpu_ = 0;
    uint32_t diag_perf_batt_ = 0;
    uint32_t diag_perf_temp_ = 0;
    double diag_perf_fps_ = 0;
    // where a frame's time goes: recording the frame's commands vs blocked on the GPU (marker /sdcard/forza_diag_perf2)
    bool perf2_ = false;
    uint32_t perf2_draws_ = 0;
    uint64_t perf2_bytes_ = 0;
    uint32_t perf2_tex_ = 0;
    uint32_t perf2_tex_new_ = 0;
    uint32_t perf2_feedback_ = 0;
    uint32_t perf2_pass_breaks_ = 0;
    uint32_t perf2_const_reuse_ = 0;
    uint32_t perf2_frames_ = 0;
    double perf2_record_ms_ = 0;
    double perf2_wait_ms_ = 0;
    double perf2_frame_ms_ = 0;
double perf2_drawframe_ms_ = 0;
uint32_t perf2_cube_blits_ = 0;
    std::chrono::steady_clock::time_point perf2_last_{};
    std::chrono::steady_clock::time_point perf2_t_ = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point fps_t_ = std::chrono::steady_clock::now();
    uint64_t fps_n_ = 0;

    // Fase 2: Pass-scoped state deduplication
    VkPipeline cur_pipe_ = VK_NULL_HANDLE;
    VkDescriptorSet cur_set_ = VK_NULL_HANDLE;
    uint32_t cur_vc_off_ = 0xFFFFFFFF;
    uint32_t cur_dc_off_ = 0xFFFFFFFF;
    VkViewport cur_vp_{};
    VkRect2D cur_sc_{};
    float cur_blend_constants_[4] = {-999.0f, -999.0f, -999.0f, -999.0f};
    uint32_t cur_stencil_ref_ = 0xFFFFFFFF;
    uint32_t cur_vb_count_ = 0;
    VkBuffer cur_vbufs_[16]{};
    VkDeviceSize cur_voffs_[16]{};
    VkBuffer cur_ib_ = VK_NULL_HANDLE;
    VkDeviceSize cur_ib_off_ = 0xFFFFFFFF;
    VkIndexType cur_ib_type_ = VK_INDEX_TYPE_MAX_ENUM;
    void ResetPassState();

    // Fase 2: Constant buffer deduplication
    struct ConstMemoEntry {
        uint64_t hash = 0;
        uint64_t present = 0;
        uint32_t bytes = 0;
        Alloc alloc{};
        uint8_t data[3072]{};
    };
    static constexpr size_t kConstMemoEntries = 64;
    ConstMemoEntry const_memo_[kConstMemoEntries]{};
};

#define VK_CHECK(x)                                                    \
    do {                                                               \
        VkResult _r = (x);                                             \
        if (_r != VK_SUCCESS) {                                        \
            LOGE("vulkan: %s failed (%d)", #x, int(_r));               \
            return false;                                              \
        }                                                              \
    } while (0)

// ---- Init

bool VulkanBackend::Init(uint32_t width, uint32_t height) {
    guest_w_ = width;
    guest_h_ = height;
    LOGR("vulkan: init begin guest=%ux%u", width, height);
    vkshader::Init(app::ExeDir() / "shadercache");
    if (!CreateInstance()) { LOGE("vulkan: init failed at instance"); return false; }
    if (!CreateSurface()) { LOGE("vulkan: init failed at surface"); return false; }
    if (!PickDevice()) { LOGE("vulkan: init failed at device selection"); return false; }
    if (!CreateDevice()) { LOGE("vulkan: init failed at device creation"); return false; }
    alloc_.Init(pd_, dev_);
    if (!CreateLayouts()) { LOGE("vulkan: init failed at layouts"); return false; }
    if (!CreateFrames()) { LOGE("vulkan: init failed at frames"); return false; }
    if (!CreateSwapchain()) { LOGE("vulkan: init failed at swapchain"); return false; }
    if (!CreateFixedImages()) { LOGE("vulkan: init failed at fixed images"); return false; }
    Begin();
    ui_ = InitUi();
    scale_ = app::RenderScale(guest_h_);
    LOGI("vulkan: %ux%u guest, %ux%u screen, render scale %.3f", width, height, swap_extent_.width,
         swap_extent_.height, scale_);
    return true;
}

bool VulkanBackend::CreateInstance() {
    uint32_t n = 0;
    const char* const* sdl = SDL_Vulkan_GetInstanceExtensions(&n);
    std::vector<const char*> exts(sdl, sdl + n);
    uint32_t loader = VK_API_VERSION_1_1;
    vkEnumerateInstanceVersion(&loader);
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = app::kGame.log;
    ai.pEngineName = "XboxXBE-Recomp";
    ai.apiVersion = std::min<uint32_t>(loader, VK_API_VERSION_1_3);
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &ai;
    ci.enabledExtensionCount = uint32_t(exts.size());
    ci.ppEnabledExtensionNames = exts.data();
    VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));
    return true;
}

// the epoch is taken only once the surface is made: until then each present tries again
bool VulkanBackend::CreateSurface() {
    uint32_t epoch = host::SurfaceEpoch();
    // Android has not handed the window back yet (resuming, or a surface change): wait for it without asking SDL,
    // which only fails, and log the outage once, not every present
    SDL_PropertiesID props = SDL_GetWindowProperties(host::Window());
    if (!SDL_GetPointerProperty(props, SDL_PROP_WINDOW_ANDROID_WINDOW_POINTER, nullptr)) {
        if (surface_failures_++ == 0) LOGI("vulkan: waiting for Android's window");
        return false;
    }
    if (!SDL_Vulkan_CreateSurface(host::Window(), instance_, nullptr, &surface_)) {
        surface_ = VK_NULL_HANDLE;
        if (surface_failures_++ == 0) LOGE("vulkan: no surface: %s", SDL_GetError());
        return false;
    }
    if (surface_failures_) LOGI("vulkan: surface made after %u tries", surface_failures_ + 1);
    surface_failures_ = 0;
    surface_epoch_ = epoch;
    surface_lost_ = false;
    return true;
}

bool HasExtension(const std::vector<VkExtensionProperties>& list, const char* name) {
    return std::any_of(list.begin(), list.end(), [&](const VkExtensionProperties& e) { return !std::strcmp(e.extensionName, name); });
}

bool VulkanBackend::PickDevice() {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance_, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(instance_, &n, pds.data());
    // rejections are reported, not warned: release logs drop warnings and these are what a bug report needs
    if (pds.empty()) LOGE("vulkan: the driver lists no GPU");
    for (VkPhysicalDevice pd : pds) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pd, &p);
        LOGR("GPU: %s, Vulkan %u.%u.%u, driver %08X", p.deviceName, VK_API_VERSION_MAJOR(p.apiVersion),
             VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion), p.driverVersion);
        if (p.apiVersion < VK_API_VERSION_1_1) {
            LOGE("vulkan: %s is below Vulkan 1.1", p.deviceName);
            continue;
        }
        uint32_t en = 0;
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, nullptr);
        std::vector<VkExtensionProperties> ext(en);
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, ext.data());
        uint32_t api = std::min<uint32_t>(p.apiVersion, VK_API_VERSION_1_3);
        bool swapchain = HasExtension(ext, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        bool dynamic = api >= VK_API_VERSION_1_3 || HasExtension(ext, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
        if (!swapchain) {
            LOGE("vulkan: %s lacks swapchain", p.deviceName);
            continue;
        }
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qs.data());
        for (uint32_t q = 0; q < qn; q++) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(pd, q, surface_, &present);
            if (!(qs[q].queueFlags & VK_QUEUE_GRAPHICS_BIT) || !present) continue;
            pd_ = pd;
            props_ = p;
            api_ = api;
            qfam_ = q;
            dev_exts_ = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
            auto want = [&](const char* name) {
                if (HasExtension(ext, name)) dev_exts_.push_back(name);
            };
            if (api < VK_API_VERSION_1_3) {
                want(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
                want(VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME);
                want(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
            }
            dynamic_ = dynamic;
            // the 1.1 drivers' timeline extension lost the device on the first submit (Mali-G76): fences there
            timeline_ok_ = api >= VK_API_VERSION_1_2;
            if (!dynamic_) LOGR("vulkan: no dynamic rendering, using render passes");
            if (!timeline_ok_) LOGR("vulkan: Vulkan 1.1, using fences for timeline semaphores");
            return true;
        }
        LOGE("vulkan: %s has no graphics queue that can present", p.deviceName);
    }
    LOGE("vulkan: no GPU with Vulkan 1.1 that can present");
    return false;
}

bool VulkanBackend::CreateDevice() {
    VkPhysicalDeviceTimelineSemaphoreFeaturesKHR ts{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR};
    VkPhysicalDeviceDynamicRenderingFeaturesKHR dr{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR};
    // the chain holds only what is used: a 1.1 driver may not know the rest
    auto chain = [&]() -> void* {
        ts.pNext = nullptr;
        dr.pNext = timeline_ok_ ? &ts : nullptr;
        return dynamic_ ? (void*)&dr : timeline_ok_ ? (void*)&ts : nullptr;
    };
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, chain()};
    vkGetPhysicalDeviceFeatures2(pd_, &f2);
    if (timeline_ok_ && !ts.timelineSemaphore) {
        LOGR("vulkan: timeline semaphores not supported, using fences");
        timeline_ok_ = false;
    }
    if (dynamic_ && !dr.dynamicRendering) {
        LOGR("vulkan: dynamic rendering listed but not supported, using render passes");
        dynamic_ = false;
    }
    feats_ = {};
    feats_.samplerAnisotropy = f2.features.samplerAnisotropy;
    feats_.fillModeNonSolid = f2.features.fillModeNonSolid;
    feats_.textureCompressionBC = f2.features.textureCompressionBC;
    feats_.depthClamp = f2.features.depthClamp;
    bc_ = feats_.textureCompressionBC;
    VkPhysicalDeviceFeatures2 on{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, chain()};
    on.features = feats_;
    dr.dynamicRendering = VK_TRUE;
    ts.timelineSemaphore = VK_TRUE;
    float prio = 1;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = qfam_;
    qi.queueCount = 1;
    qi.pQueuePriorities = &prio;
    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &on};
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &qi;
    ci.enabledExtensionCount = uint32_t(dev_exts_.size());
    ci.ppEnabledExtensionNames = dev_exts_.data();
    VK_CHECK(vkCreateDevice(pd_, &ci, nullptr, &dev_));
    vkGetDeviceQueue(dev_, qfam_, 0, &queue_);
    bool core13 = api_ >= VK_API_VERSION_1_3, core12 = api_ >= VK_API_VERSION_1_2;
    begin_rendering_ = reinterpret_cast<PFN_vkCmdBeginRenderingKHR>(
        vkGetDeviceProcAddr(dev_, core13 ? "vkCmdBeginRendering" : "vkCmdBeginRenderingKHR"));
    end_rendering_ = reinterpret_cast<PFN_vkCmdEndRenderingKHR>(
        vkGetDeviceProcAddr(dev_, core13 ? "vkCmdEndRendering" : "vkCmdEndRenderingKHR"));
    wait_semaphores_ = reinterpret_cast<PFN_vkWaitSemaphoresKHR>(
        vkGetDeviceProcAddr(dev_, core12 ? "vkWaitSemaphores" : "vkWaitSemaphoresKHR"));
    counter_value_ = reinterpret_cast<PFN_vkGetSemaphoreCounterValueKHR>(
        vkGetDeviceProcAddr(dev_, core12 ? "vkGetSemaphoreCounterValue" : "vkGetSemaphoreCounterValueKHR"));
    if ((dynamic_ && (!begin_rendering_ || !end_rendering_)) || (timeline_ok_ && (!wait_semaphores_ || !counter_value_))) {
        LOGE("vulkan: device functions missing");
        return false;
    }
    // the depth buffer: 24 bit with stencil as on the console where the GPU has it
    auto usable = [&](VkFormat f) {
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(pd_, f, &fp);
        VkFormatFeatureFlags need = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        return (fp.optimalTilingFeatures & need) == need;
    };
    depth_fmt_ = usable(VK_FORMAT_D24_UNORM_S8_UINT) ? VK_FORMAT_D24_UNORM_S8_UINT : VK_FORMAT_D32_SFLOAT_S8_UINT;
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(pd_, depth_fmt_, &fp);
    depth_linear_ = fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    vkGetPhysicalDeviceFormatProperties(pd_, VK_FORMAT_R32_SFLOAT, &fp);
    r32f_linear_ = fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    LOGI("vulkan: depth %s, BC textures %s", depth_fmt_ == VK_FORMAT_D24_UNORM_S8_UINT ? "D24S8" : "D32S8",
         bc_ ? "on the GPU" : "decoded on the CPU");
    return true;
}

bool VulkanBackend::CreateSwapchain() {
    if (!surface_) return false;
    VkSurfaceCapabilitiesKHR caps;
    VkResult cr = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd_, surface_, &caps);
    if (cr != VK_SUCCESS) {
        if (cr == VK_ERROR_SURFACE_LOST_KHR) surface_lost_ = true;
        LOGE("vulkan: surface capabilities failed (%d)", int(cr));
        return false;
    }
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd_, surface_, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd_, surface_, &n, formats.data());
    if (formats.empty()) return false;
    VkSurfaceFormatKHR chosen = formats[0];
    for (const auto& f : formats)
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = f;
            break;
        }
    // the window's own size, landscape; Android's compositor turns the picture with the screen
    int w, h;
    host::WindowPixels(w, h);
    if (w <= 0 || h <= 0) {
        w = int(caps.currentExtent.width);
        h = int(caps.currentExtent.height);
    }
    if (w <= 0 || h <= 0) return false;
    vsync_ = app::Get().vsync;
    uint32_t pn = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd_, surface_, &pn, nullptr);
    std::vector<VkPresentModeKHR> modes(pn);
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd_, surface_, &pn, modes.data());
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    if (!vsync_) {
        for (VkPresentModeKHR m : {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR})
            if (std::find(modes.begin(), modes.end(), m) != modes.end()) {
                mode = m;
                break;
            }
    }
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface_;
    ci.minImageCount = std::max(caps.minImageCount, 3u);
    if (caps.maxImageCount) ci.minImageCount = std::min(ci.minImageCount, caps.maxImageCount);
    ci.imageFormat = chosen.format;
    ci.imageColorSpace = chosen.colorSpace;
    ci.imageExtent = {uint32_t(w), uint32_t(h)};
    ci.imageArrayLayers = 1;
    swap_transfer_src_ = (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    (swap_transfer_src_ ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : VkImageUsageFlags(0));
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
                          ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
                          : caps.currentTransform;
    ci.compositeAlpha = (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
                            ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                            : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    ci.presentMode = mode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = swap_;
    // retiring the old chain gives its images back to the window: the GPU must be done drawing them
    if (swap_) vkQueueWaitIdle(queue_);
    VkSwapchainKHR fresh = VK_NULL_HANDLE;
    VkResult sr = vkCreateSwapchainKHR(dev_, &ci, nullptr, &fresh);
    if (sr != VK_SUCCESS) {
        if (sr == VK_ERROR_SURFACE_LOST_KHR || sr == VK_ERROR_NATIVE_WINDOW_IN_USE_KHR) surface_lost_ = true;
        LOGE("vulkan: swap chain %dx%d failed (%d)", w, h, int(sr));
        return false;
    }
    DestroySwapchain();
    swap_ = fresh;
    swap_fmt_ = chosen.format;
    swap_extent_ = ci.imageExtent;
    uint32_t count = 0;
    vkGetSwapchainImagesKHR(dev_, swap_, &count, nullptr);
    swap_images_.resize(count);
    vkGetSwapchainImagesKHR(dev_, swap_, &count, swap_images_.data());
    for (VkImage im : swap_images_) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = im;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = swap_fmt_;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView v = VK_NULL_HANDLE;
        vkCreateImageView(dev_, &vi, nullptr, &v);
        swap_views_.push_back(v);
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkSemaphore s = VK_NULL_HANDLE;
        vkCreateSemaphore(dev_, &si, nullptr, &s);
        swap_done_.push_back(s);
    }
    swap_stale_ = false;
    LOGI("vulkan: swap chain %ux%u, %u images, %s", w, h, count, vsync_ ? "vsync" : "no vsync");
    LOGR("vulkan: swapchain-transfer-src=%u", swap_transfer_src_);
    return true;
}

// the old chain's views and semaphores; the chain itself when it is not being replaced
void VulkanBackend::DestroySwapchain() {
    if (!swap_) return;
    vkQueueWaitIdle(queue_);
    for (VkImageView v : swap_views_) {
        DropFramebuffers(v);
        vkDestroyImageView(dev_, v, nullptr);
    }
    for (VkSemaphore s : swap_done_) vkDestroySemaphore(dev_, s, nullptr);
    swap_views_.clear();
    swap_done_.clear();
    swap_images_.clear();
    vkDestroySwapchainKHR(dev_, swap_, nullptr);
    swap_ = VK_NULL_HANDLE;
}

bool VulkanBackend::CreateLayouts() {
    VkDescriptorSetLayoutBinding b[vkshader::kBindings] = {};
    for (uint32_t i = 0; i < vkshader::kBindings; i++) {
        b[i].binding = i;
        b[i].descriptorCount = 1;
        b[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                              : i < vkshader::kBindingSamplers ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                                                               : VK_DESCRIPTOR_TYPE_SAMPLER;
        b[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = vkshader::kBindings;
    li.pBindings = b;
    VK_CHECK(vkCreateDescriptorSetLayout(dev_, &li, nullptr, &set_layout_));
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1;
    pi.pSetLayouts = &set_layout_;
    VK_CHECK(vkCreatePipelineLayout(dev_, &pi, nullptr, &layout_));
    // pipelines from earlier runs: building them is what stutters on a phone
    std::vector<char> blob;
    std::ifstream f(app::ExeDir() / "shadercache" / "pipelines.bin", std::ios::binary);
    if (f) blob.assign(std::istreambuf_iterator<char>(f), {});
    VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    ci.initialDataSize = blob.size();
    ci.pInitialData = blob.data();
    if (vkCreatePipelineCache(dev_, &ci, nullptr, &cache_) != VK_SUCCESS) {
        ci.initialDataSize = 0;
        VK_CHECK(vkCreatePipelineCache(dev_, &ci, nullptr, &cache_));
    }
    cache_saved_ = blob.size();
    return true;
}

bool VulkanBackend::CreateFrames() {
    VkSemaphoreTypeCreateInfoKHR ti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR};
    ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR;
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &ti};
    if (timeline_ok_) VK_CHECK(vkCreateSemaphore(dev_, &si, nullptr, &timeline_));
    for (Frame& f : frames_) {
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.queueFamilyIndex = qfam_;
        VK_CHECK(vkCreateCommandPool(dev_, &pi, nullptr, &f.pool));
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = f.pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(dev_, &ai, &f.cb));
        if (!CreateBuffer(f.upload, kUploadSize,
                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          true))
            return false;
        VkSemaphoreCreateInfo bi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(dev_, &bi, nullptr, &f.acquire));
    }
    return true;
}

bool VulkanBackend::CreateFixedImages() {
    constexpr VkImageUsageFlags kSampled = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (!CreateImage(lut_, kBgra, 256, 1, 1, 1, 1, kSampled, VK_IMAGE_VIEW_TYPE_2D) ||
        !CreateImage(dummy_[0], kBgra, 1, 1, 1, 1, 1, kSampled, VK_IMAGE_VIEW_TYPE_2D) ||
        !CreateImage(dummy_[1], kBgra, 1, 1, 1, 1, 1, kSampled, VK_IMAGE_VIEW_TYPE_3D) ||
        !CreateImage(dummy_[2], kBgra, 1, 1, 1, 6, 1, kSampled, VK_IMAGE_VIEW_TYPE_CUBE, true))
        return false;
    VkSamplerCreateInfo sc{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sc.magFilter = sc.minFilter = VK_FILTER_LINEAR;
    sc.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sc.addressModeU = sc.addressModeV = sc.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sc.maxLod = VK_LOD_CLAMP_NONE;
    linear_ = Sampler(sc);
    sc.magFilter = sc.minFilter = VK_FILTER_NEAREST;
    sc.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    point_ = Sampler(sc);
    // the first frame's commands: the fixed images cleared to black, the ramp the identity
    Begin();
    VkClearColorValue black{};
    for (Image* im : {&lut_, &dummy_[0], &dummy_[1], &dummy_[2]}) {
        Transition(*im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, im->layers};
        vkCmdClearColorImage(Cb(), im->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &r);
        Transition(*im, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    for (int i = 0; i < 768; i++) ramp_[i] = uint8_t(i & 255);
    UploadGammaRamp();
    Submit(true, false);
    // Submit(true) began the next recording; Init begins once more
    vkEndCommandBuffer(Cb());
    vkResetCommandPool(dev_, CurrentFrame().pool, 0);
    return true;
}

bool VulkanBackend::InitUi() {
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pi.maxSets = 16;
    pi.poolSizeCount = 1;
    pi.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(dev_, &pi, nullptr, &ui_pool_) != VK_SUCCESS) return false;
    std::lock_guard lock(app::UiLock());
    if (!ImGui::GetCurrentContext()) return false;
    ImGui_ImplVulkan_InitInfo ii{};
    ii.Instance = instance_;
    ii.PhysicalDevice = pd_;
    ii.Device = dev_;
    ii.QueueFamily = qfam_;
    ii.Queue = queue_;
    ii.DescriptorPool = ui_pool_;
    ii.MinImageCount = 2;
    ii.ImageCount = uint32_t(std::max<size_t>(swap_images_.size(), 2));
    ii.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ii.UseDynamicRendering = dynamic_;
    ii.PipelineRenderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    ii.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    ii.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swap_fmt_;
    if (!dynamic_) ii.RenderPass = Pass(swap_fmt_, VK_FORMAT_UNDEFINED);
    ImGui::GetIO().BackendRendererName = "vulkan";
    return ImGui_ImplVulkan_Init(&ii);
}

// ---- memory and objects

bool VulkanBackend::CreateImage(Image& im, VkFormat fmt, uint32_t w, uint32_t h, uint32_t d, uint32_t layers,
                                uint32_t levels, VkImageUsageFlags usage, VkImageViewType vt, bool cube) {
    im = {};
    im.fmt = fmt;
    im.w = w;
    im.h = h;
    im.d = d;
    im.layers = layers;
    im.levels = levels;
    im.depth = fmt == depth_fmt_;
    im.stencil = im.depth && HasStencil(fmt);
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    ci.imageType = d > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent = {w, h, d};
    ci.mipLevels = levels;
    ci.arrayLayers = layers;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(dev_, &ci, nullptr, &im.image) != VK_SUCCESS) return false;
    VkMemoryRequirements r;
    vkGetImageMemoryRequirements(dev_, im.image, &r);
    if (!alloc_.Get(r, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, false, im.mem) ||
        vkBindImageMemory(dev_, im.image, im.mem.mem, im.mem.off) != VK_SUCCESS) {
        Destroy(im);
        return false;
    }
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = im.image;
    vi.viewType = vt;
    vi.format = fmt;
    vi.subresourceRange = {VkImageAspectFlags(im.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0,
                           levels, 0, layers};
    if (usage & VK_IMAGE_USAGE_SAMPLED_BIT) vkCreateImageView(dev_, &vi, nullptr, &im.view);
    if (usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.subresourceRange = {VkImageAspectFlags(im.depth ? VK_IMAGE_ASPECT_DEPTH_BIT | (im.stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)
                                                           : VK_IMAGE_ASPECT_COLOR_BIT),
                               0, 1, 0, 1};
        vkCreateImageView(dev_, &vi, nullptr, &im.attach);
    }
    return true;
}

bool VulkanBackend::CreateBuffer(Buffer& b, VkDeviceSize size, VkBufferUsageFlags usage, bool host) {
    b = {};
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage;
    if (vkCreateBuffer(dev_, &ci, nullptr, &b.buf) != VK_SUCCESS) return false;
    VkMemoryRequirements r;
    vkGetBufferMemoryRequirements(dev_, b.buf, &r);
    VkMemoryPropertyFlags want = host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                      : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkMemoryPropertyFlags nice = host && (usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0;
    if (!alloc_.Get(r, want, nice, true, b.mem) || vkBindBufferMemory(dev_, b.buf, b.mem.mem, b.mem.off) != VK_SUCCESS) {
        Destroy(b);
        return false;
    }
    return true;
}

void VulkanBackend::Destroy(Image& im) {
    if (im.view) vkDestroyImageView(dev_, im.view, nullptr);
    DropFramebuffers(im.attach);
    if (im.attach) vkDestroyImageView(dev_, im.attach, nullptr);
    if (im.image) vkDestroyImage(dev_, im.image, nullptr);
    alloc_.Free(im.mem);
    im = {};
}

void VulkanBackend::Destroy(Buffer& b) {
    if (b.buf) vkDestroyBuffer(dev_, b.buf, nullptr);
    alloc_.Free(b.mem);
    b = {};
}

// bytes in this frame's ring; a big upload gets a buffer of its own, released with the frame. A full ring flushes:
// callers upload before they begin rendering and bind
VulkanBackend::Alloc VulkanBackend::Upload(VkDeviceSize size, VkDeviceSize align) {
    if (size > kUploadSize / 4) {
        Buffer b;
        if (!CreateBuffer(b, size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          true))
            return {VK_NULL_HANDLE, 0, nullptr};
        Alloc a{b.buf, 0, b.mem.map};
        CurrentFrame().buffers.push_back(b);
        return a;
    }
    perf2_bytes_ += size;
    Frame* f = &CurrentFrame();
    VkDeviceSize o = (f->used + align - 1) / align * align;
    if (o + size > kUploadSize) {
        Submit(true, false);
        f = &CurrentFrame();
        o = 0;
        }
    f->used = o + size;
    return {f->upload.buf, o, f->upload.mem.map + o};
}

VulkanBackend::Alloc VulkanBackend::Constants(const void* data, size_t bytes) {
    if (!data || !bytes) {
        Alloc a = Upload(kUboRange, kUboAlign);
        return a;
    }
    if (bytes <= sizeof(const_memo_[0].data)) {
        uint64_t h = Hash(data, bytes);
        size_t slot = (h ^ (h >> 16)) % kConstMemoEntries;
        ConstMemoEntry& e = const_memo_[slot];
        if (e.present == presents_ && e.bytes == bytes && e.hash == h && std::memcmp(e.data, data, bytes) == 0) {
            ++perf2_const_reuse_;
            return e.alloc;
        }
        Alloc a = Upload(kUboRange, kUboAlign);
        std::memcpy(a.cpu, data, std::min<size_t>(bytes, kUboRange));
        e.hash = h;
        e.present = presents_;
        e.bytes = uint32_t(bytes);
        e.alloc = a;
        std::memcpy(e.data, data, bytes);
        return a;
    }
    Alloc a = Upload(kUboRange, kUboAlign);
    std::memcpy(a.cpu, data, std::min<size_t>(bytes, kUboRange));
    return a;
}

void VulkanBackend::ResetPassState() {
    cur_pipe_ = VK_NULL_HANDLE;
    cur_set_ = VK_NULL_HANDLE;
    cur_vc_off_ = 0xFFFFFFFF;
    cur_dc_off_ = 0xFFFFFFFF;
    std::memset(&cur_vp_, 0, sizeof(cur_vp_));
    std::memset(&cur_sc_, 0, sizeof(cur_sc_));
    cur_blend_constants_[0] = -999.0f;
    cur_stencil_ref_ = 0xFFFFFFFF;
    cur_vb_count_ = 0;
    std::memset(cur_vbufs_, 0, sizeof(cur_vbufs_));
    std::memset(cur_voffs_, 0, sizeof(cur_voffs_));
    cur_ib_ = VK_NULL_HANDLE;
    cur_ib_off_ = 0xFFFFFFFF;
    cur_ib_type_ = VK_INDEX_TYPE_MAX_ENUM;
}

// ---- command recording

void VulkanBackend::Begin() {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(Cb(), &bi);
    rendering_ = false;
    rend_color_ = rend_depth_ = nullptr;
    ResetPassState();
}

uint64_t VulkanBackend::Completed() {
    if (!timeline_ok_) {
        std::lock_guard lock(fence_lock_);
        PollFences();
        return fenced_;
    }
    uint64_t v = 0;
    counter_value_(dev_, timeline_, &v);
    return v;
}

// the signalled fences, oldest first, back to the spares; called with fence_lock_ held
void VulkanBackend::PollFences() {
    size_t done = 0;
    while (done < pending_.size() && vkGetFenceStatus(dev_, pending_[done].fence) == VK_SUCCESS) {
        fenced_ = pending_[done].value;
        vkResetFences(dev_, 1, &pending_[done].fence);
        spare_fences_.push_back(pending_[done].fence);
        done++;
    }
    pending_.erase(pending_.begin(), pending_.begin() + done);
}

// the fence the submit of this value signals; called with fence_lock_ held
VkFence VulkanBackend::SubmitFence(uint64_t value) {
    VkFence f = VK_NULL_HANDLE;
    if (!spare_fences_.empty()) {
        f = spare_fences_.back();
        spare_fences_.pop_back();
    } else {
        VkFenceCreateInfo ci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(dev_, &ci, nullptr, &f) != VK_SUCCESS) return VK_NULL_HANDLE;
    }
    pending_.push_back({value, f});
    return f;
}

void VulkanBackend::WaitValue(uint64_t v) {
    if (!v || Completed() >= v) return;
    if (!timeline_ok_) {
        // the lock is held through the wait so no fence is reset under it; the GPU needs no thread to get there
        std::lock_guard lock(fence_lock_);
        for (const Pending& p : pending_)
            if (p.value >= v) {
                vkWaitForFences(dev_, 1, &p.fence, VK_TRUE, UINT64_MAX);
                break;
            }
        PollFences();
        return;
    }
    VkSemaphoreWaitInfoKHR wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO_KHR};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &timeline_;
    wi.pValues = &v;
    wait_semaphores_(dev_, &wi, UINT64_MAX);
}

// the frame's garbage freed and its pools reset, once the GPU is past it
void VulkanBackend::ResetFrame(Frame& f) {
    WaitValue(f.value);
    for (Image& im : f.images) Destroy(im);
    for (Buffer& b : f.buffers) Destroy(b);
    f.images.clear();
    f.buffers.clear();
    for (VkDescriptorPool p : f.pools) vkResetDescriptorPool(dev_, p, 0);
    f.pool_at = 0;
    // the ring starts over: an offset kept from the last frame would read other bytes now
    f.sets.clear();
    f.used = 0;
}

// present: the frame's last commands, waiting on the acquired image and signalling its present semaphore
void VulkanBackend::Submit(bool wait, bool present) {
    EndRender();
    Frame& f = CurrentFrame();
    const bool wait_acquire = f.acquire_pending;
    vkEndCommandBuffer(f.cb);
    uint64_t signal = ++value_;
    // without the timeline only the present semaphore is signalled, and a fence tells when the GPU is past
    VkSemaphore signals[2] = {timeline_, VK_NULL_HANDLE};
    uint64_t values[2] = {signal, 0};
    uint32_t first = timeline_ok_ ? 0 : 1;
    VkTimelineSemaphoreSubmitInfoKHR ti{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR};
    ti.signalSemaphoreValueCount = present ? 2 : 1;
    ti.pSignalSemaphoreValues = values;
    uint64_t wait_value = 0;
    ti.waitSemaphoreValueCount = wait_acquire ? 1 : 0;
    ti.pWaitSemaphoreValues = &wait_value;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, timeline_ok_ ? &ti : nullptr};
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    if (wait_acquire) {
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &f.acquire;
        si.pWaitDstStageMask = &stage;
    }
    if (present) {
        signals[1] = swap_done_[present_index_];
    }
    si.commandBufferCount = 1;
    si.pCommandBuffers = &f.cb;
    si.signalSemaphoreCount = (present ? 2 : 1) - first;
    si.pSignalSemaphores = signals + first;
    VkResult r;
    if (timeline_ok_) {
        r = vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
    } else {
        std::lock_guard lock(fence_lock_);
        r = vkQueueSubmit(queue_, 1, &si, SubmitFence(signal));
    }
    if (r != VK_SUCCESS)
        LOGE("vulkan: submit %llu%s failed (%d)", (unsigned long long)signal, present ? " (present)" : "", int(r));
    else if (wait_acquire)
        f.acquire_pending = false;
    f.value = signal;
    if (wait) {
        ResetFrame(f);
        vkResetCommandPool(dev_, f.pool, 0);
        Begin();
    }
}

void VulkanBackend::Transition(Image& im, VkImageLayout to) {
    if (im.layout == to) return;
    EndRender();
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    VkPipelineStageFlags src, dst;
    LayoutMasks(im.layout, src, b.srcAccessMask);
    LayoutMasks(to, dst, b.dstAccessMask);
    if (im.layout == VK_IMAGE_LAYOUT_UNDEFINED) src = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    b.oldLayout = im.layout;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im.image;
    VkImageAspectFlags aspect = im.depth ? VK_IMAGE_ASPECT_DEPTH_BIT | (im.stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)
                                         : VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange = {aspect, 0, im.levels, 0, im.layers};
    vkCmdPipelineBarrier(Cb(), src, dst, 0, 0, nullptr, 0, nullptr, 1, &b);
    im.layout = to;
}

// a rendering over the whole of the attachments, which keep what they hold; the one under way carries on when it
// has the same ones
void VulkanBackend::BeginRender(Image* color, Image* depth) {
    if (rendering_ && color == rend_color_ && depth == rend_depth_) return;
    EndRender();
    const uint32_t clear = clear_on_begin_;
    clear_on_begin_ = 0;
    if (color) Transition(*color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    if (depth) Transition(*depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    // the renderings before wrote the same attachments: their writes come first
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    VkPipelineStageFlags st = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                              VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    vkCmdPipelineBarrier(Cb(), st, st, 0, 1, &mb, 0, nullptr, 0, nullptr);
    rendering_ = true;
    rend_color_ = color;
    rend_depth_ = depth;
    ResetPassState();
    if (!dynamic_) {
        Image* any = color ? color : depth;
        VkFormat cf = color ? color->fmt : VK_FORMAT_UNDEFINED, df = depth ? depth->fmt : VK_FORMAT_UNDEFINED;
        VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        bi.renderPass = Pass(cf, df, clear);
        bi.framebuffer = Framebuffer(cf, df, color ? color->attach : VK_NULL_HANDLE, depth ? depth->attach : VK_NULL_HANDLE,
                                     any->w, any->h);
        bi.renderArea = {{0, 0}, {any->w, any->h}};
        // the clear values follow the attachment order the pass was built with: colour first
        VkClearValue cvs[2]{};
        if (color && (clear & 1)) cvs[bi.clearValueCount++].color = clear_color_;
        if (depth && (clear & 2)) cvs[bi.clearValueCount++].depthStencil = clear_ds_;
        bi.pClearValues = bi.clearValueCount ? cvs : nullptr;
        vkCmdBeginRenderPass(Cb(), &bi, VK_SUBPASS_CONTENTS_INLINE);
        return;
    }
    VkRenderingAttachmentInfoKHR ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR};
    VkRenderingAttachmentInfoKHR da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR};
    VkRenderingInfoKHR ri{VK_STRUCTURE_TYPE_RENDERING_INFO_KHR};
    Image* any = color ? color : depth;
    ri.renderArea = {{0, 0}, {any->w, any->h}};
    ri.layerCount = 1;
    if (color) {
        ca.imageView = color->attach;
        ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        ca.loadOp = (clear & 1) ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        ca.clearValue.color = clear_color_;
        ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &ca;
    }
    if (depth) {
        da.imageView = depth->attach;
        da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        da.loadOp = (clear & 2) ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        da.clearValue.depthStencil = clear_ds_;
        da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        ri.pDepthAttachment = &da;
        if (depth->stencil) ri.pStencilAttachment = &da;
    }
    begin_rendering_(Cb(), &ri);
}

void VulkanBackend::EndRender() {
    if (!rendering_) return;
    ++perf2_pass_breaks_;
    if (dynamic_)
        end_rendering_(Cb());
    else
        vkCmdEndRenderPass(Cb());
    rendering_ = false;
    rend_color_ = rend_depth_ = nullptr;
    ResetPassState();
}

// the layouts stay the attachment ones throughout, as with dynamic rendering: barriers outside do the transitions.
// pipelines only need a compatible pass, which load ops are not part of
VkRenderPass VulkanBackend::Pass(VkFormat color, VkFormat depth, uint32_t clear) {
    const uint64_t k[3] = {uint64_t(color), uint64_t(depth), uint64_t(clear)};
    uint64_t key = Hash(k, sizeof(k));
    if (auto it = passes_.find(key); it != passes_.end()) return it->second;
    VkAttachmentDescription at[2] = {};
    VkAttachmentReference cr{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference dr{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sp{};
    sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    uint32_t n = 0;
    if (color != VK_FORMAT_UNDEFINED) {
        at[n] = {0, color, VK_SAMPLE_COUNT_1_BIT, (clear & 1) ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
                 VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        cr.attachment = n++;
        sp.colorAttachmentCount = 1;
        sp.pColorAttachments = &cr;
    }
    if (depth != VK_FORMAT_UNDEFINED) {
        at[n] = {0, depth, VK_SAMPLE_COUNT_1_BIT,
                 (clear & 2) ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
                 (clear & 2) ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
                 VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        dr.attachment = n++;
        sp.pDepthStencilAttachment = &dr;
    }
    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = n;
    ci.pAttachments = at;
    ci.subpassCount = 1;
    ci.pSubpasses = &sp;
    VkRenderPass rp = VK_NULL_HANDLE;
    if (vkCreateRenderPass(dev_, &ci, nullptr, &rp) != VK_SUCCESS) LOGE("vulkan: render pass failed");
    passes_[key] = rp;
    return rp;
}

VkFramebuffer VulkanBackend::Framebuffer(VkFormat color_fmt, VkFormat depth_fmt, VkImageView color, VkImageView depth,
                                         uint32_t w, uint32_t h) {
    const uint64_t k[4] = {uint64_t(color), uint64_t(depth), w, h};
    uint64_t key = Hash(k, sizeof(k));
    if (auto it = framebuffers_.find(key); it != framebuffers_.end()) return it->second.fb;
    VkImageView views[2];
    uint32_t n = 0;
    if (color) views[n++] = color;
    if (depth) views[n++] = depth;
    VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    ci.renderPass = Pass(color_fmt, depth_fmt);
    ci.attachmentCount = n;
    ci.pAttachments = views;
    ci.width = w;
    ci.height = h;
    ci.layers = 1;
    VkFramebuffer fb = VK_NULL_HANDLE;
    if (vkCreateFramebuffer(dev_, &ci, nullptr, &fb) != VK_SUCCESS) LOGE("vulkan: framebuffer %ux%u failed", w, h);
    framebuffers_[key] = {color, depth, fb};
    return fb;
}

// a view's framebuffers go with it, once the GPU is past them
void VulkanBackend::DropFramebuffers(VkImageView v) {
    if (!v || framebuffers_.empty()) return;
    for (auto it = framebuffers_.begin(); it != framebuffers_.end();) {
        if (it->second.color == v || it->second.depth == v) {
            vkDestroyFramebuffer(dev_, it->second.fb, nullptr);
            it = framebuffers_.erase(it);
        } else {
            ++it;
        }
    }
}

// in the D3D convention, y down from the top: a negative height turns Vulkan's clip space the same way up
void VulkanBackend::SetViewport(float x, float y, float w, float h) {
    VkViewport vp{x, y + h, w, -h, 0, 1};
    if (std::memcmp(&vp, &cur_vp_, sizeof(vp)) == 0) return;
    vkCmdSetViewport(Cb(), 0, 1, &vp);
    cur_vp_ = vp;
}

void VulkanBackend::SetScissor(int32_t x0, int32_t y0, int32_t x1, int32_t y1) {
    VkRect2D r{{std::max(x0, 0), std::max(y0, 0)}, {uint32_t(std::max(x1 - std::max(x0, 0), 0)), uint32_t(std::max(y1 - std::max(y0, 0), 0))}};
    if (std::memcmp(&r, &cur_sc_, sizeof(r)) == 0) return;
    vkCmdSetScissor(Cb(), 0, 1, &r);
    cur_sc_ = r;
}

// the draw's descriptor set: the ring at b0 and b1 (offsets bound with it), the views and samplers; one per
// combination a frame
VkDescriptorSet VulkanBackend::Set(const VkImageView views[4], const VkSampler samplers[4]) {
    Frame& f = CurrentFrame();
    uint64_t key = Hash(views, sizeof(VkImageView) * 4, Hash(samplers, sizeof(VkSampler) * 4));
    if (auto it = f.sets.find(key); it != f.sets.end()) return it->second;
    VkDescriptorSet set = VK_NULL_HANDLE;
    for (;;) {
        if (f.pool_at == f.pools.size()) {
            VkDescriptorPoolSize sizes[3] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, kSetsPerPool * 2},
                                             {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kSetsPerPool * 4},
                                             {VK_DESCRIPTOR_TYPE_SAMPLER, kSetsPerPool * 4}};
            VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pi.maxSets = kSetsPerPool;
            pi.poolSizeCount = 3;
            pi.pPoolSizes = sizes;
            VkDescriptorPool p = VK_NULL_HANDLE;
            if (vkCreateDescriptorPool(dev_, &pi, nullptr, &p) != VK_SUCCESS) return VK_NULL_HANDLE;
            f.pools.push_back(p);
        }
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = f.pools[f.pool_at];
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &set_layout_;
        if (vkAllocateDescriptorSets(dev_, &ai, &set) == VK_SUCCESS) break;
        f.pool_at++;
    }
    VkDescriptorBufferInfo bi{f.upload.buf, 0, kUboRange};
    VkDescriptorImageInfo ii[4], si[4];
    VkWriteDescriptorSet w[vkshader::kBindings];
    for (uint32_t i = 0; i < vkshader::kBindings; i++) {
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = set;
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
        if (i < 2) {
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            w[i].pBufferInfo = &bi;
        } else if (i < vkshader::kBindingSamplers) {
            uint32_t s = i - vkshader::kBindingTextures;
            ii[s] = {VK_NULL_HANDLE, views[s] ? views[s] : dummy_[0].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            w[i].pImageInfo = &ii[s];
        } else {
            uint32_t s = i - vkshader::kBindingSamplers;
            si[s] = {samplers[s] ? samplers[s] : point_, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            w[i].pImageInfo = &si[s];
        }
    }
    vkUpdateDescriptorSets(dev_, vkshader::kBindings, w, 0, nullptr);
    f.sets[key] = set;
    return set;
}

void VulkanBackend::BindSet(VkDescriptorSet set, VkDeviceSize vc, VkDeviceSize dc) {
    uint32_t offs[2] = {uint32_t(vc), uint32_t(dc)};
    vkCmdBindDescriptorSets(Cb(), VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &set, 2, offs);
}

// ---- shaders and pipelines

VkShaderModule VulkanBackend::Module(const std::string& src, const char* entry, vkshader::Stage stage, uint64_t& hash) {
    uint64_t seed = Hash(entry, std::strlen(entry), stage == vkshader::Stage::Vertex ? 1 : 2);
    hash = Hash(src.data(), src.size(), seed);
    if (auto it = modules_.find(hash); it != modules_.end()) return it->second;
    std::vector<uint32_t> spirv;
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkshader::Compile(src, entry, stage, hash, spirv)) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = spirv.size() * 4;
        ci.pCode = spirv.data();
        if (vkCreateShaderModule(dev_, &ci, nullptr, &m) != VK_SUCCESS) m = VK_NULL_HANDLE;
    }
    modules_[hash] = m;
    return m;
}

// a full screen pass: the triangle from SV_VertexID, solid, no culling; depth set: writes depth only
VkPipeline VulkanBackend::FullScreenPipeline(const char* hlsl, const char* ps_entry, VkFormat color, VkFormat depth,
                                             bool over) {
    uint64_t key = Hash(hlsl, std::strlen(hlsl), uint64_t(color) << 32 | uint64_t(depth) << 1 | (over ? 1 : 0));
    key = Hash(ps_entry, std::strlen(ps_entry), key);
    if (auto it = pipelines_.find(key); it != pipelines_.end()) return it->second;
    uint64_t hv, hp;
    VkShaderModule vs = Module(hlsl, "vs", vkshader::Stage::Vertex, hv);
    VkShaderModule ps = Module(hlsl, ps_entry, vkshader::Stage::Pixel, hp);
    VkPipeline pipe = VK_NULL_HANDLE;
    if (vs && ps) {
        VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                                 {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
        st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        st[0].module = vs;
        st[0].pName = "main";
        st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        st[1].module = ps;
        st[1].pName = "main";
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.lineWidth = 1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = ds.depthWriteEnable = depth != VK_FORMAT_UNDEFINED;
        ds.depthCompareOp = VK_COMPARE_OP_ALWAYS;
        VkPipelineColorBlendAttachmentState ba{};
        ba.colorWriteMask = 0xF;
        if (over) {
            ba.blendEnable = VK_TRUE;
            ba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        }
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = color != VK_FORMAT_UNDEFINED ? 1 : 0;
        cb.pAttachments = &ba;
        VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = 2;
        dy.pDynamicStates = dyn;
        VkPipelineRenderingCreateInfoKHR rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
        rinfo.colorAttachmentCount = cb.attachmentCount;
        rinfo.pColorAttachmentFormats = &color;
        rinfo.depthAttachmentFormat = depth;
        rinfo.stencilAttachmentFormat = HasStencil(depth) ? depth : VK_FORMAT_UNDEFINED;
        VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, dynamic_ ? &rinfo : nullptr};
        if (!dynamic_) pi.renderPass = Pass(color, depth);
        pi.stageCount = 2;
        pi.pStages = st;
        pi.pVertexInputState = &vi;
        pi.pInputAssemblyState = &ia;
        pi.pViewportState = &vp;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;
        pi.pDepthStencilState = &ds;
        pi.pColorBlendState = &cb;
        pi.pDynamicState = &dy;
        pi.layout = layout_;
        VkResult result = vkCreateGraphicsPipelines(dev_, cache_, 1, &pi, nullptr, &pipe);
        if (result != VK_SUCCESS) {
            LOGE("vulkan: fullscreen-pipeline failed entry=%s color=%u depth=%u dynamic=%u result=%d vs=%p ps=%p",
                 ps_entry, uint32_t(color), uint32_t(depth), dynamic_, int(result), (void*)vs, (void*)ps);
            pipe = VK_NULL_HANDLE;
        }
    }
    if (!pipe && (!vs || !ps))
        LOGE("vulkan: fullscreen-shader missing entry=%s color=%u depth=%u vs=%p ps=%p",
             ps_entry, uint32_t(color), uint32_t(depth), (void*)vs, (void*)ps);
    pipelines_[key] = pipe;
    return pipe;
}

VkPipeline VulkanBackend::EnsurePipeline(const PsoKey& k, VkShaderModule vs, VkShaderModule ps) {
    uint64_t h = Hash(&k, sizeof(k));
    if (auto it = pipelines_.find(h); it != pipelines_.end()) return it->second;
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    st[0].module = vs;
    st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    st[1].module = ps;
    st[1].pName = "main";
    // the attributes at their register's location, as vk_shader.cpp numbers the inputs
    VkVertexInputBindingDescription vb[16];
    VkVertexInputAttributeDescription va[16];
    uint32_t na = 0;
    for (uint32_t i = 0; i < k.buffers; i++) vb[i] = {i, k.strides[i], VK_VERTEX_INPUT_RATE_VERTEX};
    for (uint32_t r = 0; r < 16; r++) {
        if (!((k.regs >> r) & 1)) continue;
        uint32_t e = k.elements[r];
        va[na++] = {r, (e >> 8) & 0xFF, VertexFormat(e & 0xFF), e >> 16};
    }
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = k.buffers;
    vi.pVertexBindingDescriptions = vb;
    vi.vertexAttributeDescriptionCount = na;
    vi.pVertexAttributeDescriptions = va;
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    Topology topo = Topology(k.topo & 0xF);
    bool strip = k.topo >> 4;
    ia.topology = topo == Topology::Points ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST
                  : topo == Topology::Lines ? (strip ? VK_PRIMITIVE_TOPOLOGY_LINE_STRIP : VK_PRIMITIVE_TOPOLOGY_LINE_LIST)
                                            : (strip ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    constexpr uint32_t kFillWireframe = 0x1B01;
    rs.polygonMode = k.fill == kFillWireframe && feats_.fillModeNonSolid ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    rs.cullMode = k.cull ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    rs.frontFace = k.ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
    rs.depthClampEnable = feats_.depthClamp;
    rs.depthBiasEnable = k.zbias != 0 || k.zslope != 0;
    rs.depthBiasConstantFactor = float(k.zbias);
    rs.depthBiasSlopeFactor = k.zslope;
    rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = k.zen;
    ds.depthWriteEnable = k.zwrite;
    ds.depthCompareOp = Cmp(k.zfunc);
    ds.stencilTestEnable = k.sten;
    VkStencilOpState so{StencilOp(k.sfail), StencilOp(k.spass), StencilOp(k.szfail), Cmp(k.sfunc), k.smask, k.swmask, 0};
    ds.front = ds.back = so;
    VkPipelineColorBlendAttachmentState ba{};
    ba.blendEnable = k.blend;
    ba.srcColorBlendFactor = Blend(k.src, false);
    ba.dstColorBlendFactor = Blend(k.dst, false);
    ba.colorBlendOp = BlendOp(k.op);
    ba.srcAlphaBlendFactor = Blend(k.src, true);
    ba.dstAlphaBlendFactor = Blend(k.dst, true);
    ba.alphaBlendOp = BlendOp(k.op);
    ba.colorWriteMask = k.wmask;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = k.has_rt ? 1 : 0;
    cb.pAttachments = &ba;
    VkDynamicState dyn[4] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
                             VK_DYNAMIC_STATE_STENCIL_REFERENCE};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = 4;
    dy.pDynamicStates = dyn;
    VkFormat color = VkFormat(k.rtfmt);
    VkPipelineRenderingCreateInfoKHR rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    rinfo.colorAttachmentCount = cb.attachmentCount;
    rinfo.pColorAttachmentFormats = &color;
    rinfo.depthAttachmentFormat = k.has_ds ? depth_fmt_ : VK_FORMAT_UNDEFINED;
    rinfo.stencilAttachmentFormat = k.has_ds && HasStencil(depth_fmt_) ? depth_fmt_ : VK_FORMAT_UNDEFINED;
    VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, dynamic_ ? &rinfo : nullptr};
    if (!dynamic_) pi.renderPass = Pass(k.has_rt ? color : VK_FORMAT_UNDEFINED, k.has_ds ? depth_fmt_ : VK_FORMAT_UNDEFINED);
    pi.stageCount = 2;
    pi.pStages = st;
    pi.pVertexInputState = &vi;
    pi.pInputAssemblyState = &ia;
    pi.pViewportState = &vp;
    pi.pRasterizationState = &rs;
    pi.pMultisampleState = &ms;
    pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb;
    pi.pDynamicState = &dy;
    pi.layout = layout_;
    VkPipeline pipe = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(dev_, cache_, 1, &pi, nullptr, &pipe) != VK_SUCCESS) {
        LOGE("vulkan: pipeline creation failed");
        pipe = VK_NULL_HANDLE;
    }
    pipelines_[h] = pipe;
    return pipe;
}

VkSampler VulkanBackend::Sampler(const VkSamplerCreateInfo& ci) {
    uint64_t key = Hash(&ci.magFilter, sizeof(VkSamplerCreateInfo) - offsetof(VkSamplerCreateInfo, magFilter));
    if (auto it = samplers_.find(key); it != samplers_.end()) return it->second;
    VkSampler s = VK_NULL_HANDLE;
    vkCreateSampler(dev_, &ci, nullptr, &s);
    samplers_[key] = s;
    return s;
}

// fixed border colours only: the console's nearest
VkBorderColor Border(uint32_t argb) {
    if ((argb >> 24) < 128) return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    uint32_t sum = ((argb >> 16) & 255) + ((argb >> 8) & 255) + (argb & 255);
    return sum >= 384 ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
}

VkSampler VulkanBackend::StageSampler(const State& s, int i, const DrawTextures& stages, bool clamp) {
    VkSamplerCreateInfo sd{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    const auto& t = s.tss[size_t(i)];
    uint32_t mag = t[kTssMagFilter], min = t[kTssMinFilter], mip = t[kTssMipFilter];
    auto lin = [](uint32_t f) { return f >= kTexfLinear; };
    bool filterable = stages.fmt[i] != VK_FORMAT_R32_SFLOAT || r32f_linear_;
    sd.minFilter = lin(min) && filterable ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    sd.magFilter = lin(mag) && filterable ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    sd.mipmapMode = mip == kTexfLinear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    bool anisotropic = min == kTexfAnisotropic || mag == kTexfAnisotropic;
    uint32_t an = anisotropic ? std::max<uint32_t>(t[kTssMaxAnisotropy], 1) : 1;
    uint32_t aniso = app::Get().aniso;
    bool mipped = mip && stages.levels[i] > 1;
    if (aniso && lin(min) && mipped && !stages.info[i].shadow) an = aniso;
    if (an > 1 && feats_.samplerAnisotropy) {
        sd.anisotropyEnable = VK_TRUE;
        sd.maxAnisotropy = std::min(float(an), props_.limits.maxSamplerAnisotropy);
    }
    uint32_t u = t[kTssAddressU], v = t[kTssAddressV];
    sd.addressModeU = Address(clamp && u == kTaddressWrap ? kTaddressClamp : u);
    sd.addressModeV = Address(clamp && v == kTaddressWrap ? kTaddressClamp : v);
    sd.addressModeW = Address(t[kTssAddressW]);
    float limit = props_.limits.maxSamplerLodBias;
    sd.mipLodBias = std::clamp(Bits(t[kTssMipLodBias]), -limit, limit);
    sd.minLod = std::min(float(t[kTssMaxMipLevel]), float(stages.levels[i] - 1));
    sd.maxLod = mip ? VK_LOD_CLAMP_NONE : sd.minLod;
    sd.borderColor = Border(t[kTssBorderColor]);
    if (stages.info[i].shadow) {
        // the console tests texel OP r, Vulkan r OP texel
        static const uint8_t swap[8] = {0, 4, 2, 6, 1, 5, 3, 7};
        bool linear = (lin(min) || lin(mag)) && depth_linear_;
        sd.minFilter = sd.magFilter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        sd.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sd.compareEnable = VK_TRUE;
        sd.compareOp = VkCompareOp(swap[CmpIndex(s.rs[kRsShadowFunc])]);
        sd.anisotropyEnable = VK_FALSE;
        sd.maxAnisotropy = 1;
    }
    return Sampler(sd);
}

// ---- targets

Target* VulkanBackend::FindTarget(const Surface& s, bool depth) {
    if (!s.header) return nullptr;
    uint32_t data = mem::VirtualFromPhysical(s.data), gw, gh;
    SurfaceSize(s, gw, gh);
    auto it = targets_.find(TargetMapKey(data, depth, gw, gh));
    if (it == targets_.end()) return nullptr;
    it->second.use_serial = ++target_use_serial_;
    return &it->second;
}

Target* VulkanBackend::FindTargetByData(uint32_t data, bool depth) {
    Target* best = nullptr;
    for (auto& [key, target] : targets_) {
        if (target.data == data && target.depth == depth && (!best || target.use_serial > best->use_serial))
            best = &target;
    }
    return best;
}

void VulkanBackend::ReleaseTarget(std::unordered_map<uint64_t, Target>::iterator it) {
    const uint64_t key = TargetMapKey(it->second.data, it->second.depth, it->second.gw, it->second.gh);
    if (!it->second.depth) {
        // A recreated colour target starts with content_serial=0. Do not compare that serial with
        // the old allocation's value or keep presenting a different target after a resize.
        presented_serial_.erase(key);
        if (last_present_key_ == key) last_present_key_ = 0;
    }
    if (draw_rt_ == &it->second) draw_rt_ = nullptr;
    if (draw_ds_ == &it->second) draw_ds_ = nullptr;
    Retire(it->second.img);
    Retire(it->second.fb);
    targets_.erase(it);
}

Target* VulkanBackend::EnsureTarget(const Surface& s, bool depth) {
    if (!s.header) return nullptr;
    uint32_t data = mem::VirtualFromPhysical(s.data), gw, gh;
    SurfaceSize(s, gw, gh);
    uint64_t key = TargetMapKey(data, depth, gw, gh);
    auto it = targets_.find(key);
    if (it != targets_.end()) {
        it->second.use_serial = ++target_use_serial_;
        return &it->second;
    }
    if (gw <= 1 && gh <= 1) {
        for (auto& [other_key, other] : targets_) {
            if (other.data == data && other.depth == depth && other.gw > 1 && other.gh > 1) {
                other.use_serial = ++target_use_serial_;
                return &other;
            }
        }
    }
    static std::atomic<uint32_t> diag_alias{0};
    for (const auto& [other_key, other] : targets_) {
        if (other.data == data && other.depth == depth) {
            uint32_t n = diag_alias.fetch_add(1, std::memory_order_relaxed);
            if (n < 32)
                LOGR("vulkan: target-alias[%u] data=%08X depth=%u header=%08X fmt=%08X size=%08X old=%ux%u serial=%llu new=%ux%u",
                     n, data, depth, s.header, s.format, s.size, other.gw, other.gh,
                     static_cast<unsigned long long>(other.content_serial), gw, gh);
            break;
        }
    }
    Target t;
    t.depth = depth;
    t.gw = gw;
    t.gh = gh;
    t.w = std::max(1u, uint32_t(std::lround(gw * scale_)));
    t.h = std::max(1u, uint32_t(std::lround(gh * scale_)));
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                              (depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    if (!CreateImage(t.img, depth ? depth_fmt_ : kBgra, t.w, t.h, 1, 1, 1, usage, VK_IMAGE_VIEW_TYPE_2D)) {
        LOGE("vulkan: no %ux%u target", t.w, t.h);
        return nullptr;
    }
    // a new target starts cleared
    Transition(t.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    if (depth) {
        VkClearDepthStencilValue one{1, 0};
        VkImageSubresourceRange r{VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT | (t.img.stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)), 0, 1, 0, 1};
        vkCmdClearDepthStencilImage(Cb(), t.img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &one, 1, &r);
    } else {
        VkClearColorValue black{};
        VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(Cb(), t.img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &r);
    }
    t.data = data;
    t.bytes = depth ? gw * gh * 4 : GuestPitch(gw) * gh;
    t.epoch = DirtyEpoch(data);
    t.sig = MemSig(data, t.bytes);
    t.use_serial = ++target_use_serial_;
    Target& ref = targets_[key];
    ref = std::move(t);
    return &ref;
}

// BGRA pixels into a new w x h image, left readable
bool VulkanBackend::PixelImage(Image& im, const std::vector<uint8_t>& px, uint32_t pitch, uint32_t w, uint32_t h) {
    if (!CreateImage(im, kBgra, w, h, 1, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                     VK_IMAGE_VIEW_TYPE_2D))
        return false;
    Alloc a = Upload(VkDeviceSize(w) * h * 4, 16);
    if (!a.cpu) return false;
    for (uint32_t y = 0; y < h; y++) std::memcpy(a.cpu + size_t(y) * w * 4, &px[size_t(y) * pitch], w * 4);
    Transition(im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy c{a.off, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {w, h, 1}};
    vkCmdCopyBufferToImage(Cb(), a.buf, im.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    Transition(im, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return true;
}

// CPU pixels of a colour surface into its target, scaled; after a Sync only the changed ones, over the picture
void VulkanBackend::UploadTarget(Target& t, const Surface& s) {
    TextureDesc d;
    if (!DescribeSurface(s, d) && !Describe(s.header, d)) return;
    UploadTarget(t, d);
}

void VulkanBackend::UploadTarget(Target& t, const TextureDesc& d) {
    if (Format(d.format).depth) return;
    if (!trace_dir_.empty())
        LOGR("vulkan: frame-trace upload draw=%u target=%08X/%ux%u source=%08X/%ux%u pitch=%u format=%u serial=%llu",
             trace_draw_, t.data, t.gw, t.gh, d.data, d.width, d.height, d.pitch, d.format,
             static_cast<unsigned long long>(t.content_serial));
    std::vector<uint8_t> px;
    uint32_t pitch;
    bool over = !t.drawn && d.width == t.gw && d.height == t.gh && t.read.size() == size_t(t.gw) * t.gh * 4;
    if (over) {
        pitch = t.gw * 4;
        px = LatchCpuChanges(t);
    } else {
        DecodeLevel(d, 0, 0, static_cast<const uint8_t*>(kernel::HostPtr(d.data)), nullptr, px, pitch);
    }
    Image tex;
    if (!PixelImage(tex, px, pitch, d.width, d.height)) return;
    float uv[4] = {0, 0, 1, 1};
    Blit(t.img, {{0, 0}, {t.w, t.h}}, tex, uv, {.alpha = true, .over = over});
    Retire(tex);
    ++t.content_serial;
    t.epoch = DirtyEpoch(t.data);
    t.sig = MemSig(t.data, t.bytes);
}

// src over the rect of dst through the blit shader: its constants first, then the rendering
void VulkanBackend::Blit(Image& dst, const VkRect2D& to, Image& src, const float uvrect[4], const BlitOptions& o) {
    float c[8] = {uvrect[0], uvrect[1], uvrect[2], uvrect[3], o.gamma ? 1.0f : 0.0f, o.alpha ? 1.0f : 0.0f,
                  float(o.taps), o.mul};
    Alloc cb = Constants(c, sizeof(c));
    VkPipeline pipe = FullScreenPipeline(kBlitHlsl, "psmain", dst.fmt, VK_FORMAT_UNDEFINED, o.over);
    if (!pipe) return;
    Transition(src, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    Transition(lut_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    const VkImageView views[4] = {src.view, lut_.view, VK_NULL_HANDLE, VK_NULL_HANDLE};
    const VkSampler samplers[4] = {linear_, point_, point_, point_};
    VkDescriptorSet set = Set(views, samplers);
    BeginRender(&dst, nullptr);
    vkCmdBindPipeline(Cb(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    BindSet(set, cb.off, cb.off);
    SetViewport(float(to.offset.x), float(to.offset.y), float(to.extent.width), float(to.extent.height));
    SetScissor(0, 0, int32_t(dst.w), int32_t(dst.h));
    vkCmdDraw(Cb(), 3, 1, 0, 0);
}

// a copy of the target to read while drawing to it
Image* VulkanBackend::Feedback(Target& t) {
    ++perf2_feedback_;
    if (!t.fb && !CreateImage(t.fb, t.img.fmt, t.w, t.h, 1, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                              VK_IMAGE_VIEW_TYPE_2D))
        return nullptr;
    Transition(t.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    Transition(t.fb, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageAspectFlags aspect = t.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageCopy c{{aspect, 0, 0, 1}, {0, 0, 0}, {aspect, 0, 0, 1}, {0, 0, 0}, {t.w, t.h, 1}};
    vkCmdCopyImage(Cb(), t.img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.fb.image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    Transition(t.fb, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return &t.fb;
}

// a depth buffer drawn as colour into another depth buffer's memory: copied as depth
bool VulkanBackend::DepthCopy(const State& s) {
    TextureDesc d;
    if (!s.rt.header || !s.textures[0] || !Describe(s.textures[0], d) || Format(d.format).depth || d.cube ||
        d.depth != 1)
        return false;
    uint32_t rt_w, rt_h;
    SurfaceSize(s.rt, rt_w, rt_h);
    auto di = targets_.find(TargetMapKey(mem::VirtualFromPhysical(s.rt.data), true, rt_w, rt_h));
    auto si = targets_.find(TargetMapKey(d.data, true, d.width, d.height));
    if (di == targets_.end() || si == targets_.end() || di == si) return false;
    Target& dst = di->second;
    Target& src = si->second;
    Alloc cb = Constants(nullptr, 0);
    VkPipeline pipe = FullScreenPipeline(kDepthHlsl, "psmain", VK_FORMAT_UNDEFINED, depth_fmt_, false);
    if (!pipe) return false;
    Transition(src.img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    const VkImageView views[4] = {src.img.view, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    const VkSampler samplers[4] = {linear_, point_, point_, point_};
    VkDescriptorSet set = Set(views, samplers);
    BeginRender(nullptr, &dst.img);
    vkCmdBindPipeline(Cb(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    BindSet(set, cb.off, cb.off);
    SetViewport(0, 0, float(dst.w), float(dst.h));
    SetScissor(0, 0, int32_t(dst.w), int32_t(dst.h));
    vkCmdDraw(Cb(), 3, 1, 0, 0);
    dst.drawn = true;
    ++dst.content_serial;
    return true;
}

// src filtered into a new w x h BGRA image, left as a copy source
bool VulkanBackend::Shrink(Image& src, uint32_t w, uint32_t h, Image& down, uint32_t taps) {
    if (!down || down.w != w || down.h != h) {
        Retire(down);
        if (!CreateImage(down, kBgra, w, h, 1, 1, 1,
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_VIEW_TYPE_2D))
            return false;
    }
    float uv[4] = {0, 0, 1, 1};
    Blit(down, {{0, 0}, {w, h}}, src, uv, {.alpha = true, .taps = taps});
    Transition(down, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    return true;
}

// the target as a guest sized BGRA copy source: itself, or shrunk into down
Image* VulkanBackend::CopySource(Target& t, Image& down) {
    if (t.depth) return nullptr;
    if (t.w != t.gw || t.h != t.gh) {
        uint32_t taps = uint32_t(std::ceil(float(t.w) / t.gw - 1e-3f));
        return Shrink(t.img, t.gw, t.gh, down, std::max(taps, 1u)) ? &down : nullptr;
    }
    Transition(t.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    return &t.img;
}

// a copy source to the CPU, rows packed; waits for the GPU
bool VulkanBackend::ReadBack(Image& im, uint32_t w, uint32_t h, std::vector<uint8_t>& out) {
    Buffer rb;
    if (!CreateBuffer(rb, VkDeviceSize(w) * h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true)) return false;
    VkBufferImageCopy c{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {w, h, 1}};
    vkCmdCopyImageToBuffer(Cb(), im.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buf, 1, &c);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(Cb(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    Submit(true, false);
    out.assign(rb.mem.map, rb.mem.map + size_t(w) * h * 4);
    Destroy(rb);
    return true;
}

// ---- draws

void VulkanBackend::EnsureVertexShader(const State& s, const Assembled& a, DrawShaders& sh) {
    if (auto it = shader_src_.find(sh.vkey); it != shader_src_.end()) {
        sh.vh = it->second;
        sh.vs = modules_[sh.vh];
        return;
    }
    std::string src = VertexHlsl(s, a, sh);
    static std::atomic<bool> frontend_shader_logged{false};
    if (IsForzaFrontendShader(s) && !frontend_shader_logged.exchange(true, std::memory_order_relaxed)) {
        LOGR("vulkan: frontend-vs-source regs=%04X decode=%016llX vertices=%u buffers=%u program=%zu",
             a.regs, static_cast<unsigned long long>(a.decode), a.vertex_count, a.buffer_count,
             s.vs ? s.vs->program.size() / 4 : 0);
        size_t start = 0;
        while (start < src.size()) {
            size_t end = src.find('\n', start);
            if (end == std::string::npos) end = src.size();
            LOGR("vulkan: frontend-vs-hlsl %.*s", int(end - start), src.data() + start);
            start = end + 1;
        }
    }
    const char* entry = "main";
    if (a.topology == Topology::Points) {
        std::string points = PointsHlsl(src);
        if (!points.empty()) {
            src = std::move(points);
            entry = "main_points";
        }
    }
    sh.vs = Module(src, entry, vkshader::Stage::Vertex, sh.vh);
    shader_src_[sh.vkey] = sh.vh;
}

void VulkanBackend::EnsurePixelShader(const PixelKey& pk, DrawShaders& sh) {
    if (auto it = shader_src_.find(sh.pkey ^ kPixelKeySalt); it != shader_src_.end()) {
        sh.ph = it->second;
        sh.ps = modules_[sh.ph];
        return;
    }
    sh.ps = Module(PixelHlsl(pk), "main", vkshader::Stage::Pixel, sh.ph);
    shader_src_[sh.pkey ^ kPixelKeySalt] = sh.ph;
}

constexpr float kUpscaledMin = 1.01f;

VkImageView VulkanBackend::BindTargetTexture(const State& s, int stage, const TextureDesc& d, Target& t, bool box,
                                             DrawTextures& stages) {
    TexInfo& info = stages.info[stage];
    if (&t == draw_ds_) return VK_NULL_HANDLE;
    bool feedback = &t == draw_rt_;
    stages.rt[stage] = true;
    stages.gw[stage] = t.depth ? 0 : t.gw;
    if (!t.depth && LatchCpuWrite(t)) UploadTarget(t, d);
    info.kind = kTex2D;
    Target* rt = draw_rt_;
    float k = float(t.w) / (t.gw * stages.k[stage]);
    bool blur = rt && t.gw < guest_w_ && rt->gw < guest_w_;
    if (!t.depth && box && k > kUpscaledMin && rt && rt != &t && (blur || t.gw > rt->gw || t.gh > rt->gh)) {
        stages.wide[stage] = k;
        stages.blur[stage] = blur;
        info.flags |= kTexWide;
    }
    stages.fmt[stage] = t.img.fmt;
    if (feedback) {
        Image* fb = Feedback(t);
        return fb ? fb->view : VK_NULL_HANDLE;
    }
    // Capture the first rendered 512x512 depth target sampled by the main
    // scene. The Civic's dark panels use this target for the shadow compare.
    if (!trace_dir_.empty() && draw_rt_ && !draw_rt_->depth && draw_rt_->gw == 640 && draw_rt_->gh == 480 &&
        t.depth && t.drawn && t.gw == 512 && t.gh == 512 && trace_depth_captured_.insert(t.data).second) {
        char name[64];
        std::snprintf(name, sizeof(name), "sampled-depth-%08X", t.data);
        TraceDepth(t, name);
    }
    // Capture the first completed, sizeable render targets sampled by the main
    // Forza scene. This distinguishes a missing reflection/light pass from a
    // bad UV or combiner result without relying on guest allocation addresses.
    if (!trace_dir_.empty() && draw_rt_ && !draw_rt_->depth && draw_rt_->gw == 640 && draw_rt_->gh == 480 &&
        !t.depth && t.data != draw_rt_->data && t.gw >= 128 && t.gh >= 128) {
        static std::set<uint32_t> captured_targets;
        if (captured_targets.size() < 8 && captured_targets.insert(t.data).second) {
            char name[64];
            std::snprintf(name, sizeof(name), "sampled-target-%08X", t.data);
            TraceImage(t.img, name);
        }
    }
    if (t.depth) info.shadow = ShadowDepthKind(d.format);
    Transition(t.img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return t.img.view;
}

bool VulkanBackend::CreateTexture(Tex& tx, const TextureDesc& d, const FormatInfo& fi, uint64_t key) {
    Retire(tx.img);
    tx = Tex{};
    tx.d = d;
    bool bc = fi.compressed;
    tx.decode = bc && !bc_;
    switch (fi.host) {
        case Host::BC1: tx.fmt = VK_FORMAT_BC1_RGBA_UNORM_BLOCK; break;
        case Host::BC2: tx.fmt = VK_FORMAT_BC2_UNORM_BLOCK; break;
        case Host::BC3: tx.fmt = VK_FORMAT_BC3_UNORM_BLOCK; break;
        case Host::R32F: tx.fmt = VK_FORMAT_R32_SFLOAT; break;
        default: tx.fmt = kBgra; break;
    }
    if (tx.decode) tx.fmt = kBgra;
    uint32_t w = d.width, h = d.height;
    tx.levels = d.levels;
    if (bc && ((w & 3) || (h & 3))) {
        w = (w + 3) & ~3u;
        h = (h + 3) & ~3u;
        tx.levels = 1;
    }
    VkImageViewType vt = d.cube ? VK_IMAGE_VIEW_TYPE_CUBE : d.depth > 1 ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
    if (!CreateImage(tx.img, tx.fmt, w, h, d.cube ? 1 : d.depth, d.cube ? 6 : 1, tx.levels,
                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, vt, d.cube)) {
        textures_.erase(key);
        return false;
    }
    return true;
}

void VulkanBackend::UploadTexture(Tex& tx, const TextureDesc& d, const FormatInfo& fi, const uint8_t* p,
                                  const uint32_t* colours) {
    ++perf2_tex_;
    if (d.format == 0x12 && d.width == 720 && d.height == 480) {
        static std::set<uint32_t> dumped_screen_textures;
        if (dumped_screen_textures.insert(d.data).second) {
            const char* root = "/storage/emulated/0/Android/data/com.racas.forzamotorsport/files/forza-debug";
            std::error_code ec;
            std::filesystem::create_directories(root, ec);
            char name[96];
            std::snprintf(name, sizeof(name), "screen-%08X.raw", d.data);
            std::ofstream raw(std::filesystem::path(root) / name, std::ios::binary);
            size_t bytes = TotalSize(d);
            if (raw && p) raw.write(reinterpret_cast<const char*>(p), bytes);
            std::vector<uint8_t> decoded;
            uint32_t pitch = 0;
            DecodeLevel(d, 0, 0, p, colours, decoded, pitch);
            std::snprintf(name, sizeof(name), "screen-%08X.rgba", d.data);
            std::ofstream rgba(std::filesystem::path(root) / name, std::ios::binary);
            if (rgba && !decoded.empty()) rgba.write(reinterpret_cast<const char*>(decoded.data()), decoded.size());
            LOGR("vulkan: screen-texture-dump data=%08X raw=%d decoded=%d bytes=%zu pitch=%u error=%d",
                 d.data, raw.good(), rgba.good(), bytes, pitch, int(ec.value()));
        }
    }
    Transition(tx.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    uint32_t faces = d.cube ? 6 : 1;
    std::vector<uint8_t> px, decoded;
    bool bc = fi.compressed && !tx.decode;
    uint32_t block = fi.host == Host::BC1 ? 8 : 16;
    for (uint32_t f = 0; f < faces; f++)
        for (uint32_t l = 0; l < tx.levels; l++) {
            uint32_t pitch;
            DecodeLevel(d, l, f, p, colours, px, pitch);
            uint32_t mw = std::max(tx.img.w >> l, 1u), mh = std::max(tx.img.h >> l, 1u), md = std::max(tx.img.d >> l, 1u);
            const std::vector<uint8_t>* src = &px;
            if (tx.decode) {
                DecodeDxt(fi.host, px, pitch, mw, mh, md, decoded);
                pitch = mw * 4;
                src = &decoded;
            }
            // rows as Vulkan reads them packed: block rows for DXT, texel rows else
            uint32_t row = bc ? std::max(1u, (mw + 3) / 4) * block : mw * 4;
            uint32_t rows = bc ? std::max(1u, (mh + 3) / 4) : mh;
            size_t src_rows = pitch ? src->size() / pitch / md : 0;
            Alloc a = Upload(VkDeviceSize(row) * rows * md, 16);
            if (!a.cpu) continue;
            for (uint32_t z = 0; z < md; z++)
                for (uint32_t y = 0; y < rows; y++) {
                    uint8_t* dst = a.cpu + (size_t(z) * rows + y) * row;
                    if (y < src_rows) std::memcpy(dst, &(*src)[(size_t(z) * src_rows + y) * pitch], std::min(pitch, row));
                    else std::memset(dst, 0, row);
                }
            VkBufferImageCopy c{a.off, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, l, f, 1}, {0, 0, 0}, {mw, mh, md}};
            vkCmdCopyBufferToImage(Cb(), a.buf, tx.img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        }
    Transition(tx.img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

VkImageView VulkanBackend::BindGuestTexture(const State& s, int stage, const TextureDesc& d, const FormatInfo& fi,
                                            DrawTextures& stages) {
    static std::atomic<uint32_t> diag_guest_textures{0};
    const uint32_t diag_n = diag_guest_textures.fetch_add(1, std::memory_order_relaxed);
    if (diag_n < 32)
        LOGR("vulkan: guest-texture[%u] stage=%d header=%08X data=%08X size=%u format=%s dims=%ux%ux%u levels=%u linear=%u",
             diag_n, stage, s.textures[size_t(stage)], d.data, TotalSize(d), fi.name, d.width, d.height, d.depth,
             d.levels, d.linear);
    TexInfo& info = stages.info[stage];
    StagePalette palette;
    ReadStagePalette(s, stage, d, palette);
    if (d.format == 0x0B) {
        static std::atomic<uint32_t> diag_p8{0};
        uint32_t n = diag_p8.fetch_add(1, std::memory_order_relaxed);
        if (n < 12) {
            uint32_t header = s.palettes[size_t(stage)];
            LOGR("vulkan: p8-palette[%u] stage=%d tex=%08X data=%08X palette=%08X palette-data=%08X hash=%016llX colors=%08X/%08X/%08X/%08X",
                 n, stage, s.textures[size_t(stage)], d.data, header,
                 header ? kernel::R32(header + 4) : 0, static_cast<unsigned long long>(palette.hash),
                 palette.colours[0], palette.colours[1], palette.colours[2], palette.colours[3]);
        }
        static std::set<uint32_t> dumped_p8;
        if (d.width == 512 && d.height == 512 && dumped_p8.insert(d.data).second) {
            const char* root = "/storage/emulated/0/Android/data/com.racas.forzamotorsport/files/forza-debug";
            std::error_code ec;
            std::filesystem::create_directories(root, ec);
            char name[96];
            std::snprintf(name, sizeof(name), "p8-%08X.raw", d.data);
            std::ofstream raw(std::filesystem::path(root) / name, std::ios::binary);
            const auto* bytes = static_cast<const uint8_t*>(kernel::HostPtr(d.data));
            if (raw && bytes) raw.write(reinterpret_cast<const char*>(bytes), 512 * 512);
            TextureDesc dump_desc = d;
            std::vector<uint8_t> decoded;
            uint32_t pitch = 0;
            DecodeLevel(dump_desc, 0, 0, bytes, palette.va ? palette.colours : nullptr, decoded, pitch);
            std::snprintf(name, sizeof(name), "p8-%08X-decoded.rgba", d.data);
            std::ofstream rgba(std::filesystem::path(root) / name, std::ios::binary);
            if (rgba && !decoded.empty()) rgba.write(reinterpret_cast<const char*>(decoded.data()), decoded.size());
            LOGR("vulkan: p8-dump data=%08X raw=%d decoded=%d pitch=%u palette=%08X dir-error=%d",
                 d.data, raw.good(), rgba.good(), pitch, palette.va, int(ec.value()));
        }
    }
    const uint64_t key = uint64_t(palette.va) << 32 | d.data;
    Tex& tx = textures_[key];
    bool same = SameTexture(tx, d);
    if (!same && !CreateTexture(tx, d, fi, key)) return VK_NULL_HANDLE;
    Tex& t = textures_[key];
    if (t.frame != presents_ + 1 || !same || t.palette != palette.hash) {
        t.frame = presents_ + 1;
        t.palette = palette.hash;
        uint64_t sig = MemSig(d.data, TotalSize(d)) ^ DirtyEpoch(d.data) ^ palette.hash;
        if (sig != t.sig || !same) {
            t.sig = sig;
            UploadTexture(t, d, fi, static_cast<const uint8_t*>(kernel::HostPtr(d.data)), palette.va ? palette.colours : nullptr);
        }
    }
    // Xbox cubemaps may be assembled from six render-target faces. Their guest
    // memory is not necessarily current while the renderer owns the face, so
    // uploading the cube as an ordinary guest texture can leave reflections
    // stale (or black). Preserve guest-uploaded faces without a live target and
    // replace only faces backed by color targets with their current Vulkan image.
    if (d.cube && d.depth == 1 && !fi.compressed && !fi.depth) {
        bool target_face = false;
        // A car material may combine the guest texture with a cube assembled
        // from six render targets. Capture those live faces during a requested
        // frame trace so reflection defects can be separated from combiner or
        // cube-coordinate translation defects.
        static std::set<uint32_t> traced_cubes;
        const bool trace_cube = !trace_dir_.empty() && draw_rt_ && !draw_rt_->depth &&
                                draw_rt_->gw == 640 && draw_rt_->gh == 480 &&
                                trace_draw_ >= 400 && trace_draw_ <= 630 &&
                                traced_cubes.size() < 3 && traced_cubes.insert(d.data).second;
        for (uint32_t face = 0; face < 6; ++face) {
            const uint32_t face_data = d.data + face * FaceSize(d);
            Target* source = FindTargetByData(face_data, false);
            if (!source || source->depth || !source->img) continue;
            target_face = true;
            if (t.cube_img != t.img.image) {
                t.cube_img = t.img.image;
                for (uint64_t& s : t.cube_face_serial) s = 0;
            }
            // the face already holds what the target rendered last: copying it again only costs bandwidth, and the
            // layout transitions around the copy can end the pass the draw is in
            if (t.cube_face_serial[face] == source->content_serial) continue;
            t.cube_face_serial[face] = source->content_serial;
            Transition(source->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            Transition(t.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.srcOffsets[1] = {int32_t(source->w), int32_t(source->h), 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, face, 1};
            blit.dstOffsets[1] = {int32_t(d.width), int32_t(d.height), 1};
            vkCmdBlitImage(Cb(), source->img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           t.img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
            ++perf2_cube_blits_;
            static std::set<uint64_t> logged_cube_faces;
            const uint64_t face_key = (uint64_t(d.data) << 8) | face;
            if (logged_cube_faces.insert(face_key).second)
                LOGR("vulkan: cubemap-target-face data=%08X face=%u source=%08X size=%ux%u content=%llu",
                     d.data, face, face_data, source->gw, source->gh,
                     static_cast<unsigned long long>(source->content_serial));
            if (trace_cube) {
                char face_name[80];
                std::snprintf(face_name, sizeof(face_name), "cubemap-%08X-face-%u", d.data, face);
                TraceImage(source->img, face_name);
            }
        }
        if (target_face) Transition(t.img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    Transition(t.img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (fi.depth) info.shadow = ShadowDepthKind(d.format);
    stages.fmt[stage] = t.fmt;
    stages.levels[stage] = t.levels;
    return t.img.view;
}

VkImageView VulkanBackend::BindTexture(const State& s, int stage, DrawTextures& stages) {
    TexInfo& info = stages.info[stage];
    info = {};
    stages.rt[stage] = false;
    stages.gw[stage] = 0;
    stages.k[stage] = 1;
    stages.wide[stage] = 0;
    stages.fmt[stage] = kBgra;
    stages.levels[stage] = 1;
    TextureDesc d;
    if (!s.textures[size_t(stage)] || !Describe(s.textures[size_t(stage)], d)) {
        static std::atomic<uint32_t> bad_frontend_texture{0};
        if (stage == 0 && IsForzaFrontendShader(s)) {
            uint32_t n = bad_frontend_texture.fetch_add(1, std::memory_order_relaxed);
            if (n < 12) LOGR("vulkan: frontend-texture[%u] invalid header=%08X", n, s.textures[size_t(stage)]);
        }
        return VK_NULL_HANDLE;
    }
    static std::atomic<uint32_t> frontend_texture{0};
    if (stage == 0 && IsForzaFrontendShader(s)) {
        uint32_t n = frontend_texture.fetch_add(1, std::memory_order_relaxed);
        if (n < 24) LOGR("vulkan: frontend-texture[%u] header=%08X data=%08X format=%08X dims=%ux%u levels=%u linear=%u",
                         n, s.textures[size_t(stage)], d.data, d.format, d.width, d.height, d.levels, d.linear);
    }
    const FormatInfo& fi = Format(d.format);
    info.kind = TextureKind(d);
    // Forza's swizzled P8 frontend atlas feeds texel-space UVs (e.g. 1,1),
    // while Vulkan samples normalized coordinates. Convert those to the
    // texel-center range just as we do for linear rectangle textures.
    info.rect = d.linear || d.format == 0x0B;
    VkImageView v = VK_NULL_HANDLE;
    const char* path = "dummy";
    auto tit = targets_.find(TargetMapKey(d.data, fi.depth, d.width, d.height));
    if (tit != targets_.end() && !d.cube && d.depth == 1 && !fi.compressed) {
        v = BindTargetTexture(s, stage, d, tit->second, true, stages);
        path = "target";
    } else {
        v = BindGuestTexture(s, stage, d, fi, stages);
        path = v ? "guest" : "dummy";
    }
    if (!trace_dir_.empty()) {
        char alias_str[32] = "-";
        for (const auto& kv : targets_) {
            if (kv.second.data == d.data && (kv.second.gw != d.width || kv.second.gh != d.height)) {
                snprintf(alias_str, sizeof(alias_str), "%ux%u", kv.second.gw, kv.second.gh);
                break;
            }
        }
        char cube_target_str[32] = "-";
        if (d.cube) {
            for (const auto& kv : targets_) {
                if (kv.second.data >= d.data && kv.second.data < d.data + TotalSize(d)) {
                    snprintf(cube_target_str, sizeof(cube_target_str), "%08X(%ux%u)", kv.second.data, kv.second.gw, kv.second.gh);
                    break;
                }
            }
        }
        LOGR("vulkan: frame-trace tex draw=%u stage=%d header=%08X data=%08X fmt=%02X %ux%ux%u cube=%u lv=%u path=%s alias=%s cubetgt=%s view=%p",
             trace_draw_, stage, s.textures[size_t(stage)], d.data, d.format, d.width, d.height, d.depth,
             d.cube, d.levels, path, alias_str, cube_target_str, (void*)v);
    }
    return v;
}

bool VulkanBackend::BindStageTextures(const State& s, DrawTextures& stages) {
    for (int i = 0; i < 4; i++) stages.view[i] = BindTexture(s, i, stages);
    // a stage without a texture reads a black one of its kind
    for (int i = 0; i < 4; i++) {
        if (stages.view[i]) continue;
        uint8_t kind = stages.info[i].kind;
        stages.view[i] = dummy_[kind == kTex3D ? 1 : kind == kTexCube ? 2 : 0].view;
        if (stages.info[i].shadow) stages.info[i].shadow = 0;
        if (!trace_dir_.empty()) {
            LOGR("vulkan: frame-trace tex-dummy draw=%u stage=%d kind=%u view=%p",
                 trace_draw_, i, kind, (void*)stages.view[i]);
        }
    }
    return true;
}

constexpr float kDepth24Max = 16777215.0f;

void VulkanBackend::KeyDepthStencil(const State& s, const Assembled& a, PsoKey& k) {
    k.zen = s.rs[kRsZEnable] != 0;
    if (k.zen) {
        k.zwrite = SimpleState(s, kMZWriteEnable) != 0;
        k.zfunc = SimpleState(s, kMZFunc);
    }
    k.sten = s.rs[kRsStencilEnable] != 0;
    if (k.sten) {
        k.sfunc = SimpleState(s, kMStencilFunc);
        k.smask = SimpleState(s, kMStencilMask) & 0xFF;
        k.swmask = SimpleState(s, kMStencilWriteMask) & 0xFF;
        k.sfail = s.rs[kRsStencilFail];
        k.szfail = SimpleState(s, kMStencilZFail);
        k.spass = SimpleState(s, kMStencilPass);
    }
    if (g_diag_nodepth) {
        // diagnostic: drop depth testing and writing for every draw
        k.zen = 0;
        k.zwrite = 0;
    }
    if (a.topology == Topology::Triangles && SimpleState(s, kMSolidOffsetEnable)) {
        k.zbias = int32_t(std::lround(Bits(SimpleState(s, kMDepthOffset)) * kDepth24Max / s.zscale));
        k.zslope = Bits(SimpleState(s, kMSlopeScale)) * (draw_ds_ ? float(draw_ds_->w) / draw_ds_->gw : 1.0f);
    }
}

void VulkanBackend::BuildPipelineKey(const State& s, const Assembled& a, const DrawShaders& sh, PsoKey& k) {
    std::memset(&k, 0, sizeof(k));
    k.vs = sh.vh;
    k.ps = sh.ph;
    k.regs = a.regs;
    for (int r = 0; r < 16; r++)
        if ((a.regs >> r) & 1) {
            const VertexElement& e = a.elements[size_t(r)];
            k.elements[r] = e.format | uint32_t(e.buffer) << 8 | uint32_t(e.offset) << 16;
        }
    k.buffers = a.buffer_count;
    for (uint32_t i = 0; i < a.buffer_count; i++) k.strides[i] = a.buffers[i].stride;
    k.topo = uint32_t(a.topology) | (a.strip ? 0x10 : 0);
    k.has_rt = draw_rt_ != nullptr;
    k.has_ds = draw_ds_ != nullptr;
    k.rtfmt = draw_rt_ ? uint32_t(draw_rt_->img.fmt) : 0;
    KeyBlend(s, k);
    if (k.has_ds) KeyDepthStencil(s, a, k);
    KeyRasterizer(s, a, k);
}

// the glow's last step: a render target at most half the frame's width added (blend ONE onto the frame) over a
// full size target. BLOOM off skips it, the glow passes before it then go unseen. A composite done any other way
// (one pass mixing both, alpha blended) is left alone, so the frame never goes black for it
bool VulkanBackend::GlowComposite(const State& s, const DrawTextures& stages) const {
    constexpr uint32_t kBlendOne = 1;
    if (!draw_rt_ || draw_rt_->gw < guest_w_ || !SimpleState(s, kMBlendEnable) || SimpleState(s, kMDestBlend) != kBlendOne)
        return false;
    for (int i = 0; i < 4; i++) {
        if (!stages.gw[i] || stages.gw[i] * 2 > draw_rt_->gw) continue;
        // each kind once in the log, to tell what it took
        static std::set<uint64_t> seen;
        uint64_t kind = uint64_t(stages.gw[i]) << 32 | SimpleState(s, kMSrcBlend) << 16 | draw_rt_->gw;
        if (seen.size() < 16 && seen.insert(kind).second)
            LOGR("vulkan: bloom off skips %u wide onto %u wide, src blend %u", stages.gw[i], draw_rt_->gw,
                 SimpleState(s, kMSrcBlend));
        return true;
    }
    return false;
}

void VulkanBackend::Draw(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) {
    static std::atomic<uint32_t> diag_draws{0};
    uint32_t diag_n = diag_draws.fetch_add(1, std::memory_order_relaxed);
    if (diag_n < 8) LOGR("vulkan: draw[%u] prim=%u start=%u count=%u indexed=%d rt=%08X size=%u fmt=%u", diag_n,
                        prim, start, count, indices != nullptr, s.rt.data, s.rt.size, s.rt.format);
    // Indexed geometry first appears when the track and car load. Sample it
    // separately from immediate quads, which fill the post-input log window.
    static uint32_t indexed_draws = 0;
    const uint32_t mesh = indices ? ++indexed_draws : 0;
    const bool trace_mesh = mesh && (mesh <= 16 || mesh % 100000 == 0);
    if (trace_mesh)
        LOGR("vulkan: indexed[%u] prim=%u count=%u base=%u vs=%08X/%zu ps=%08X stream0=%08X/%u rt=%08X zs=%08X vp=%u,%u,%u,%u",
             mesh, prim, count, s.base_vertex, s.vs_handle, s.vs ? s.vs->program.size() : 0,
             s.ps ? s.ps->handle : 0, s.streams[0].vb, s.streams[0].stride, s.rt.data, s.zs.data,
             s.viewport.x, s.viewport.y, s.viewport.w, s.viewport.h);
    if (DepthCopy(s)) {
        if (trace_mesh) LOGR("vulkan: indexed[%u] consumed by DepthCopy", mesh);
        return;
    }
    Assembled& a = asm_;
    if (!Assemble(s, prim, start, count, indices, a)) {
        if (trace_mesh) LOGR("vulkan: indexed[%u] rejected by Assemble", mesh);
        return;
    }

    // Capture a small generic window after controller input. The frontend-only
    // probe misses the fixed-function draws used by the loaded race scene.
    const uint32_t post_input_n = gpu::g_post_input_draw_diag.load(std::memory_order_acquire);
    if (post_input_n < 128) {
        const uint32_t n = gpu::g_post_input_draw_diag.fetch_add(1, std::memory_order_acq_rel);
        if (n < 32) {
            TextureDesc rd{};
            const bool has_rt_desc = Describe(s.rt.header, rd);
            LOGR("vulkan: post-input-draw[%u] guestdraw=%u prim=%u start=%u count=%u indices=%u assembled=%u verts=%u regs=%04X decode=%08llX buffers=%u rt=%08X/%08X/%08X desc=%u:%ux%u pitch=%u fmt=%u zs=%08X/%08X/%08X vp=%u,%u,%u,%u scissor=%u vs=%08X/%zu ps=%08X color=%08X blend=%u:%u,%u cull=%u z=%u:%u,%u tex=%08X/%08X pal=%08X",
                 n, diag_n, prim, start, count, indices != nullptr, uint32_t(a.topology), a.vertex_count,
                 a.regs, static_cast<unsigned long long>(a.decode), a.buffer_count, s.rt.data, s.rt.size, s.rt.format,
                 has_rt_desc, rd.width, rd.height, rd.pitch, rd.format,
                 s.zs.data, s.zs.size, s.zs.format, s.viewport.x, s.viewport.y, s.viewport.w, s.viewport.h,
                 s.scissor_count, s.vs_handle, s.vs ? s.vs->program.size() : 0,
                 s.ps ? s.ps->handle : 0, SimpleState(s, kMColorWriteEnable),
                 SimpleState(s, kMBlendEnable), SimpleState(s, kMSrcBlend), SimpleState(s, kMDestBlend),
                 s.rs[kRsCullMode], s.rs[kRsZEnable], SimpleState(s, kMZWriteEnable),
                 SimpleState(s, kMZFunc), s.textures[0], s.textures[1], s.palettes[0]);
            for (uint32_t r : {0u, 3u, 4u, 9u}) {
                if (!(a.regs & (1u << r))) continue;
                const VertexElement& e = a.elements[r];
                if (e.buffer >= a.buffer_count) continue;
                const VertexBuffer& b = a.buffers[e.buffer];
                float v[4]{};
                if (!b.src || e.offset + sizeof(v) > b.stride || b.bytes < e.offset + sizeof(v)) continue;
                std::memcpy(v, b.src + e.offset, sizeof(v));
                LOGR("vulkan: post-input-vertex[%u] reg=%u fmt=%u buffer=%u offset=%u stride=%u first=(%.5f,%.5f,%.5f,%.5f)",
                     n, r, e.format, e.buffer, e.offset, b.stride, v[0], v[1], v[2], v[3]);
            }
        }
    }

    // The Forza front-end uses an immediate custom vertex shader. Log the actual
    // input registers after assembly, rather than only the push-buffer values, so
    // a device capture can distinguish bad guest data from packing/layout loss.
    static std::atomic<uint32_t> diag_forza_inputs{0};
    if (IsForzaFrontendShader(s)) {
        uint32_t n = diag_forza_inputs.fetch_add(1, std::memory_order_relaxed);
        if (n < 16) {
            LOGR("vulkan: forza-input[%u] regs=%04X vertices=%u buffers=%u converted=%u stride0=%u",
                 n, a.regs, a.vertex_count, a.buffer_count, !a.vertices.empty(),
                 a.buffer_count ? a.buffers[0].stride : 0);
            LOGR("vulkan: forza-vs-constants[%u] c96=(%.5f,%.5f,%.5f,%.5f) c97=(%.5f,%.5f,%.5f,%.5f)",
                 n, s.vsc[96][0], s.vsc[96][1], s.vsc[96][2], s.vsc[96][3],
                 s.vsc[97][0], s.vsc[97][1], s.vsc[97][2], s.vsc[97][3]);
            for (uint32_t r : {0u, 3u, 4u, 9u}) {
                if (!(a.regs & (1u << r))) continue;
                const VertexElement& e = a.elements[r];
                if (e.buffer >= a.buffer_count) continue;
                const VertexBuffer& b = a.buffers[e.buffer];
                float v[4]{};
                if (!b.src || e.offset + sizeof(v) > b.stride || b.bytes < e.offset + sizeof(v)) continue;
                std::memcpy(v, b.src + e.offset, sizeof(v));
                LOGR("vulkan: forza-input-v[%u] reg=%u fmt=%u buffer=%u offset=%u value=(%.5f,%.5f,%.5f,%.5f)",
                     n, r, e.format, e.buffer, e.offset, v[0], v[1], v[2], v[3]);
            }
        }
        uint32_t armed = gpu::g_frontend_draw_diag.load(std::memory_order_acquire);
        if (armed) {
            armed = gpu::g_frontend_draw_diag.fetch_add(1, std::memory_order_acq_rel);
            if (armed <= 48) {
                TextureDesc rd{};
                const bool described_rt = Describe(s.rt.header, rd);
                LOGR("vulkan: frontend-after-input[%u] draw=%u prim=%u count=%u rt=%08X/%08X/%08X desc=%u:%ux%u pitch=%u fmt=%u vp=%u,%u,%u,%u scissor=%u regs=%04X verts=%u converted=%u tex=%08X pal=%08X tss=%u,%u,%u,%u,%u,%u,%u,%u,%u,%u ps=%08X",
                     armed, diag_n, prim, count, s.rt.data, s.rt.size, s.rt.format, described_rt, rd.width, rd.height,
                     rd.pitch, rd.format, s.viewport.x, s.viewport.y,
                     s.viewport.w, s.viewport.h, s.scissor_count, a.regs, a.vertex_count, !a.vertices.empty(),
                     s.textures[0], s.palettes[0], s.tss[0][kTssColorOp], s.tss[0][kTssColorArg1],
                     s.tss[0][kTssColorArg2], s.tss[0][kTssAlphaOp], s.tss[0][kTssAlphaArg1],
                     s.tss[0][kTssAlphaArg2], s.tss[0][kTssAddressU], s.tss[0][kTssAddressV],
                     s.tss[0][kTssMinFilter], s.tss[0][kTssMagFilter], s.ps ? s.ps->handle : 0);
                if (armed == 1) {
                    gpu::g_capture_front_after_input.store(true, std::memory_order_release);
                    for (uint32_t vi = 0; vi < std::min<uint32_t>(a.vertex_count, 4); ++vi) {
                        float value[3][4]{};
                        if (a.vertices.empty()) {
                            for (uint32_t j = 0; j < 3; ++j) {
                                const uint32_t r = j == 0 ? 0 : j == 1 ? 3 : 4;
                                const VertexElement& e = a.elements[r];
                                if (!(a.regs & (1u << r)) || e.buffer >= a.buffer_count) continue;
                                const VertexBuffer& b = a.buffers[e.buffer];
                                const size_t off = size_t(vi) * b.stride + e.offset;
                                if (!b.src || off + sizeof(value[j]) > b.bytes) continue;
                                std::memcpy(value[j], b.src + off, sizeof(value[j]));
                            }
                        }
                        LOGR("vulkan: frontend-after-input-vertex[%u:%u] v0=(%.4f,%.4f,%.4f,%.4f) v3=(%.4f,%.4f,%.4f,%.4f) v4=(%.4f,%.4f,%.4f,%.4f)",
                             armed, vi, value[0][0], value[0][1], value[0][2], value[0][3], value[1][0],
                             value[1][1], value[1][2], value[1][3], value[2][0], value[2][1], value[2][2],
                             value[2][3]);
                    }
                }
            }
        }
    }

    // the targets, CPU writes to them first
    Target* rt = EnsureTarget(s.rt, false);
    Target* ds = s.zs.header ? EnsureTarget(s.zs, true) : nullptr;
    // Xbox titles may provide an oversized depth surface (Forza's is 1280x480
    // for a 640x480 color target). Vulkan framebuffer attachments may exceed
    // the framebuffer extent; reject only surfaces that cannot cover it.
    if (rt && ds && (ds->w < rt->w || ds->h < rt->h)) {
        LOGR("vulkan: depth attachment too small color=%ux%u depth=%ux%u; disabling",
             rt->w, rt->h, ds->w, ds->h);
        ds = nullptr;
    }
    if (!rt && !ds) {
        if (trace_mesh) LOGR("vulkan: indexed[%u] missing attachments", mesh);
        return;
    }
    if (rt && LatchCpuWrite(*rt)) UploadTarget(*rt, s.rt);
    draw_rt_ = rt;
    draw_ds_ = ds;

    DrawShaders sh = ClassifyShaders(s, a);
    if (sh.prog) EnsureProgramHash(s);
    sh.vkey = VertexKey(s, a, sh);
    EnsureVertexShader(s, a, sh);

    if (!trace_dir_.empty()) ++trace_draw_;
    DrawTextures stages;
    if (!BindStageTextures(s, stages) || !draw_rt_ && !draw_ds_) {
        if (trace_mesh) LOGR("vulkan: indexed[%u] texture binding failed", mesh);
        return;
    }
    if (post_input_n < 32) {
        LOGR("vulkan: post-input-resources[%u] fixed=%u vh=%016llX ph=%016llX rt=%p:%ux%u ds=%p:%ux%u tex-kind=%u/%u/%u/%u views=%u/%u/%u/%u",
             post_input_n, sh.fixed_transform, static_cast<unsigned long long>(sh.vh),
             static_cast<unsigned long long>(sh.ph),
             static_cast<void*>(draw_rt_), draw_rt_ ? draw_rt_->w : 0, draw_rt_ ? draw_rt_->h : 0,
             static_cast<void*>(draw_ds_), draw_ds_ ? draw_ds_->w : 0, draw_ds_ ? draw_ds_->h : 0,
             stages.info[0].kind, stages.info[1].kind, stages.info[2].kind, stages.info[3].kind,
             stages.view[0] != VK_NULL_HANDLE, stages.view[1] != VK_NULL_HANDLE,
             stages.view[2] != VK_NULL_HANDLE, stages.view[3] != VK_NULL_HANDLE);
    }
    // Forza's immediate frontend vertex shader (B01D04F1) sources stage 0 UVs
    // from v0.zw. Some P8 draws provide normalized atlas coordinates while
    // others provide texel coordinates; treating every P8 texture as a rect
    // divides normalized coordinates by the atlas dimensions a second time.
    if (IsForzaFrontendShader(s) && stages.info[0].rect) {
        TextureDesc td{};
        if (Describe(s.textures[0], td) && td.format == 0x0B && (a.regs & 1)) {
            const VertexElement& e = a.elements[0];
            float max_uv = 0.0f;
            if (e.buffer < a.buffer_count) {
                const VertexBuffer& b = a.buffers[e.buffer];
                for (uint32_t vi = 0; b.src && vi < a.vertex_count; ++vi) {
                    const size_t off = size_t(vi) * b.stride + e.offset + sizeof(float) * 2;
                    float uv[2]{};
                    if (off + sizeof(uv) > b.bytes) break;
                    std::memcpy(uv, b.src + off, sizeof(uv));
                    max_uv = std::max({max_uv, std::fabs(uv[0]), std::fabs(uv[1])});
                }
            }
            const bool texel_space = max_uv > 1.0f;
            stages.info[0].rect = texel_space;
            if (gpu::g_frontend_draw_diag.load(std::memory_order_acquire) <= 49)
                LOGR("vulkan: frontend-p8-uv max=%.6f dims=%ux%u space=%s rect=%u",
                     max_uv, td.width, td.height, texel_space ? "texel" : "normalized", stages.info[0].rect);
        }
    }
    if (diag_n < 16)
        LOGR("vulkan: draw-state[%u] vs=%08X vs-program=%zu ps=%08X tex=%08X/%08X/%08X/%08X stage-kind=%u/%u/%u/%u views=%d/%d/%d/%d",
             diag_n, s.vs_handle, s.vs ? s.vs->program.size() : 0, s.ps ? s.ps->handle : 0,
             s.textures[0], s.textures[1], s.textures[2], s.textures[3], stages.info[0].kind,
             stages.info[1].kind, stages.info[2].kind, stages.info[3].kind,
             stages.view[0] != VK_NULL_HANDLE, stages.view[1] != VK_NULL_HANDLE,
             stages.view[2] != VK_NULL_HANDLE, stages.view[3] != VK_NULL_HANDLE);
    if (!game::Opt().bloom && draw_rt_ && (draw_rt_->gw == 320 && draw_rt_->gh == 240) && !draw_rt_->depth) return;
    if (!game::Opt().bloom && GlowComposite(s, stages)) {
        if (trace_mesh) LOGR("vulkan: indexed[%u] skipped by GlowComposite", mesh);
        return;
    }
    PixelKey pk = MakePixelKey(s, stages.info);
    sh.pkey = Hash(&pk, sizeof(pk));
    EnsurePixelShader(pk, sh);
    if (!sh.vs || !sh.ps) {
        if (trace_mesh) LOGR("vulkan: indexed[%u] missing shader vs=%p ps=%p", mesh, sh.vs, sh.ps);
        return;
    }

    PsoKey k;
    BuildPipelineKey(s, a, sh, k);
    if (IsForzaFrontendShader(s) && gpu::g_frontend_draw_diag.load(std::memory_order_acquire) == 2) {
        LOGR("vulkan: frontend-pipeline-state cw=%08X wmask=%X blend=%u src=%u dst=%u op=%u cull=%u front-ccw=%u fill=%u z=%u zwrite=%u zfunc=%u stencil=%u rt=%08X fmt=%u topo=%u strip=%u index=%u draw=%u",
             SimpleState(s, kMColorWriteEnable), k.wmask, k.blend, k.src, k.dst, k.op, k.cull, k.ccw, k.fill,
             k.zen, k.zwrite, k.zfunc, k.sten, s.rt.data, k.rtfmt, k.topo & 0xF, k.topo >> 4,
             a.index_count, a.draw_count);
        for (uint32_t r : {0u, 3u, 4u}) {
            const VertexElement& e = a.elements[r];
            LOGR("vulkan: frontend-pipeline-attribute r=%u present=%u fmt=%u buffer=%u offset=%u stride=%u decode=%u",
                 r, !!(a.regs & (1u << r)), e.format, e.buffer, e.offset,
                 e.buffer < a.buffer_count ? a.buffers[e.buffer].stride : 0,
                 uint32_t((a.decode >> (r * 4)) & 0xF));
        }
    }
    VkPipeline pipe = EnsurePipeline(k, sh.vs, sh.ps);
    if (trace_mesh)
        LOGR("vulkan: indexed[%u] pipeline=%p prog=%u vkey=%016llX pkey=%016llX regs=%04X verts=%u indices=%u mask=%X cull=%u depth=%u fixed=%u tex=%u/%u/%u/%u modes=%08X",
             mesh, pipe, sh.prog, static_cast<unsigned long long>(sh.vkey),
             static_cast<unsigned long long>(sh.pkey), a.regs, a.vertex_count, a.index_count,
             k.wmask, k.cull, k.zen, sh.fixed_transform, stages.info[0].kind, stages.info[1].kind,
             stages.info[2].kind, stages.info[3].kind, pk.tex_modes);
    if (IsForzaFrontendShader(s) && gpu::g_frontend_draw_diag.load(std::memory_order_acquire) == 2)
        LOGR("vulkan: frontend-pipeline-ready pipe=%p vs=%p ps=%p", pipe, sh.vs, sh.ps);
    if (!pipe) return;

    uint32_t capture_mesh = 0;
    uint32_t capture_id = 0;
    const uint32_t capture_request = gpu::g_capture_indexed_color.load(std::memory_order_acquire);
    if (capture_request && indices && a.index_count >= 32 && sh.prog && (k.wmask & 7) &&
        draw_rt_ && draw_rt_->gw == 640 && draw_rt_->gh == 480 && s.viewport.w >= 320) {
        bool matching_material = true;
        if (capture_request == 1) {
            TextureDesc diffuse{}, secondary{};
            matching_material = s.ps && s.ps->handle == 0x0066F2A0 &&
                                Describe(s.textures[0], diffuse) && diffuse.format == 0x0F &&
                                Describe(s.textures[1], secondary) && secondary.format == 0x0C;
            if (matching_material)
                capture_mesh = gpu::g_capture_indexed_color.exchange(0, std::memory_order_acq_rel);
        } else if (capture_request == 3) {
            TextureDesc diffuse{}, secondary{};
            matching_material = s.vs && s.ps;
            const bool has_diffuse = matching_material && Describe(s.textures[0], diffuse);
            const bool has_secondary = matching_material && Describe(s.textures[1], secondary);
            uint32_t budget = gpu::g_capture_indexed_sequence.load(std::memory_order_acquire);
            if (matching_material && budget) {
                static std::set<uint64_t> sampled_materials;
                static uint64_t sampled_frame = UINT64_MAX;
                static uint32_t window_unique = 0;
                if (sampled_frame != presents_) {
                    sampled_materials.clear();
                    sampled_frame = presents_;
                    window_unique = 0;
                }
                uint64_t pair = (uint64_t(has_diffuse ? diffuse.data : 0) << 32) |
                                uint64_t(has_secondary ? secondary.data : 0);
                pair ^= sh.vkey * 0x9E3779B185EBCA87ull;
                pair ^= sh.pkey * 0xC2B2AE3D27D4EB4Full;
                pair ^= uint64_t(s.vs_handle) << 17;
                pair ^= uint64_t(s.ps->handle) << 1;
                if (sampled_materials.insert(pair).second) {
                    const uint32_t ordinal = window_unique++;
                    const uint32_t skip = gpu::g_capture_indexed_sequence_skip.load(std::memory_order_acquire);
                    if (ordinal >= skip) {
                        while (budget && !gpu::g_capture_indexed_sequence.compare_exchange_weak(
                                              budget, budget - 1, std::memory_order_acq_rel)) {}
                        if (budget) {
                            capture_id = 16 - budget;
                            capture_mesh = 3;
                        }
                    }
                }
            }
        } else if (capture_request == 2) {
            capture_mesh = gpu::g_capture_indexed_color.exchange(0, std::memory_order_acq_rel);
        }
    }
    bool forced_color = false;
    if (capture_mesh == 2) {
        PsoKey solid = k;
        const VkShaderModule pixel = Module("float4 main() : SV_Target0 { return float4(1,0,1,1); }",
                                            "main", vkshader::Stage::Pixel, solid.ps);
        solid.wmask = 15;
        solid.blend = solid.cull = solid.zen = solid.zwrite = solid.sten = 0;
        if (pixel) {
            if (VkPipeline probe = EnsurePipeline(solid, sh.vs, pixel)) {
                pipe = probe;
                forced_color = true;
            }
        }
    }
    std::vector<uint8_t> mesh_before;
    char before_name[80] = "forza-indexed-before.bmp";
    if (capture_mesh == 3)
        std::snprintf(before_name, sizeof(before_name), "forza-material-%02u-before.bmp", capture_id);
    const bool captured_before = capture_mesh && CaptureIndexedColor(*draw_rt_, before_name, mesh_before);
    if (capture_mesh) {
        const auto dir = app::ExeDir();
        const std::string prefix = capture_mesh == 3 ? "forza-material-" + std::to_string(capture_id) + "-"
                                                      : "forza-indexed-";
        std::ofstream(dir / (prefix + "vertex.hlsl")) << VertexHlsl(s, a, sh);
        std::ofstream(dir / (prefix + "pixel.hlsl")) << PixelHlsl(pk);
        std::ofstream constants(dir / (prefix + "constants.bin"), std::ios::binary);
        constants.write(reinterpret_cast<const char*>(s.vsc.data()), sizeof(s.vsc));
        LOGR("vulkan: material-capture id=%u guestdraw=%u rt=%08X/%ux%u count=%u verts=%u regs=%04X vs=%08X/%zu vh=%016llX ps=%08X ph=%016llX vkey=%016llX pkey=%016llX cull=%u depth=%u:%u,%u blend=%u:%u,%u color=%08X fog=%u/%u",
             capture_id, diag_n, draw_rt_->data, draw_rt_->gw, draw_rt_->gh, a.index_count, a.vertex_count,
             a.regs, s.vs_handle, s.vs ? s.vs->program.size() : 0,
             static_cast<unsigned long long>(sh.vh), s.ps ? s.ps->handle : 0,
             static_cast<unsigned long long>(sh.ph), static_cast<unsigned long long>(sh.vkey),
             static_cast<unsigned long long>(sh.pkey), s.rs[kRsCullMode], s.rs[kRsZEnable],
             SimpleState(s, kMZWriteEnable), SimpleState(s, kMZFunc), SimpleState(s, kMBlendEnable),
             SimpleState(s, kMSrcBlend), SimpleState(s, kMDestBlend), SimpleState(s, kMColorWriteEnable),
             s.rs[kRsFogEnable], s.rs[kRsFogTableMode]);
        for (int stage = 0; stage < 4; ++stage) {
            TextureDesc td{};
            if (!s.textures[size_t(stage)] || !Describe(s.textures[size_t(stage)], td)) continue;
            StagePalette palette{};
            ReadStagePalette(s, stage, td, palette);
            std::vector<uint8_t> encoded, decoded;
            uint32_t row_pitch = 0;
            const auto* source = static_cast<const uint8_t*>(kernel::HostPtr(td.data));
            if (source) {
                DecodeLevel(td, 0, 0, source, palette.va ? palette.colours : nullptr, encoded, row_pitch);
                if (Format(td.format).compressed)
                    DecodeDxt(Format(td.format).host, encoded, row_pitch, td.width, td.height, 1, decoded);
                else
                    decoded = std::move(encoded);
            }
            uint64_t rgb[3]{};
            uint8_t lo[3] = {255, 255, 255}, hi[3]{};
            uint32_t samples = 0;
            for (size_t p = 0; p + 3 < decoded.size(); p += 4) {
                for (int ch = 0; ch < 3; ++ch) {
                    const uint8_t value = decoded[p + ch];
                    rgb[ch] += value;
                    lo[ch] = std::min(lo[ch], value);
                    hi[ch] = std::max(hi[ch], value);
                }
                ++samples;
            }
            const FormatInfo& fi = Format(td.format);
            LOGR("vulkan: indexed-texture stage=%d header=%08X data=%08X fmt=%08X(%s) dims=%ux%u pitch=%u levels=%u linear=%u swizzled=%u compressed=%u palette=%08X/%016llX decoded=%ux%u samples=%u mean=%u,%u,%u min=%u,%u,%u max=%u,%u,%u",
                 stage, s.textures[size_t(stage)], td.data, td.format, fi.name, td.width, td.height, td.pitch, td.levels,
                 td.linear, fi.swizzled, fi.compressed, palette.va, static_cast<unsigned long long>(palette.hash),
                 td.width, td.height, samples,
                 samples ? uint32_t(rgb[0] / samples) : 0, samples ? uint32_t(rgb[1] / samples) : 0,
                 samples ? uint32_t(rgb[2] / samples) : 0, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]);
            if (capture_mesh && !decoded.empty()) {
                char filename[80];
                if (capture_mesh == 3)
                    std::snprintf(filename, sizeof(filename), "forza-material-%02u-stage-%d.bgra", capture_id, stage);
                else
                    std::snprintf(filename, sizeof(filename), "forza-indexed-stage-%d.bgra", stage);
                std::ofstream dump(dir / filename, std::ios::binary);
                dump.write(reinterpret_cast<const char*>(decoded.data()), std::streamsize(decoded.size()));
                LOGR("vulkan: indexed-texture-dump stage=%d path=%s bytes=%zu saved=%u",
                     stage, filename, decoded.size(), dump.good());
            }
        }
    }

    bool trace_sample = false;
    std::string trace_name;
    if (!trace_dir_.empty() && draw_rt_) {
        const uint32_t n = trace_draw_;
        LOGR("vulkan: frame-depth draw=%u guest-enable=%u guest-zs=%08X/%08X/%08X attachment=%u color-size=%ux%u depth-size=%ux%u",
             n, s.rs[kRsZEnable], s.zs.data, s.zs.size, s.zs.format, draw_ds_ != nullptr,
             draw_rt_->w, draw_rt_->h, draw_ds_ ? draw_ds_->w : 0, draw_ds_ ? draw_ds_->h : 0);
        const bool screen_pass = count <= 4 && draw_rt_->gw >= 320 && !k.zen;
        const bool target_switch = trace_last_target_ != draw_rt_;
        trace_last_target_ = draw_rt_;
        LOGR("vulkan: frame-trace draw=%u rt=%08X/%ux%u desc=%08X/%08X serial=%llu count=%u vsh=%08X psh=%08X vs=%016llX ps=%016llX modes=%08X mask=%X blend=%u:%u/%u/%u cull=%u z=%u:%u,%u vp=%u,%u,%u,%u",
             n, draw_rt_->data, draw_rt_->gw, draw_rt_->gh, s.rt.size, s.rt.format,
             static_cast<unsigned long long>(draw_rt_->content_serial),
             count, s.vs_handle, s.ps ? s.ps->handle : 0,
             static_cast<unsigned long long>(sh.vh), static_cast<unsigned long long>(sh.ph),
             pk.tex_modes, k.wmask, k.blend, k.src, k.dst, k.op,
             k.cull, k.zen, k.zwrite, k.zfunc,
             s.viewport.x, s.viewport.y, s.viewport.w, s.viewport.h);
        for (int i = 0; i < 4; ++i) {
            TextureDesc d{};
            if (!Describe(s.textures[i], d)) continue;
            LOGR("vulkan: frame-trace texture draw=%u stage=%d data=%08X/%ux%u pitch=%u format=%u target=%u view=%p",
                 n, i, d.data, d.width, d.height, d.pitch, d.format, stages.rt[i], (void*)stages.view[i]);
        }
        // Global draw numbers change with reflection passes and other cars.
        // Sample relative to entering the main target, with dense coverage
        // of the first draws where a sky/background pass would occur.
        // the 1x1 and 320x240 targets are also announced as 640x480, so go by the image the draws land in
        const bool main_target = draw_rt_->gw == 640 && draw_rt_->gh == 480 && draw_rt_->img.w >= 640;
        if (target_switch) trace_scene_draw_ = 0;
        const uint32_t scene_draw = main_target ? trace_scene_draw_++ : 0;
        trace_relative_ = scene_draw;
        trace_sky_window_ = !trace_dir_.empty() && main_target && scene_draw < 128;
        const bool scene_sample = main_target &&
                                  (scene_draw < 20 || (scene_draw < 320 && scene_draw % 16 == 0));
        if (main_target)
            LOGR("vulkan: frame-scene draw=%u relative=%u", n, scene_draw);
        if (trace_sky_window_)
            LOGR("vulkan: frame-sky-state relative=%u count=%u vp=%u,%u,%u,%u sc=%u,%u,%u,%u cull=%u z=%u:%u,%u blend=%u mask=%X fog=%u/%u table=%u color=%08X start=%08X end=%08X density=%08X vs=%08X ps=%08X",
                 scene_draw, count, s.viewport.x, s.viewport.y, s.viewport.w, s.viewport.h,
                 s.scissors[0], s.scissors[1], s.scissors[2], s.scissors[3],
                 s.rs[kRsCullMode], s.rs[kRsZEnable], SimpleState(s, kMZWriteEnable), SimpleState(s, kMZFunc),
                 SimpleState(s, kMBlendEnable), SimpleState(s, kMColorWriteEnable),
                 s.rs[kRsFogEnable], game::Opt().fog ? 1u : 0u, s.rs[kRsFogTableMode], s.rs[kRsFogColor],
                 s.rs[kRsFogStart], s.rs[kRsFogEnd], s.rs[kRsFogDensity],
                 s.vs_handle, s.ps ? s.ps->handle : 0);
        trace_strip_ = g_diag_skystrip && trace_strip_count_ < 520;
        if (trace_strip_) {
            ++trace_strip_count_;
            trace_strip_rt_ = draw_rt_->data;
        }
        trace_sample = (scene_sample || (target_switch && draw_rt_->gw == 640 && draw_rt_->gh == 480)) &&
                       trace_samples_ < 40;
        if (trace_sample) {
            ++trace_samples_;
            trace_name = "draw-" + std::to_string(n);
            TraceImage(draw_rt_->img, trace_name + "-before");
            std::ofstream(trace_dir_ / (trace_name + "-ps.hlsl")) << PixelHlsl(pk);
            if (screen_pass || scene_sample)
                std::ofstream(trace_dir_ / (trace_name + "-vs.hlsl")) << VertexHlsl(s, a, sh);
            if (pk.tex_modes == 0x00018442 && pk.tex[0].kind == kTex3D) {
                TextureDesc volume{};
                if (Describe(s.textures[0], volume) && volume.depth > 1 && Format(volume.format).compressed) {
                    const auto* guest = static_cast<const uint8_t*>(kernel::HostPtr(volume.data));
                    std::vector<uint8_t> blocks, pixels;
                    uint32_t row_pitch = 0;
                    DecodeLevel(volume, 0, 0, guest, nullptr, blocks, row_pitch);
                    DecodeDxt(Format(volume.format).host, blocks, row_pitch, volume.width, volume.height,
                             volume.depth, pixels);
                    std::string prefix = "car-volume-" + std::to_string(volume.data);
                    std::ofstream(trace_dir_ / (prefix + ".dxt"), std::ios::binary)
                        .write(reinterpret_cast<const char*>(blocks.data()), std::streamsize(blocks.size()));
                    std::ofstream(trace_dir_ / (prefix + ".bgra"), std::ios::binary)
                        .write(reinterpret_cast<const char*>(pixels.data()), std::streamsize(pixels.size()));
                    uint64_t rgb[3]{};
                    for (size_t p = 0; p + 3 < pixels.size(); p += 4)
                        for (int ch = 0; ch < 3; ++ch) rgb[ch] += pixels[p + ch];
                    const uint64_t samples = pixels.size() / 4;
                    LOGR("vulkan: car-volume-probe draw=%u data=%08X fmt=%08X size=%ux%ux%u row=%u blocks=%zu pixels=%zu mean=%u,%u,%u",
                         n, volume.data, volume.format, volume.width, volume.height, volume.depth, row_pitch,
                         blocks.size(), pixels.size(), samples ? uint32_t(rgb[0] / samples) : 0,
                         samples ? uint32_t(rgb[1] / samples) : 0, samples ? uint32_t(rgb[2] / samples) : 0);
                }
            }
        }
    }
    // everything uploaded before the rendering: a full ring flushes
    VkDeviceSize need = 2 * kUboRange + 1024;
    for (uint32_t i = 0; i < a.buffer_count; i++) need += a.buffers[i].bytes + 16;
    need += VkDeviceSize(a.index_count) * 4 + 16;
    if (need < kUploadSize / 4 && CurrentFrame().used + need > kUploadSize) Submit(true, false);
    GuestExtent size;
    SurfaceSize(draw_rt_ ? s.rt : s.zs, size.w, size.h);
    DrawConstants c{};
    FillSurfaceConstants(s, size, c);
    FillFogConstants(s, sh.prog, c);
    FillCombinerConstants(s, c);
    FillTextureConstants(s, stages, c);
    if (!trace_dir_.empty() && draw_rt_ && draw_rt_->gw == 640 && draw_rt_->gh == 480 &&
        pk.tex_modes == 0x00018442 && pk.tex[0].kind == kTex3D &&
        sh.ph == 0x6A13EF336A4075FEull && trace_mesh_samples_ < 16) {
        ++trace_mesh_samples_;
        const std::string trace_name = "draw-" + std::to_string(trace_draw_);
        std::ofstream(trace_dir_ / (trace_name + "-vs.hlsl")) << VertexHlsl(s, a, sh);
        std::ofstream(trace_dir_ / (trace_name + "-ps.hlsl")) << PixelHlsl(pk);
        LOGR("vulkan: car-light-probe draw=%u specular-enable=%u misc-w=%.1f", trace_draw_,
             s.rs[kRsSpecularEnable], c.misc[3]);
        for (int i : {2, 3, 6, 7, 8, 14})
            LOGR("vulkan: car-light-pc draw=%u index=%d value=(%.6f,%.6f,%.6f,%.6f)", trace_draw_, i,
                 c.pc[i][0], c.pc[i][1], c.pc[i][2], c.pc[i][3]);
        std::ofstream(trace_dir_ / (trace_name + "-vertex-constants.bin"), std::ios::binary)
            .write(reinterpret_cast<const char*>(s.vsc.data()), sizeof(s.vsc));
        // Preserve the exact input-assembler bytes, including formats and base
        // vertex, so lighting can be evaluated without guessing guest packing.
        std::ofstream layout(trace_dir_ / (trace_name + "-mesh.txt"));
        layout << "vertices " << a.vertex_count << " indices " << a.index_count
               << " index16 " << a.index16 << " base " << a.base_vertex
               << " draw " << a.draw_count << " topology " << uint32_t(a.topology)
               << " strip " << a.strip << '\n';
        for (uint32_t r = 0; r < 16; ++r) {
            if (!(a.regs & (1u << r))) continue;
            const auto& e = a.elements[r];
            layout << "attribute " << r << " format " << uint32_t(e.format)
                   << " buffer " << uint32_t(e.buffer) << " offset " << e.offset
                   << " decode " << ((a.decode >> (r * 4)) & 15) << '\n';
        }
        for (uint32_t i = 0; i < a.buffer_count; ++i) {
            const auto& b = a.buffers[i];
            layout << "buffer " << i << " stride " << b.stride << " bytes " << b.bytes << '\n';
            if (b.src && b.bytes)
                std::ofstream(trace_dir_ / (trace_name + "-vb-" + std::to_string(i) + ".bin"), std::ios::binary)
                    .write(reinterpret_cast<const char*>(b.src), b.bytes);
        }
        if (a.index_src && a.index_count)
            std::ofstream(trace_dir_ / (trace_name + "-indices.bin"), std::ios::binary)
                .write(reinterpret_cast<const char*>(a.index_src),
                       std::streamsize(a.index_count) * (a.index16 ? 2 : 4));
    }
    if (!trace_dir_.empty() && draw_rt_) {
        LOGR("vulkan: frame-trace consts draw=%u fogmode=%.1f fog=(%.2f,%.2f,%.2f) fogcol=(%.2f,%.2f,%.2f,%.2f) pc0=(%.2f,%.2f,%.2f,%.2f) pc1=(%.2f,%.2f,%.2f,%.2f)",
             trace_draw_, c.fog[3], c.fog[0], c.fog[1], c.fog[2],
             c.fog_color[0], c.fog_color[1], c.fog_color[2], c.fog_color[3],
             c.pc[0][0], c.pc[0][1], c.pc[0][2], c.pc[0][3],
             c.pc[1][0], c.pc[1][1], c.pc[1][2], c.pc[1][3]);
    }
    if (sh.fixed_transform) {
        FillTransformConstants(s, size, c);
        FillLightConstants(s, sh.fk, c);
    }
    Alloc vc = Constants(s.vsc.data(), sizeof(s.vsc));
    Alloc dc = Constants(&c, sizeof(c));
    VkBuffer vbufs[16];
    VkDeviceSize voffs[16];
    for (uint32_t i = 0; i < a.buffer_count; i++) {
        const VertexBuffer& b = a.buffers[i];
        Alloc v = Upload(std::max<uint32_t>(b.bytes, 4), 16);
        if (b.src && b.bytes) std::memcpy(v.cpu, b.src, b.bytes);
        vbufs[i] = v.buf;
        voffs[i] = v.off;
    }
    Alloc ib{};
    if (a.index_count) {
        uint32_t ibytes = a.index_count * (a.index16 ? 2 : 4);
        ib = Upload(ibytes, 4);
        if (a.index_src && ibytes) std::memcpy(ib.cpu, a.index_src, ibytes);
    }
    bool clamp = AnyStageWraps(s) && UnitTexcoords(a);
    VkSampler samplers[4];
    for (int i = 0; i < 4; i++) samplers[i] = StageSampler(s, i, stages, clamp);
    VkDescriptorSet set = Set(stages.view, samplers);

    // the rendering
    BeginRender(draw_rt_ ? &draw_rt_->img : nullptr, draw_ds_ ? &draw_ds_->img : nullptr);
    if (draw_rt_) {
        draw_rt_->drawn = true;
        ++draw_rt_->content_serial;
        ++draw_rt_->perf2_draws;
    } else if (draw_ds_) {
        ++draw_ds_->perf2_draws;
    }
    if (draw_ds_) draw_ds_->drawn = true;
    if (pipe != cur_pipe_) {
        vkCmdBindPipeline(Cb(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        cur_pipe_ = pipe;
    }
    if (set != cur_set_ || vc.off != cur_vc_off_ || dc.off != cur_dc_off_) {
        BindSet(set, vc.off, dc.off);
        cur_set_ = set;
        cur_vc_off_ = uint32_t(vc.off);
        cur_dc_off_ = uint32_t(dc.off);
    }
    Target& bt = draw_rt_ ? *draw_rt_ : *draw_ds_;
    float W = float(bt.w), H = float(bt.h);
    SetViewport(0, 0, W, H);
    float sc = W / size.w;
    int32_t x0 = int32_t(s.viewport.x * sc), y0 = int32_t(s.viewport.y * sc);
    int32_t x1 = int32_t((s.viewport.x + s.viewport.w) * sc), y1 = int32_t((s.viewport.y + s.viewport.h) * sc);
    if (s.scissor_count) {
        x0 = std::max(x0, int32_t(s.scissors[0] * sc));
        y0 = std::max(y0, int32_t(s.scissors[1] * sc));
        x1 = std::min(x1, int32_t(s.scissors[2] * sc));
        y1 = std::min(y1, int32_t(s.scissors[3] * sc));
    }
    SetScissor(std::clamp(x0, 0, int32_t(W)), std::clamp(y0, 0, int32_t(H)), std::clamp(x1, 0, int32_t(W)),
               std::clamp(y1, 0, int32_t(H)));
    float bf[4];
    Color(SimpleState(s, kMBlendColor), bf);
    if (std::memcmp(bf, cur_blend_constants_, sizeof(bf)) != 0) {
        vkCmdSetBlendConstants(Cb(), bf);
        std::memcpy(cur_blend_constants_, bf, sizeof(bf));
    }
    uint32_t sref = SimpleState(s, kMStencilRef) & 0xFF;
    if (sref != cur_stencil_ref_) {
        vkCmdSetStencilReference(Cb(), VK_STENCIL_FACE_FRONT_AND_BACK, sref);
        cur_stencil_ref_ = sref;
    }
    if (a.buffer_count) {
        bool vb_diff = (a.buffer_count != cur_vb_count_);
        if (!vb_diff) {
            for (uint32_t i = 0; i < a.buffer_count; ++i) {
                if (cur_vbufs_[i] != vbufs[i] || cur_voffs_[i] != voffs[i]) {
                    vb_diff = true;
                    break;
                }
            }
        }
        if (vb_diff) {
            vkCmdBindVertexBuffers(Cb(), 0, a.buffer_count, vbufs, voffs);
            cur_vb_count_ = a.buffer_count;
            std::memcpy(cur_vbufs_, vbufs, a.buffer_count * sizeof(VkBuffer));
            std::memcpy(cur_voffs_, voffs, a.buffer_count * sizeof(VkDeviceSize));
        }
    } else {
        cur_vb_count_ = 0;
    }
    if (a.index_count) {
        VkIndexType itype = a.index16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
        if (ib.buf != cur_ib_ || ib.off != cur_ib_off_ || itype != cur_ib_type_) {
            vkCmdBindIndexBuffer(Cb(), ib.buf, ib.off, itype);
            cur_ib_ = ib.buf;
            cur_ib_off_ = ib.off;
            cur_ib_type_ = itype;
        }
        ++perf2_draws_;
        vkCmdDrawIndexed(Cb(), a.index_count, 1, 0, a.base_vertex, 0);
    } else {
        cur_ib_ = VK_NULL_HANDLE;
        cur_ib_off_ = 0xFFFFFFFF;
        ++perf2_draws_;
        vkCmdDraw(Cb(), a.draw_count, 1, 0, 0);
    }
    if (capture_mesh) {
        LOGR("vulkan: indexed-fog enabled=%u table=%u color=%08X translated-mode=%.0f",
             s.rs[kRsFogEnable], s.rs[kRsFogTableMode], s.rs[kRsFogColor], c.fog[3]);
        std::vector<uint8_t> after;
        char after_name[80] = "forza-indexed-after.bmp";
        if (capture_mesh == 3)
            std::snprintf(after_name, sizeof(after_name), "forza-material-%02u-after.bmp", capture_id);
        const bool captured_after = CaptureIndexedColor(*draw_rt_, after_name, after);
        uint32_t changed_rgb = 0, changed_alpha = 0;
        if (captured_before && captured_after && mesh_before.size() == after.size()) {
            for (size_t i = 0; i + 3 < after.size(); i += 4) {
                if (std::memcmp(&mesh_before[i], &after[i], 3)) ++changed_rgb;
                if (mesh_before[i + 3] != after[i + 3]) ++changed_alpha;
            }
        }
        LOGR("vulkan: indexed-color frame=%llu mesh=%u capture=%u rt=%08X indices=%u mask=%X vp=%u,%u,%u,%u vs=%016llX ps=%016llX mode=%u forced=%u before=%u after=%u changed-rgb=%u changed-alpha=%u",
             static_cast<unsigned long long>(presents_), mesh, capture_id, draw_rt_->data, a.index_count, k.wmask,
             s.viewport.x, s.viewport.y, s.viewport.w, s.viewport.h,
             static_cast<unsigned long long>(sh.vkey), static_cast<unsigned long long>(sh.pkey),
             capture_mesh, forced_color, captured_before, captured_after, changed_rgb, changed_alpha);
    }
    if (trace_sample) TraceImage(draw_rt_->img, trace_name + "-after");
    if (trace_strip_ && !trace_dir_.empty()) TraceTopStrip(draw_rt_->img, trace_relative_, trace_strip_rt_);
    if (trace_sky_window_)
        TraceSkyBand(draw_rt_->img, "rel-" + std::to_string(trace_relative_), trace_relative_);

}

VkRect2D ScaledRect(const Target& t, int x0, int y0, int x1, int y1) {
    float sc = float(t.w) / t.gw;
    int32_t l = std::clamp(int32_t(x0 * sc), 0, int32_t(t.w)), tp = std::clamp(int32_t(y0 * sc), 0, int32_t(t.h));
    int32_t r = std::clamp(int32_t(x1 * sc), 0, int32_t(t.w)), b = std::clamp(int32_t(y1 * sc), 0, int32_t(t.h));
    return {{l, tp}, {uint32_t(std::max(r - l, 0)), uint32_t(std::max(b - tp, 0))}};
}

void VulkanBackend::ClearTarget(Target& t, const State& s, uint32_t count, const uint8_t* rects, uint32_t flags,
                                uint32_t color, float z, uint32_t stencil) {
    std::vector<VkClearRect> rs;
    auto add = [&](const VkRect2D& r) {
        if (r.extent.width && r.extent.height) rs.push_back({r, 0, 1});
    };
    if (count && rects) {
        for (uint32_t i = 0; i < count; i++) {
            int32_t r[4];
            std::memcpy(r, rects + size_t(i) * sizeof(r), sizeof(r));
            add(ScaledRect(t, r[0], r[1], r[2], r[3]));
        }
    } else {
        const auto& v = s.viewport;
        add(ScaledRect(t, int(v.x), int(v.y), int(v.x + v.w), int(v.y + v.h)));
    }
    t.drawn = true;
    ++t.content_serial;
    if (rs.empty()) return;
    VkClearAttachment ca{};
    if (t.depth) {
        ca.aspectMask = ((flags & kClearZ) ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) |
                        ((flags & kClearStencil) && t.img.stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
        if (!ca.aspectMask) return;
        ca.clearValue.depthStencil = {std::clamp(z, 0.0f, 1.0f), stencil & 0xFF};
        BeginRender(nullptr, &t.img);
    } else {
        // pixels the CPU wrote before this clear go in first, a later upload would paint over the clear
        if (LatchCpuWrite(t)) UploadTarget(t, s.rt);
        ca.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        Color(color, ca.clearValue.color.float32);
        BeginRender(&t.img, nullptr);
    }
    vkCmdClearAttachments(Cb(), 1, &ca, uint32_t(rs.size()), rs.data());
}

void VulkanBackend::Clear(const State& s, uint32_t count, const uint8_t* rects, uint32_t flags, uint32_t color,
                          float z, uint32_t stencil) {
    static std::atomic<uint32_t> diag_clears{0};
    uint32_t diag_n = diag_clears.fetch_add(1, std::memory_order_relaxed);
    if (!trace_dir_.empty())
        LOGR("vulkan: frame-trace clear after=%u rt=%08X descriptor=%08X/%08X flags=%08X color=%08X rects=%u",
             trace_draw_, s.rt.data, s.rt.size, s.rt.format, flags, color, count);
    if (diag_n < 8) LOGR("vulkan: clear[%u] flags=%08X color=%08X rt=%08X size=%u", diag_n, flags, color,
                        s.rt.data, s.rt.size);
    Target* rt = (flags & kClearTarget) ? EnsureTarget(s.rt, false) : nullptr;
    Target* ds = (flags & (kClearZ | kClearStencil)) ? EnsureTarget(s.zs, true) : nullptr;
    // diagnostic: the scene's own clear colour makes painted-black and never-painted look the same
    if (g_diag_noclear || g_diag_skycheck) {
        LOG_FIRST(10, "vulkan: diag clear-seen rt=%08X gw=%u gh=%u image=%ux%u flags=%08X noclear=%u skycheck=%u",
                  s.rt.data, rt ? rt->gw : 0, rt ? rt->gh : 0, rt ? rt->img.w : 0, rt ? rt->img.h : 0, flags,
                  g_diag_noclear ? 1u : 0u, g_diag_skycheck ? 1u : 0u);
    }
    if (rt && g_diag_skycheck && (flags & kClearTarget) && rt->img.w >= 640) {
        color = 0xFFFF00FFu;
        LOG_FIRST(4, "vulkan: diag sky-check clear rt=%08X image=%ux%u", s.rt.data, rt->img.w, rt->img.h);
    }
    // diagnostic: keep the scene's own clear, so whatever was drawn before it survives
    if (rt && g_diag_noclear && rt->img.w >= 640) {
        flags &= ~kClearTarget;
        LOG_FIRST(4, "vulkan: diag no-clear rt=%08X flags=%08X", s.rt.data, flags);
    }
    std::vector<VkClearRect> rs;
    auto add = [&](const VkRect2D& r) {
        if (!r.extent.width || !r.extent.height) return;
        rs.push_back({r, 0, 1});
    };
    Target* primary = rt ? rt : ds;
    if (primary) {
        if (count && rects) {
            for (uint32_t i = 0; i < count; i++) {
                int32_t r[4];
                std::memcpy(r, rects + size_t(i) * sizeof(r), sizeof(r));
                add(ScaledRect(*primary, r[0], r[1], r[2], r[3]));
            }
        } else {
            const auto& v = s.viewport;
            add(ScaledRect(*primary, int(v.x), int(v.y), int(v.x + v.w), int(v.y + v.h)));
        }
        if (rt) {
            rt->drawn = true;
            ++rt->content_serial;
            if (LatchCpuWrite(*rt)) UploadTarget(*rt, s.rt);
        }
        if (ds) {
            ds->drawn = true;
            ++ds->content_serial;
        }
        if (!rs.empty()) {
            // a clear of the whole attachment is the pass's load op: the tile starts at the clear value instead
            // of being loaded from DRAM, and no clear command follows the pass's begin
            const bool full = rs.size() == 1 && rs[0].rect.offset.x == 0 && rs[0].rect.offset.y == 0 &&
                              rs[0].rect.extent.width == primary->img.w &&
                              rs[0].rect.extent.height == primary->img.h &&
                              (!ds || (ds->img.w == primary->img.w && ds->img.h == primary->img.h));
            const bool clear_color = rt && (flags & kClearTarget);
            // a depth load op clears depth and stencil together, so both have to be asked for
            const bool clear_depth = ds && (flags & kClearZ) && (!ds->img.stencil || (flags & kClearStencil));
            const bool covered = (!rt || !(flags & kClearTarget) || clear_color) &&
                                 (!ds || !(flags & (kClearZ | kClearStencil)) || clear_depth);
            const bool open_here =
                !rendering_ || ((!rt || &rt->img != rend_color_) && (!ds || &ds->img != rend_depth_));
            if (full && covered && (clear_color || clear_depth) && open_here) {
                clear_on_begin_ = (clear_color ? 1u : 0u) | (clear_depth ? 2u : 0u);
                if (clear_color) Color(color, clear_color_.float32);
                if (clear_depth) clear_ds_ = {std::clamp(z, 0.0f, 1.0f), stencil & 0xFF};
                BeginRender(rt ? &rt->img : nullptr, ds ? &ds->img : nullptr);
                return;
            }
            VkClearAttachment cas[2]{};
            uint32_t nca = 0;
            if (rt && (flags & kClearTarget)) {
                cas[nca].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                Color(color, cas[nca].clearValue.color.float32);
                nca++;
            }
            if (ds && ((flags & (kClearZ | kClearStencil)))) {
                cas[nca].aspectMask = ((flags & kClearZ) ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) |
                                      ((flags & kClearStencil) && ds->img.stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
                if (cas[nca].aspectMask) {
                    cas[nca].clearValue.depthStencil = {std::clamp(z, 0.0f, 1.0f), stencil & 0xFF};
                    nca++;
                }
            }
            if (nca) {
                BeginRender(rt ? &rt->img : nullptr, ds ? &ds->img : nullptr);
                vkCmdClearAttachments(Cb(), nca, cas, uint32_t(rs.size()), rs.data());
            }
        }
    }
}

void VulkanBackend::Copy(const Surface& src, const Surface& dst, uint32_t count, uint32_t src_rects,
                         uint32_t dst_points) {
    if (!trace_dir_.empty())
        LOGR("vulkan: frame-trace copy after=%u source=%08X/%08X/%08X target=%08X/%08X/%08X rects=%u",
             trace_draw_, src.data, src.size, src.format, dst.data, dst.size, dst.format, count);
    Target* src_t = FindTarget(src, false);
    // pixels the CPU made: the copy happens in memory, textures and targets pick it up from there
    if (!src_t || !src_t->drawn) {
        if (!CopySurfaceMemory(src, dst, count, src_rects, dst_points)) {
            LOG_FIRST(8, "vulkan: CopyRects %08X to %08X skipped", src.data, dst.data);
        } else if (Target* dst_t = FindTarget(dst, false)) {
            UploadTarget(*dst_t, dst);
        }
        return;
    }
    Target* dst_t = EnsureTarget(dst, false);
    src_t = FindTarget(src, false);
    if (!dst_t || !src_t || src_t == dst_t) return;
    Transition(src_t->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    Transition(dst_t->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    float dst_scale = float(dst_t->w) / dst_t->gw;
    bool listed = count && src_rects;
    uint32_t n = listed ? count : 1;
    std::vector<VkImageCopy> regions;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t l = 0, t = 0, r = std::min(src_t->w, dst_t->w), b = std::min(src_t->h, dst_t->h), dx = 0, dy = 0;
        if (listed) {
            GuestRect g = ReadGuestRect(src_rects, i);
            l = ScaleUnsigned(g.left, dst_scale);
            t = ScaleUnsigned(g.top, dst_scale);
            r = std::min(ScaleUnsigned(g.right, dst_scale), src_t->w);
            b = std::min(ScaleUnsigned(g.bottom, dst_scale), src_t->h);
            dx = l;
            dy = t;
            if (dst_points) {
                int32_t px, py;
                ReadGuestPoint(dst_points, i, px, py);
                dx = ScaleUnsigned(px, dst_scale);
                dy = ScaleUnsigned(py, dst_scale);
            }
        }
        if (r <= l || b <= t || dx >= dst_t->w || dy >= dst_t->h) continue;
        r = std::min(r, l + dst_t->w - dx);
        b = std::min(b, t + dst_t->h - dy);
        regions.push_back({{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                           {int32_t(l), int32_t(t), 0},
                           {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                           {int32_t(dx), int32_t(dy), 0},
                           {r - l, b - t, 1}});
    }
    if (!regions.empty())
        vkCmdCopyImage(Cb(), src_t->img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst_t->img.image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, uint32_t(regions.size()), regions.data());
    dst_t->drawn = true;
    ++dst_t->content_serial;
}

// ---- read-backs

static bool SaveTraceBmp(const std::filesystem::path& path, uint32_t w, uint32_t h,
                         const std::vector<uint8_t>& pixels) {
    std::ofstream file(path, std::ios::binary);
    uint8_t header[54] = {'B', 'M'};
    auto put = [&](size_t offset, uint32_t value) { std::memcpy(header + offset, &value, sizeof(value)); };
    put(2, uint32_t(sizeof(header) + pixels.size()));
    put(10, sizeof(header));
    put(14, 40);
    put(18, w);
    put(22, uint32_t(-int32_t(h)));
    header[26] = 1;
    header[28] = 32;
    put(34, uint32_t(pixels.size()));
    file.write(reinterpret_cast<const char*>(header), sizeof(header));
    file.write(reinterpret_cast<const char*>(pixels.data()), std::streamsize(pixels.size()));
    return file.good();
}

bool VulkanBackend::CaptureIndexedColor(Target& target, const char* filename, std::vector<uint8_t>& pixels) {
    EndRender();
    Transition(target.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    if (!ReadBack(target.img, target.w, target.h, pixels)) return false;
    const auto path = app::ExeDir() / filename;
    return SaveTraceBmp(path, target.w, target.h, pixels);
}

void VulkanBackend::TraceImage(Image& image, const std::string& name) {
    EndRender();
    Transition(image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    std::vector<uint8_t> pixels;
    if (!ReadBack(image, image.w, image.h, pixels)) return;
    uint64_t rgb = 0, alpha = 0;
    uint32_t nonblack = 0, samples = 0;
    for (uint32_t y = image.h / 4; y < image.h * 3 / 4; ++y) {
        for (uint32_t x = image.w / 4; x < image.w * 3 / 4; ++x) {
            const uint8_t* pixel = pixels.data() + (size_t(y) * image.w + x) * 4;
            rgb += pixel[0] + pixel[1] + pixel[2];
            alpha += pixel[3];
            nonblack += std::max({pixel[0], pixel[1], pixel[2]}) > 3;
            ++samples;
        }
    }
    const bool saved = SaveTraceBmp(trace_dir_ / (name + ".bmp"), image.w, image.h, pixels);
    LOGR("vulkan: frame-trace image=%s size=%ux%u center=%u/%u rgb=%.3f alpha=%.3f saved=%u",
         name.c_str(), image.w, image.h, nonblack, samples, samples ? double(rgb) / (3.0 * samples) : 0.0,
         samples ? double(alpha) / samples : 0.0, saved);
}

// The top of the frame is where a sky would land, and a full read-back is the
// only way to tell whether the scene ever paints it: the draw log says nothing
// about coverage. Samples are numbered relative to entering the main target, so
// reflections and other cars cannot shift what the numbers mean.
void VulkanBackend::TraceSkyBand(Image& image, const std::string& name, uint32_t relative) {
    EndRender();
    Transition(image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    std::vector<uint8_t> pixels;
    if (!ReadBack(image, image.w, image.h, pixels)) return;
    const uint32_t rows = std::max(image.h / 3, 1u);
    uint64_t band = 0, frame = 0;
    uint32_t band_on = 0, band_n = 0, frame_on = 0, frame_n = 0;
    for (uint32_t y = 0; y < image.h; ++y) {
        for (uint32_t x = 0; x < image.w; ++x) {
            const uint8_t* p = pixels.data() + (size_t(y) * image.w + x) * 4;
            const uint32_t on = std::max({p[0], p[1], p[2]}) > 3;
            frame += p[0] + p[1] + p[2];
            frame_on += on;
            ++frame_n;
            if (y < rows) {
                band += p[0] + p[1] + p[2];
                band_on += on;
                ++band_n;
            }
        }
    }
    const double band_mean = band_n ? double(band) / (3.0 * band_n) : 0.0;
    const double frame_mean = frame_n ? double(frame) / (3.0 * frame_n) : 0.0;
    LOGR("vulkan: frame-sky relative=%u band=%.3f lit=%u/%u frame=%.3f on=%u/%u",
         relative, band_mean, band_on, band_n, frame_mean, frame_on, frame_n);
    // only the draws that change what the sky looks like are worth a bitmap
    const bool changed = std::abs(band_mean - trace_sky_mean_) > 0.5 ||
                         (band_on > trace_sky_on_ ? band_on - trace_sky_on_ : trace_sky_on_ - band_on) > band_n / 50;
    trace_sky_mean_ = band_mean;
    trace_sky_on_ = band_on;
    if (changed && trace_sky_saved_ < 48) {
        ++trace_sky_saved_;
        LOGR("vulkan: frame-sky-save relative=%u name=%s saved=%u", relative, name.c_str(),
             SaveTraceBmp(trace_dir_ / (name + ".bmp"), image.w, image.h, pixels));
    }
}

// whose draw touches the top of the scene, and does any of them touch it at all
void VulkanBackend::TraceTopStrip(Image& image, uint32_t relative, uint32_t rt) {
    EndRender();
    Transition(image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    std::vector<uint8_t> pixels;
    if (!ReadBack(image, image.w, image.h, pixels)) return;
    const uint32_t rows = std::clamp(image.h / 12, 1u, image.h);
    uint64_t sum = 0, painted = 0;
    uint32_t magenta = 0;
    for (uint32_t y = 0; y < rows; ++y) {
        for (uint32_t x = 0; x < image.w; ++x) {
            const uint8_t* p = pixels.data() + (size_t(y) * image.w + x) * 4;
            sum += p[0] + p[1] + p[2];
            // magenta is the diagnostic clear, so anything else was drawn over it
            if (p[0] > 240 && p[1] < 16 && p[2] > 240) ++magenta;
        }
    }
    const uint64_t total = uint64_t(rows) * image.w;
    // a draw that covers nothing anywhere is the interesting case: geometry submitted but never rasterized
    uint32_t changed = 0, top_changed = 0;
    const bool compare = trace_prev_rt_ == rt && trace_prev_pixels_.size() == pixels.size();
    if (compare) {
        for (size_t i = 0; i + 3 < pixels.size(); i += 4) {
            if (pixels[i] != trace_prev_pixels_[i] || pixels[i + 1] != trace_prev_pixels_[i + 1] ||
                pixels[i + 2] != trace_prev_pixels_[i + 2]) {
                ++changed;
                if (i / 4 < total) ++top_changed;
            }
        }
    }
    if (image.w >= 640) {
        trace_prev_pixels_ = pixels;
        trace_prev_rt_ = rt;
    }
    LOGR("vulkan: sky-strip rt=%08X rel=%u size=%ux%u mean=%.2f magenta=%u/%u changed=%u topchg=%u ct=%u",
         rt, relative, image.w, image.h, total ? double(sum) / (3.0 * double(total)) : 0.0, magenta,
         unsigned(total), changed, top_changed, unsigned(compare));
}

bool VulkanBackend::TraceDepth(Target& target, const std::string& name) {
    if (!target.depth || !target.img.depth || !target.img.image) return false;
    EndRender();
    Transition(target.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    Buffer rb;
    VkDeviceSize bytes = VkDeviceSize(target.w) * target.h * 4;
    if (!CreateBuffer(rb, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true)) return false;
    VkBufferImageCopy copy{0, 0, 0, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, {0, 0, 0},
                           {target.w, target.h, 1}};
    vkCmdCopyImageToBuffer(Cb(), target.img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buf, 1, &copy);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(Cb(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                         1, &barrier, 0, nullptr, 0, nullptr);
    Submit(true, false);

    const auto* raw_bytes = static_cast<const uint8_t*>(rb.mem.map);
    std::vector<uint8_t> low24(size_t(target.w) * target.h * 4);
    std::vector<uint8_t> high24(size_t(target.w) * target.h * 4);
    uint32_t low_min = 0xFFFFFF, low_max = 0, high_min = 0xFFFFFF, high_max = 0;
    uint64_t low_sum = 0;
    for (size_t i = 0; i < size_t(target.w) * target.h; ++i) {
        uint32_t word;
        std::memcpy(&word, raw_bytes + i * 4, sizeof(word));
        uint32_t lo = word & 0xFFFFFF, hi = word >> 8;
        low_min = std::min(low_min, lo); low_max = std::max(low_max, lo); low_sum += lo;
        high_min = std::min(high_min, hi); high_max = std::max(high_max, hi);
        for (auto [pixels, value] : {std::pair{&low24, lo}, std::pair{&high24, hi}}) {
            uint8_t gray = uint8_t(value >> 16);
            (*pixels)[i * 4 + 0] = gray;
            (*pixels)[i * 4 + 1] = gray;
            (*pixels)[i * 4 + 2] = gray;
            (*pixels)[i * 4 + 3] = 255;
        }
    }
    std::ofstream raw(trace_dir_ / (name + ".raw"), std::ios::binary);
    raw.write(reinterpret_cast<const char*>(raw_bytes), size_t(bytes));
    bool saved = SaveTraceBmp(trace_dir_ / (name + "-low24.bmp"), target.w, target.h, low24);
    saved = SaveTraceBmp(trace_dir_ / (name + "-high24.bmp"), target.w, target.h, high24) && saved;
    LOGR("vulkan: frame-trace depth=%s size=%ux%u format=%u low24=%06X..%06X mean=%llu high24=%06X..%06X saved=%u",
         name.c_str(), target.w, target.h, uint32_t(target.img.fmt), low_min, low_max,
         static_cast<unsigned long long>(low_sum / (uint64_t(target.w) * target.h)), high_min, high_max, saved);
    Destroy(rb);
    return saved;
}

void VulkanBackend::Sync(uint32_t data, uint32_t bytes) {
    Target* selected = nullptr;
    for (auto& [key, target] : targets_) {
        if (target.data != data || target.depth || !target.drawn || (bytes && target.bytes != bytes)) continue;
        if (!selected || target.use_serial > selected->use_serial) selected = &target;
    }
    if (!selected && bytes) {
        for (auto& [key, target] : targets_) {
            if (target.data != data || target.depth || !target.drawn) continue;
            if (!selected || target.use_serial > selected->use_serial) selected = &target;
        }
    }
    if (!selected) return;
    Target& t = *selected;
    Image down;
    Image* src = CopySource(t, down);
    std::vector<uint8_t> px;
    bool ok = src && ReadBack(*src, t.gw, t.gh, px);
    Destroy(down);  // ReadBack waited for the GPU
    if (!ok) return;
    uint32_t pitch = GuestPitch(t.gw);
    uint8_t* guest = static_cast<uint8_t*>(kernel::HostPtr(data));
    for (uint32_t y = 0; y < t.gh; y++) std::memcpy(guest + size_t(y) * pitch, &px[size_t(y) * t.gw * 4], t.gw * 4);
    t.read = std::move(px);
    t.drawn = false;
    t.sig = MemSig(t.data, t.bytes);
}

void VulkanBackend::SyncLate(uint32_t data) {
    if (!DeliverLate(data)) Sync(data, 0);
    RecordLate(data);
}

bool VulkanBackend::DeliverLate(uint32_t data) {
    std::lock_guard lock(late_lock_);
    auto it = late_.find(data);
    if (it == late_.end()) return false;
    Late& l = it->second;
    uint64_t done = Completed(), now = presents_seen_.load();
    int best = -1;
    for (int i = 0; i < kFrames; i++) {
        bool finished = l.fence[i] && l.fence[i] <= done;
        bool recent = now - l.frame[i] <= uint64_t(kFrames) + 1;
        if (finished && recent && (best < 0 || l.fence[i] > l.fence[best])) best = i;
    }
    if (best < 0) return false;
    if (l.fence[best] > l.delivered) {
        const uint8_t* p = l.rb[best].mem.map;
        uint32_t pitch = GuestPitch(l.gw);
        uint8_t* dst = static_cast<uint8_t*>(kernel::HostPtr(data));
        for (uint32_t y = 0; y < l.gh; y++) std::memcpy(dst + size_t(y) * pitch, p + size_t(y) * l.gw * 4, l.gw * 4);
        l.delivered = l.fence[best];
    }
    return true;
}

void VulkanBackend::RecordLate(uint32_t data) {
    Target* selected = FindTargetByData(data, false);
    if (!selected) return;
    Target& t = *selected;
    t.sig = MemSig(t.data, t.bytes);
    t.read.clear();
    std::lock_guard lock(late_lock_);
    Late& l = late_[data];
    if (l.gw != t.gw || l.gh != t.gh) {
        for (Buffer& b : l.rb) Retire(b);
        Retire(l.down);
        l = Late{};
        l.gw = t.gw;
        l.gh = t.gh;
    }
    uint32_t slot = l.next;
    if (l.fence[slot] > Completed()) return;
    Image* src = CopySource(t, l.down);
    if (!src) return;
    if (!l.rb[slot].buf && !CreateBuffer(l.rb[slot], VkDeviceSize(t.gw) * t.gh * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true))
        return;
    EndRender();
    VkBufferImageCopy c{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {t.gw, t.gh, 1}};
    vkCmdCopyImageToBuffer(Cb(), src->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, l.rb[slot].buf, 1, &c);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(Cb(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    l.fence[slot] = value_ + 1;  // the next Submit carries it
    l.frame[slot] = presents_;
    l.next = (slot + 1) % kFrames;
}

// ---- present

// the swap chain made again when Android gave a new window, the size or vsync changed; the surface too when the
// window is new, lost, or could not be made last time. false: nothing to show on, the frame is dropped
bool VulkanBackend::AcquireImage(uint32_t& index) {
    if (!host::Foreground()) return false;
    if (!surface_ || surface_lost_ || host::SurfaceEpoch() != surface_epoch_) {
        if (surface_) LOGI("vulkan: new surface (%s)", surface_lost_ ? "lost" : "new window");
        DestroySwapchain();
        if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
        if (!CreateSurface()) return false;
        swap_stale_ = true;
    }
    int w, h;
    host::WindowPixels(w, h);
    if (!swap_ || swap_stale_ || uint32_t(w) != swap_extent_.width || uint32_t(h) != swap_extent_.height ||
        app::Get().vsync != vsync_) {
        if (!CreateSwapchain()) return false;
    }
    // a bounded wait: a window Android is taking away may never hand an image back
    constexpr uint64_t kAcquireTimeout = 500'000'000;
    for (int tries = 0; tries < 2; tries++) {
        VkResult r = vkAcquireNextImageKHR(dev_, swap_, kAcquireTimeout, CurrentFrame().acquire, VK_NULL_HANDLE, &index);
        // suboptimal is left alone: Android says so every frame on a turned screen, where only the compositor's
        // rotation differs, and making the chain again each frame freed images the GPU was still drawing (Mali)
        if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) {
            CurrentFrame().acquire_pending = true;
            return true;
        }
        if (r == VK_TIMEOUT || r == VK_NOT_READY) return false;
        if (r == VK_ERROR_SURFACE_LOST_KHR) {
            surface_lost_ = true;
            return false;
        }
        if (r != VK_ERROR_OUT_OF_DATE_KHR || !CreateSwapchain()) return false;
    }
    return false;
}

bool VulkanBackend::UpdateVideoOverlay() {
    const uint32_t mmio = mem::kNv2aMmioBase;
    const uint32_t buffer = kernel::R32(mmio + 0x8700);
    const uint32_t format = kernel::R32(mmio + 0x8958);
    if (!(buffer & 1u) || ((format >> 16) & 3u) != 1u) return false;

    const uint32_t base = kernel::R32(mmio + 0x8900);
    const uint32_t offset = kernel::R32(mmio + 0x8920);
    const uint32_t src = base + offset;
    const uint32_t in_size = kernel::R32(mmio + 0x8928);
    const uint32_t in_w = in_size & 0x7FFu, in_h = (in_size >> 16) & 0x7FFu;
    const uint32_t in_point = kernel::R32(mmio + 0x8930);
    const uint32_t in_x = in_point & 0x7FFFu, in_y = (in_point >> 16) & 0x7FFFu;
    const uint32_t step_x = kernel::R32(mmio + 0x8938) ? kernel::R32(mmio + 0x8938) : 0x00100000u;
    const uint32_t step_y = kernel::R32(mmio + 0x8940) ? kernel::R32(mmio + 0x8940) : 0x00100000u;
    const uint32_t out_point = kernel::R32(mmio + 0x8948);
    const uint32_t out_size = kernel::R32(mmio + 0x8950);
    const uint32_t out_w = out_size & 0xFFFu, out_h = (out_size >> 16) & 0xFFFu;
    const uint32_t pitch = format & 0x1FFFu;

    // PVIDEO addresses UMA with physical Xbox addresses; its normal RAM alias is
    // at 0x80000000 | physical. The frame is YUY2, two pixels per four bytes.
    if (!src || !in_w || !in_h || !out_w || !out_h || in_w > 2048 || in_h > 2048 || out_w > 2048 || out_h > 2048 ||
        pitch < in_w * 2u || src >= mem::kPhysicalSize ||
        uint64_t(src) + uint64_t(pitch) * in_h > mem::kPhysicalSize ||
        uint64_t(out_w) * out_h * 4 > (32u << 20))
        return false;

    const uint8_t* yuy2 = static_cast<const uint8_t*>(kernel::HostPtr(mem::VirtualFromPhysical(src)));
    std::vector<uint8_t> rgba(size_t(out_w) * out_h * 4);
    auto clip = [](int v) { return uint8_t(std::clamp(v, 0, 255)); };
    for (uint32_t y = 0; y < out_h; ++y) {
        const uint32_t sy = std::min(in_h - 1, in_y + uint32_t((uint64_t(y) * step_y) >> 20));
        const uint8_t* row = yuy2 + size_t(sy) * pitch;
        for (uint32_t x = 0; x < out_w; ++x) {
            const uint32_t sx = std::min(in_w - 1, in_x + uint32_t((uint64_t(x) * step_x) >> 20));
            const uint8_t* p = row + (sx & ~1u) * 2u;
            const int yy = p[(sx & 1u) ? 2 : 0] - 16;
            const int u = int(p[1]) - 128, v = int(p[3]) - 128;
            const uint8_t r = clip((298 * yy + 409 * v + 128) >> 8);
            const uint8_t g = clip((298 * yy - 100 * u - 208 * v + 128) >> 8);
            const uint8_t b = clip((298 * yy + 516 * u + 128) >> 8);
            uint8_t* d = rgba.data() + (size_t(y) * out_w + x) * 4;
            d[0] = b; d[1] = g; d[2] = r; d[3] = 255;
        }
    }

    if (!video_overlay_ || video_overlay_w_ != out_w || video_overlay_h_ != out_h) {
        Retire(video_overlay_);
        if (!CreateImage(video_overlay_, kBgra, out_w, out_h, 1, 1, 1,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_VIEW_TYPE_2D)) {
            video_overlay_w_ = video_overlay_h_ = 0;
            return false;
        }
        video_overlay_w_ = out_w;
        video_overlay_h_ = out_h;
    }
    Alloc upload = Upload(rgba.size(), 16);
    if (!upload.cpu) return false;
    std::memcpy(upload.cpu, rgba.data(), rgba.size());
    Transition(video_overlay_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy copy{upload.off, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {out_w, out_h, 1}};
    vkCmdCopyBufferToImage(Cb(), upload.buf, video_overlay_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    Transition(video_overlay_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    static std::atomic<uint32_t> overlay_logs{0};
    const uint32_t n = overlay_logs.fetch_add(1, std::memory_order_relaxed);
    if (n < 8) LOGR("vulkan: bink-overlay[%u] src=%08X in=%ux%u pitch=%u out=%ux%u at=%u,%u",
                    n, src, in_w, in_h, pitch, out_w, out_h, out_point & 0xFFFu, (out_point >> 16) & 0xFFFu);
    return true;
}

void VulkanBackend::DrawFrame(Target* t, uint32_t index) {
    EndRender();
    const bool has_video_overlay = UpdateVideoOverlay();
    static std::atomic<uint32_t> draw_frame_diag{0};
    VkCommandBuffer cb = Cb();
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = swap_images_[index];
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &b);
    // the frame letterboxed to the game's aspect over black
    float aspect = game::Aspect();
    float w = float(swap_extent_.width), h = float(swap_extent_.height);
    float vw = std::min(w, h * aspect), vh = vw / aspect;
    uint32_t filter = std::min(game::Opt().screen_filter, 8u);
    uint32_t crt = filter <= 2 ? filter : 0;
    bool fxaa = game::Opt().anti_aliasing == 1, ssaa = game::Opt().anti_aliasing == 2;
    bool post = game::Opt().hdr || fxaa;
    // opts.z: the taps a side over each screen pixel, 2 averaging SSAA's samples
    float c[24] = {0, 0, 1, 1, gamma_ ? 1.0f : 0.0f, 0, ssaa ? 2.0f : 1.0f, 1};
    c[20] = game::Opt().hdr ? 1.0f : 0.0f;
    c[21] = fxaa ? 1.0f : 0.0f;
    if (t) {
        // a texel of the frame, seconds that wrap before a float loses them, the colour look (4 vibrant on)
        using clock = std::chrono::steady_clock;
        float secs = std::fmod(std::chrono::duration<float>(clock::now().time_since_epoch()).count(), 1000.0f);
        float src[4] = {1.0f / float(t->img.w), 1.0f / float(t->img.h), secs, float(filter >= 4 ? filter - 4 : 0)};
        std::memcpy(c + 16, src, sizeof(src));
    }
    if (crt) {
        // light, then strong: the gaps' darkness, the curve, the mask's dimming, the corners' shade
        constexpr float kGap[] = {0.35f, 0.6f};
        constexpr float kCurve[] = {0.02f, 0.045f};
        constexpr float kMask[] = {0.15f, 0.3f};
        constexpr float kCorner[] = {0.1f, 0.2f};
        int k = int(crt) - 1;
        // the game's lines, as many screen pixels each as come nearest
        float per = std::max(1.0f, std::round(vh / float(guest_h_)));
        float lines = std::floor(vh / per);
        // on average a line keeps 1 - gap / 2 and the mask 1 - 2 mask / 3: that much taken back, most of it
        float kept = (1 - kGap[k] / 2) * (1 - 2 * kMask[k] / 3);
        float rest[8] = {lines, kGap[k], kCurve[k], kMask[k], kCorner[k], 1 + (1 / kept - 1) * 0.8f, 0, 0};
        std::memcpy(c + 8, rest, sizeof(rest));
    }
    Alloc cbuf = Constants(c, sizeof(c));
    const char* hlsl = filter || post ? kFilterHlsl : kBlitHlsl;
    const char* entry = crt ? "pscrt" : filter == 3 ? "pssharpen" : filter ? "pscolor" : post ? "psplain" : "psmain";
    VkPipeline pipe = (t || has_video_overlay) ? FullScreenPipeline(hlsl, entry, swap_fmt_, VK_FORMAT_UNDEFINED, false) : VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (pipe) {
        Transition(t->img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        Transition(lut_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        const VkImageView views[4] = {t->img.view, lut_.view, VK_NULL_HANDLE, VK_NULL_HANDLE};
        const VkSampler samplers[4] = {linear_, point_, point_, point_};
        set = Set(views, samplers);
    }
    uint32_t draw_diag = draw_frame_diag.fetch_add(1, std::memory_order_relaxed);
    if (draw_diag < 8 || draw_diag % 600 == 0)
        LOGR("vulkan: draw-frame[%u] target=%p image=%p view=%p layout=%u image-size=%ux%u swap=%ux%u fmt=%u overlay=%u pipe=%p set=%p entry=%s viewport=%.1fx%.1f",
             draw_diag, (void*)t, t ? (void*)t->img.image : nullptr, t ? (void*)t->img.view : nullptr,
             t ? uint32_t(t->img.layout) : 0, t ? t->img.w : 0, t ? t->img.h : 0,
             swap_extent_.width, swap_extent_.height, uint32_t(swap_fmt_), has_video_overlay,
             (void*)pipe, (void*)set, entry, vw, vh);
    VkDescriptorSet overlay_set = VK_NULL_HANDLE;
    if (pipe && has_video_overlay) {
        Transition(video_overlay_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        Transition(lut_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        const VkImageView views[4] = {video_overlay_.view, lut_.view, VK_NULL_HANDLE, VK_NULL_HANDLE};
        const VkSampler samplers[4] = {linear_, point_, point_, point_};
        overlay_set = Set(views, samplers);
    }
    VkRenderingAttachmentInfoKHR ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR};
    ca.imageView = swap_views_[index];
    ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ca.clearValue.color = {{0, 0, 0, 1}};
    VkRenderingInfoKHR ri{VK_STRUCTURE_TYPE_RENDERING_INFO_KHR};
    ri.renderArea = {{0, 0}, swap_extent_};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    if (dynamic_) {
        begin_rendering_(cb, &ri);
    } else {
        VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        bi.renderPass = Pass(swap_fmt_, VK_FORMAT_UNDEFINED, true);
        bi.framebuffer = Framebuffer(swap_fmt_, VK_FORMAT_UNDEFINED, swap_views_[index], VK_NULL_HANDLE,
                                     swap_extent_.width, swap_extent_.height);
        bi.renderArea = ri.renderArea;
        bi.clearValueCount = 1;
        bi.pClearValues = &ca.clearValue;
        vkCmdBeginRenderPass(cb, &bi, VK_SUBPASS_CONTENTS_INLINE);
    }
    if (pipe) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        SetScissor(0, 0, int32_t(w), int32_t(h));
        if (set) {
            BindSet(set, cbuf.off, cbuf.off);
            SetViewport((w - vw) / 2, (h - vh) / 2, vw, vh);
            vkCmdDraw(cb, 3, 1, 0, 0);
        }
        if (overlay_set) {
            const uint32_t ox = kernel::R32(mem::kNv2aMmioBase + 0x8948) & 0xFFFu;
            const uint32_t oy = (kernel::R32(mem::kNv2aMmioBase + 0x8948) >> 16) & 0xFFFu;
            BindSet(overlay_set, cbuf.off, cbuf.off);
            SetViewport((w - vw) / 2 + ox * vw / float(guest_w_),
                        (h - vh) / 2 + oy * vh / float(guest_h_),
                        video_overlay_w_ * vw / float(guest_w_), video_overlay_h_ * vh / float(guest_h_));
            vkCmdDraw(cb, 3, 1, 0, 0);
        }
    }
    RenderUi();
    if (dynamic_)
        end_rendering_(cb);
    else
        vkCmdEndRenderPass(cb);
    bool captured_swap = false;
    if (gpu::g_capture_present_after_input.exchange(false, std::memory_order_acq_rel) || !trace_dir_.empty()) {
        if (!swap_transfer_src_) {
            LOGR("vulkan: composed-frame capture unavailable; swapchain lacks transfer-src usage");
        } else {
            const uint32_t cw = swap_extent_.width, ch = swap_extent_.height;
            Buffer rb;
            const VkDeviceSize size = VkDeviceSize(cw) * ch * 4;
            if (CreateBuffer(rb, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true) && rb.mem.map) {
                VkImageMemoryBarrier to_copy{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                to_copy.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                to_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                to_copy.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                to_copy.srcQueueFamilyIndex = to_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                to_copy.image = swap_images_[index];
                to_copy.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(Cb(), VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_copy);
                VkBufferImageCopy copy{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {cw, ch, 1}};
                vkCmdCopyImageToBuffer(Cb(), swap_images_[index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buf, 1, &copy);
                VkBufferMemoryBarrier host_read{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                host_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                host_read.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                host_read.srcQueueFamilyIndex = host_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                host_read.buffer = rb.buf;
                host_read.offset = 0;
                host_read.size = size;
                vkCmdPipelineBarrier(Cb(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                                     0, nullptr, 1, &host_read, 0, nullptr);
                Submit(true, false);
                const auto* pixels = static_cast<const uint8_t*>(rb.mem.map);
                uint32_t non_black = 0;
                for (VkDeviceSize i = 0; i + 3 < size; i += 4)
                    if (pixels[i] || pixels[i + 1] || pixels[i + 2]) ++non_black;
                std::filesystem::path path = trace_dir_.empty()
                    ? app::ExeDir() / "forza-composed-after-input.bmp" : trace_dir_ / "composed.bmp";
                std::vector<uint8_t> bgra(pixels, pixels + size);
                if (swap_fmt_ == VK_FORMAT_R8G8B8A8_UNORM || swap_fmt_ == VK_FORMAT_R8G8B8A8_SRGB)
                    for (size_t i = 0; i < bgra.size(); i += 4) std::swap(bgra[i], bgra[i + 2]);
                std::error_code ec;
                std::filesystem::create_directories(path.parent_path(), ec);
                std::ofstream f(path, std::ios::binary);
                uint8_t header[54] = {'B','M'};
                auto put = [&](size_t o, uint32_t v) { std::memcpy(header + o, &v, sizeof(v)); };
                put(2, uint32_t(54 + size)); put(10, 54); put(14, 40);
                put(18, cw); put(22, uint32_t(-int32_t(ch)));
                header[26] = 1; header[28] = 32; put(34, uint32_t(size));
                if (f) {
                    f.write(reinterpret_cast<const char*>(header), sizeof(header));
                    f.write(reinterpret_cast<const char*>(bgra.data()), std::streamsize(bgra.size()));
                }
                LOGR("vulkan: composed-frame-capture path=%s size=%ux%u format=%u nonblack=%u saved=%u",
                     path.string().c_str(), cw, ch, uint32_t(swap_fmt_), non_black, f.good());
                Destroy(rb);
                captured_swap = true;
            } else {
                Destroy(rb);
                LOGR("vulkan: composed-frame-capture staging allocation failed size=%llu",
                     static_cast<unsigned long long>(size));
            }
        }
    }
    b.srcAccessMask = captured_swap ? VK_ACCESS_TRANSFER_READ_BIT : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    b.dstAccessMask = 0;
    b.oldLayout = captured_swap ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier(Cb(), captured_swap ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &b);
}

#if defined(__ANDROID__)
#include <jni.h>
#include <unistd.h>
extern "C" void* SDL_GetAndroidJNIEnv();
extern "C" void* SDL_GetAndroidActivity();
#endif

void VulkanBackend::UpdatePerf() {
    diag_perf_fps_ = fps_;
    diag_perf_ = game::Opt().fps_counter;
    if (!diag_perf_) return;

    static uint32_t perf_tick = 0;
    if (++perf_tick < 60) return;
    perf_tick = 0;

    for (int i = 0; i < 8; i++) {
        char path[64];
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", i);
        std::ifstream f(path);
        uint32_t v = 0;
        if (f) f >> v;
        diag_perf_cpu_[i] = v / 1000u;
    }
    for (auto p : {"/sys/class/devfreq/17000000.devfreq_gpu/cur_freq",
                   "/sys/class/devfreq/17000040.devfreq_disp/cur_freq",
                   "/sys/class/devfreq/17000010.devfreq_mif/cur_freq"}) {
        std::ifstream f(p);
        uint32_t v = 0;
        if (f) { f >> v; diag_perf_gpu_ = v / 1000u; break; }
    }
    std::ifstream b("/sys/class/thermal/thermal_zone0/temp");
    uint32_t t = 0;
    if (b) b >> t;
    if (t > 0) diag_perf_temp_ = t > 1000u ? t / 1000u : t;

    // Query battery info (temperature and level) via JNI
    static uint32_t jni_tick = 0;
    if (++jni_tick >= 60) {
        jni_tick = 0;
#if defined(__ANDROID__)
        auto* env = static_cast<JNIEnv*>(SDL_GetAndroidJNIEnv());
        auto activity = static_cast<jobject>(SDL_GetAndroidActivity());
        if (env && activity) {
            jclass cls = env->GetObjectClass(activity);
            jmethodID m = env->GetMethodID(cls, "getBatteryInfo", "()[I");
            if (m) {
                auto arr = static_cast<jintArray>(env->CallObjectMethod(activity, m));
                if (arr) {
                    jint buf[2] = {0, 0};
                    env->GetIntArrayRegion(arr, 0, 2, buf);
                    if (buf[0] > 0) diag_perf_temp_ = uint32_t(buf[0] / 10);
                    if (buf[1] > 0) diag_perf_batt_ = uint32_t(buf[1]);
                    env->DeleteLocalRef(arr);
                }
            }
            if (env->ExceptionCheck()) env->ExceptionClear();
        }
#endif
    }

    static uint32_t diag_perf_logged = 0;
    if (diag_perf_logged < 4) {
        ++diag_perf_logged;
        LOGR("vulkan: diag perf marker=%u cpu=%u gpu=%u temp=%u fps=%.1f",
             diag_perf_ ? 1u : 0u, diag_perf_cpu_[0], diag_perf_gpu_, diag_perf_temp_, diag_perf_fps_);
    }
}

void VulkanBackend::RenderUi() {
    if (!ui_) return;
    std::lock_guard lock(app::UiLock());
    UpdatePerf();
    ImGui_ImplVulkan_NewFrame();
    ui::NewFrame(float(swap_extent_.width), float(swap_extent_.height));
    ImGui::NewFrame();
    if (app::kGame.ui) app::kGame.ui(fps_);
    touch::Draw();

    // CPU utilization calculation from /proc/self/stat
    static uint64_t last_utime = 0, last_stime = 0;
    static auto last_cpu_sample = std::chrono::steady_clock::now();
    static float diag_cpu_pct = 0.0f;

    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - last_cpu_sample).count();
    if (dt >= 0.5) {
        std::ifstream stat_f("/proc/self/stat");
        if (stat_f) {
            std::string line;
            if (std::getline(stat_f, line)) {
                size_t rparen = line.rfind(')');
                if (rparen != std::string::npos) {
                    std::istringstream iss(line.substr(rparen + 2));
                    char state;
                    long ppid, pgrp, session, tty_nr, tpgid;
                    unsigned long flags, minflt, cminflt, majflt, cmajflt, utime = 0, stime = 0;
                    if (iss >> state >> ppid >> pgrp >> session >> tty_nr >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt >> utime >> stime) {
                        if (last_utime != 0 || last_stime != 0) {
                            uint64_t d_ticks = (utime - last_utime) + (stime - last_stime);
#if defined(__ANDROID__)
                            long clk_tck = sysconf(_SC_CLK_TCK);
                            long n_cpu = sysconf(_SC_NPROCESSORS_ONLN);
#else
                            long clk_tck = 100;
                            long n_cpu = 8;
#endif
                            if (clk_tck > 0 && n_cpu > 0 && dt > 0.0) {
                                diag_cpu_pct = float((double(d_ticks) / (double(clk_tck) * double(n_cpu) * dt)) * 100.0);
                                if (diag_cpu_pct < 0.0f) diag_cpu_pct = 0.0f;
                                if (diag_cpu_pct > 100.0f) diag_cpu_pct = 100.0f;
                            }
                        }
                        last_utime = utime;
                        last_stime = stime;
                    }
                }
            }
        }
        last_cpu_sample = now;
    }

    // diagnostic overlay: FPS + frame time + CPU % / MHz + GPU MHz + temperature + battery
    if (diag_perf_) {
        char buf[160];
        float frame_ms = diag_perf_fps_ > 0 ? (1000.0f / float(diag_perf_fps_)) : 0.0f;
        uint32_t cpu_mhz = diag_perf_cpu_[4] > 0 ? diag_perf_cpu_[4] : diag_perf_cpu_[0];
        if (diag_perf_gpu_ > 0) {
            std::snprintf(buf, sizeof(buf), "FPS: %.1f (%.1f ms) | CPU: %.0f%% (%u MHz) | GPU: %u MHz | %u\xC2\xB0" "C | Bat: %u%%",
                          diag_perf_fps_, frame_ms, diag_cpu_pct, cpu_mhz, diag_perf_gpu_, diag_perf_temp_, diag_perf_batt_);
        } else {
            std::snprintf(buf, sizeof(buf), "FPS: %.1f (%.1f ms) | CPU: %.0f%% (%u MHz) | %u\xC2\xB0" "C | Bat: %u%%",
                          diag_perf_fps_, frame_ms, diag_cpu_pct, cpu_mhz, diag_perf_temp_, diag_perf_batt_);
        }
        auto* dl = ImGui::GetForegroundDrawList();
        ImVec2 sz = ImGui::CalcTextSize(buf);
        float x = (float(swap_extent_.width) - sz.x) * 0.5f;
        float y = 14.0f;
        ImVec2 pos(x, y);
        dl->AddRectFilled(ImVec2(pos.x - 8, pos.y - 4), ImVec2(pos.x + sz.x + 8, pos.y + sz.y + 4),
                          IM_COL32(0, 0, 0, 200), 6.0f);
        dl->AddText(pos, IM_COL32(50, 255, 50, 255), buf);
    }
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), Cb());
}

void VulkanBackend::UploadGammaRamp() {
    ramp_dirty_ = false;
    Alloc a = Upload(256 * 4, 16);
    bool identity = true;
    for (int i = 0; i < 256; i++) {
        uint8_t* p = a.cpu + i * 4;
        p[0] = ramp_[512 + i];
        p[1] = ramp_[256 + i];
        p[2] = ramp_[i];
        p[3] = 255;
        identity &= ramp_[i] == i && ramp_[256 + i] == i && ramp_[512 + i] == i;
    }
    gamma_ = !identity;
    Transition(lut_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy c{a.off, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {256, 1, 1}};
    vkCmdCopyBufferToImage(Cb(), a.buf, lut_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    Transition(lut_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

// the touch controls' skins as an ImGui texture; without them the controls keep their drawn icons
void VulkanBackend::UploadSkin() {
    skin_tried_ = true;
    std::vector<uint8_t> rgba;
    uint32_t w = 0, h = 0;
    if (!ui_ || !touch::LoadSkin(rgba, w, h)) return;
    constexpr VkImageUsageFlags kSampled = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (!CreateImage(skin_, VK_FORMAT_R8G8B8A8_UNORM, w, h, 1, 1, 1, kSampled, VK_IMAGE_VIEW_TYPE_2D)) return;
    Alloc a = Upload(rgba.size(), 16);
    if (!a.cpu) {
        Destroy(skin_);
        return;
    }
    std::memcpy(a.cpu, rgba.data(), rgba.size());
    Transition(skin_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy c{a.off, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {w, h, 1}};
    vkCmdCopyBufferToImage(Cb(), a.buf, skin_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    Transition(skin_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    std::lock_guard lock(app::UiLock());
    VkDescriptorSet set = ImGui_ImplVulkan_AddTexture(linear_, skin_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    touch::SetSkinTexture(uint64_t(reinterpret_cast<uintptr_t>(set)));
}

void VulkanBackend::ReleaseOldTextures() {
    constexpr uint64_t kSweepEvery = 60;
    constexpr size_t kTexturesKept = 1024;  // a phone has less memory than a PC: fewer kept
    constexpr uint64_t kIdleFrames = 600;
    if (presents_ % kSweepEvery != 0 || textures_.size() <= kTexturesKept) return;
    for (auto it = textures_.begin(); it != textures_.end();) {
        if (it->second.frame + kIdleFrames < presents_) {
            Retire(it->second.img);
            it = textures_.erase(it);
        } else {
            ++it;
        }
    }
}

void VulkanBackend::ApplySettings() {
    float want = app::RenderScale(guest_h_);
    if (std::fabs(want - scale_) <= 1e-4f) return;
    Submit(true, false);
    for (auto& [key, t] : targets_) {
        Retire(t.img);
        Retire(t.fb);
    }
    targets_.clear();
    draw_rt_ = draw_ds_ = nullptr;
    scale_ = want;
    LOGI("vulkan: render scale %.3f", scale_);
}

void VulkanBackend::SavePipelineCache() {
    size_t size = 0;
    if (vkGetPipelineCacheData(dev_, cache_, &size, nullptr) != VK_SUCCESS || size == cache_saved_) return;
    std::vector<char> blob(size);
    if (vkGetPipelineCacheData(dev_, cache_, &size, blob.data()) != VK_SUCCESS) return;
    std::filesystem::path p = app::ExeDir() / "shadercache" / "pipelines.bin", tmp = p;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        f.write(blob.data(), std::streamsize(size));
    }
    std::error_code ec;
    std::filesystem::rename(tmp, p, ec);
    cache_saved_ = size;
}

void VulkanBackend::Present(const Surface& front) {
    const auto perf2_start = std::chrono::steady_clock::now();
    Target* t = EnsureTarget(front, false);
    static std::atomic<uint32_t> diag_presents{0};
    uint32_t diag_n = diag_presents.fetch_add(1, std::memory_order_relaxed);
    if (diag_n < 8) LOGR("vulkan: present[%u] front=%08X size=%u target=%s drawn=%d", diag_n, front.data,
                        front.size, t ? "yes" : "no", t ? t->drawn : 0);
    if (diag_n < 8 && t) LOGR("vulkan: present[%u] guest-target=%ux%u gpu-target=%ux%u", diag_n, t->gw, t->gh,
                              t->w, t->h);
    if (t && LatchCpuWrite(*t)) UploadTarget(*t, front);
    Target* display = t;
    bool reused_display = false;
    if (t) {
        const uint64_t key = TargetMapKey(t->data, false, t->gw, t->gh);
        uint64_t& presented = presented_serial_[key];
        if (t->content_serial > presented) {
            presented = t->content_serial;
            last_present_key_ = key;
        } else if (last_present_key_ && last_present_key_ != key) {
            auto previous = targets_.find(last_present_key_);
            if (previous != targets_.end()) {
                display = &previous->second;
                reused_display = true;
            } else {
                last_present_key_ = 0;
            }
        }
    }
    if (g_capture_front_after_input.exchange(false, std::memory_order_acq_rel)) {
        LOGR("vulkan: post-input-present-select front=%08X target=%08X/%ux%u serial=%llu drawn=%u display=%08X/%ux%u serial=%llu drawn=%u reused=%u last=%016llX key=%016llX",
             front.data, t ? t->data : 0, t ? t->gw : 0, t ? t->gh : 0,
             static_cast<unsigned long long>(t ? t->content_serial : 0), t ? t->drawn : 0,
             display ? display->data : 0, display ? display->gw : 0, display ? display->gh : 0,
             static_cast<unsigned long long>(display ? display->content_serial : 0), display ? display->drawn : 0,
             reused_display, static_cast<unsigned long long>(last_present_key_),
             static_cast<unsigned long long>(t ? TargetMapKey(t->data, false, t->gw, t->gh) : 0));
        if (!t) {
            LOGR("vulkan: post-input-front-capture missing target front=%08X", front.data);
        } else {
            EndRender();
            Image down;
            Image* src = CopySource(*t, down);
            std::vector<uint8_t> px;
            bool ok = src && ReadBack(*src, t->gw, t->gh, px);
            Destroy(down);
            if (ok) {
                std::filesystem::path path = app::ExeDir() / "forza-front-after-input.bmp";
                std::error_code ec;
                std::filesystem::create_directories(path.parent_path(), ec);
                std::ofstream f(path, std::ios::binary);
                uint8_t header[54] = {'B','M'};
                auto put = [&](size_t o, uint32_t v) { std::memcpy(header + o, &v, sizeof(v)); };
                put(2, uint32_t(54 + px.size())); put(10, 54); put(14, 40);
                put(18, t->gw); put(22, uint32_t(-int32_t(t->gh)));
                header[26] = 1; header[28] = 32; put(34, uint32_t(px.size()));
                if (f) {
                    f.write(reinterpret_cast<const char*>(header), sizeof(header));
                    f.write(reinterpret_cast<const char*>(px.data()), std::streamsize(px.size()));
                }
                uint32_t non_black = 0, alpha_nonzero = 0;
                for (size_t i = 0; i + 3 < px.size(); i += 4) {
                    if (px[i] || px[i + 1] || px[i + 2]) ++non_black;
                    if (px[i + 3]) ++alpha_nonzero;
                }
                LOGR("vulkan: post-input-front-capture path=%s front=%08X size=%ux%u drawn=%d nonblack=%u alpha=%u saved=%d",
                     path.string().c_str(), front.data, t->gw, t->gh, t->drawn,
                     non_black, alpha_nonzero, f.good());
            } else {
                LOGR("vulkan: post-input-front-capture readback failed front=%08X size=%ux%u",
                     front.data, t->gw, t->gh);
            }
        }
        // Capture the target actually selected for the fullscreen blit as well as the
        // just-presented surface. Rotating XDK back buffers can make those differ.
        if (display && display != t) {
            EndRender();
            Image down;
            Image* src = CopySource(*display, down);
            std::vector<uint8_t> px;
            bool ok = src && ReadBack(*src, display->gw, display->gh, px);
            Destroy(down);
            if (ok) {
                std::filesystem::path path = app::ExeDir() / "forza-display-after-input.bmp";
                std::error_code ec;
                std::filesystem::create_directories(path.parent_path(), ec);
                std::ofstream f(path, std::ios::binary);
                uint8_t header[54] = {'B','M'};
                auto put = [&](size_t o, uint32_t v) { std::memcpy(header + o, &v, sizeof(v)); };
                put(2, uint32_t(54 + px.size())); put(10, 54); put(14, 40);
                put(18, display->gw); put(22, uint32_t(-int32_t(display->gh)));
                header[26] = 1; header[28] = 32; put(34, uint32_t(px.size()));
                if (f) {
                    f.write(reinterpret_cast<const char*>(header), sizeof(header));
                    f.write(reinterpret_cast<const char*>(px.data()), std::streamsize(px.size()));
                }
                LOGR("vulkan: post-input-display-capture path=%s data=%08X size=%ux%u serial=%llu saved=%u",
                     path.string().c_str(), display->data, display->gw, display->gh,
                     static_cast<unsigned long long>(display->content_serial), f.good());
            } else {
                LOGR("vulkan: post-input-display-capture readback failed data=%08X size=%ux%u",
                     display->data, display->gw, display->gh);
            }
        }
        // The title rotates three XDK surfaces while composing the menu. The
        // selected front can therefore be black even when another surface has
        // the actual UI. Capture each full-size color target once, in the same
        // present window, before changing the buffer-rotation policy.
        static uint32_t target_capture_id = 0;
        uint32_t captured_targets = 0;
        for (auto& [target_key, candidate] : targets_) {
            (void)target_key;
            if (candidate.depth || candidate.gw != 640 || candidate.gh != 480 || !candidate.drawn) continue;
            if (captured_targets++ >= 3) break;
            EndRender();
            Image down;
            Image* src = CopySource(candidate, down);
            std::vector<uint8_t> px;
            const bool ok = src && ReadBack(*src, candidate.gw, candidate.gh, px);
            Destroy(down);
            uint32_t non_black = 0;
            for (size_t i = 0; i + 3 < px.size(); i += 4)
                if (px[i] || px[i + 1] || px[i + 2]) ++non_black;
            char filename[80];
            std::snprintf(filename, sizeof(filename), "forza-target-after-input-%08X.bmp", candidate.data);
            const std::filesystem::path path = app::ExeDir() / filename;
            std::error_code ec;
            std::filesystem::create_directories(path.parent_path(), ec);
            std::ofstream f(path, std::ios::binary);
            uint8_t header[54] = {'B','M'};
            auto put = [&](size_t o, uint32_t v) { std::memcpy(header + o, &v, sizeof(v)); };
            put(2, uint32_t(54 + px.size())); put(10, 54); put(14, 40);
            put(18, candidate.gw); put(22, uint32_t(-int32_t(candidate.gh)));
            header[26] = 1; header[28] = 32; put(34, uint32_t(px.size()));
            if (f && ok) {
                f.write(reinterpret_cast<const char*>(header), sizeof(header));
                f.write(reinterpret_cast<const char*>(px.data()), std::streamsize(px.size()));
            }
            LOGR("vulkan: target-after-input[%u] data=%08X size=%ux%u serial=%llu drawn=%u nonblack=%u ok=%u saved=%u",
                 target_capture_id++, candidate.data, candidate.gw, candidate.gh,
                 static_cast<unsigned long long>(candidate.content_serial), candidate.drawn,
                 non_black, ok, f.good());
        }
    }
    if (!trace_dir_.empty()) {
        LOGR("vulkan: frame-trace end draws=%u front=%08X display=%08X reused=%u serial=%llu",
             trace_draw_, front.data, display ? display->data : 0, reused_display,
             static_cast<unsigned long long>(display ? display->content_serial : 0));
        if (display) {
            TraceImage(display->img, "present-raw");
            Image down;
            Image* src = CopySource(*display, down);
            if (src && src != &display->img) TraceImage(*src, "present-shrink");
            Destroy(down);
        }
    }
    if (ramp_dirty_) UploadGammaRamp();
    if (!skin_tried_) UploadSkin();
    uint32_t index = 0;
    bool shown = AcquireImage(index);
    if (diag_n < 8 || diag_n % 600 == 0)
        LOGR("vulkan: present-cycle[%u] foreground=%u front=%08X target=%p display=%p reused=%u serial=%llu acquired=%u swap=%p extent=%ux%u stale=%u surface-lost=%u",
             diag_n, host::Foreground(), front.data, (void*)t, (void*)display, reused_display,
             static_cast<unsigned long long>(display ? display->content_serial : 0), shown, (void*)swap_,
             swap_extent_.width, swap_extent_.height, swap_stale_, surface_lost_);
    if (shown) {
        present_index_ = index;
        const auto draw_frame_start = perf2_ ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
        DrawFrame(display, index);
        if (perf2_)
            perf2_drawframe_ms_ +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - draw_frame_start).count();
    }
    Submit(false, shown);
    const auto perf2_submitted = std::chrono::steady_clock::now();
    present_value_[presents_ % (kFrames + 1)].store(value_);
    if (shown) {
        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &swap_done_[index];
        pi.swapchainCount = 1;
        pi.pSwapchains = &swap_;
        pi.pImageIndices = &index;
        VkResult r = vkQueuePresentKHR(queue_, &pi);
        if (diag_n < 8) LOGR("vulkan: present[%u] acquire=ok queue-present=%d", diag_n, int(r));
        if (r == VK_ERROR_OUT_OF_DATE_KHR) swap_stale_ = true;
        if (r == VK_ERROR_SURFACE_LOST_KHR) surface_lost_ = true;
    }
    presents_++;
    presents_seen_.store(presents_);
    frame_++;
    Frame& next = CurrentFrame();
    ResetFrame(next);
    vkResetCommandPool(dev_, next.pool, 0);
    Begin();
    const auto perf2_waited = std::chrono::steady_clock::now();
    if (perf2_) {
        ++perf2_frames_;
        perf2_record_ms_ += std::chrono::duration<double, std::milli>(perf2_submitted - perf2_start).count();
        perf2_wait_ms_ += std::chrono::duration<double, std::milli>(perf2_waited - perf2_submitted).count();
        if (perf2_last_.time_since_epoch().count())
            perf2_frame_ms_ += std::chrono::duration<double, std::milli>(perf2_start - perf2_last_).count();
        perf2_last_ = perf2_start;
        const double window = std::chrono::duration<double>(perf2_waited - perf2_t_).count();
        if (window >= 1.0) {
            const double d = double(perf2_frames_);
            LOGR("vulkan: perf2 fps=%.1f frames=%u frame=%.2fms record=%.2fms wait=%.2fms drawframe=%.2fms "
                 "draws=%.0f upload=%.1fMB tex=%.1f newtex=%.1f feedback=%.1f breaks=%.1f constreuse=%.0f cube=%u",
                 d / window, perf2_frames_, perf2_frame_ms_ / d, perf2_record_ms_ / d, perf2_wait_ms_ / d,
                 perf2_drawframe_ms_ / d, double(perf2_draws_) / d, double(perf2_bytes_) / d / (1024.0 * 1024.0),
                 double(perf2_tex_) / d, double(perf2_tex_new_) / d, double(perf2_feedback_) / d,
                 double(perf2_pass_breaks_) / d, double(perf2_const_reuse_) / d, perf2_cube_blits_);
            {
                // the render targets that take the frame's draws, biggest first
                std::vector<Target*> hot;
                for (auto& entry : targets_) hot.push_back(&entry.second);
                std::sort(hot.begin(), hot.end(),
                          [](const Target* a, const Target* b) { return a->perf2_draws > b->perf2_draws; });
                for (size_t i = 0; i < hot.size() && i < 5; ++i) {
                    if (!hot[i]->perf2_draws) break;
                    LOGR("vulkan: perf2-rt data=%08X %ux%u guest=%ux%u draws=%u depth=%u",
                         hot[i]->data, hot[i]->w, hot[i]->h, hot[i]->gw, hot[i]->gh, hot[i]->perf2_draws,
                         hot[i]->depth ? 1u : 0u);
                }
                for (Target* t : hot) t->perf2_draws = 0;
            }
            perf2_t_ = perf2_waited;
            perf2_frames_ = 0;
            perf2_draws_ = 0;
            perf2_bytes_ = 0;
            perf2_tex_ = perf2_tex_new_ = perf2_feedback_ = perf2_pass_breaks_ = perf2_const_reuse_ = 0;
            perf2_record_ms_ = perf2_wait_ms_ = perf2_frame_ms_ = perf2_drawframe_ms_ = 0;
            perf2_cube_blits_ = 0;
        }
    }
    ReleaseOldTextures();
    fps_n_++;
    auto now = std::chrono::steady_clock::now();
    double el = std::chrono::duration<double>(now - fps_t_).count();
    if (el >= 0.5) {
        fps_ = fps_n_ / el;
        fps_n_ = 0;
        fps_t_ = now;
    }
    if (presents_ % 600 == 0) SavePipelineCache();
    dynres::Frame();
    ApplySettings();
    trace_dir_.clear();
    std::error_code trigger_ec;
    if (presents_ % 60 == 0) {
        perf2_ = std::filesystem::exists("/sdcard/forza_diag_perf2", trigger_ec);
        g_diag_skycheck = std::filesystem::exists("/sdcard/forza_diag_skycheck", trigger_ec);
        g_diag_skystrip = std::filesystem::exists("/sdcard/forza_diag_skystrip", trigger_ec);
        g_diag_noclear = std::filesystem::exists("/sdcard/forza_diag_noclear", trigger_ec);
        g_diag_nocull = std::filesystem::exists("/sdcard/forza_diag_nocull", trigger_ec);
        g_diag_nodepth = std::filesystem::exists("/sdcard/forza_diag_nodepth", trigger_ec);
        if (std::filesystem::exists("/sdcard/forza_trace_frame", trigger_ec) ||
            std::filesystem::exists("/sdcard/Download/forza_trace_frame", trigger_ec)) {
            std::filesystem::remove("/sdcard/forza_trace_frame", trigger_ec);
            std::filesystem::remove("/sdcard/Download/forza_trace_frame", trigger_ec);
            g_capture_render_frame.store(true, std::memory_order_release);
        }
    }
    if (g_capture_render_frame.exchange(false, std::memory_order_acq_rel)) {
        const auto stamp = std::chrono::system_clock::now().time_since_epoch().count();
        auto dir = app::ExeDir() / ("forza-frame-" + std::to_string(stamp) + "-" + std::to_string(presents_));
        std::error_code error;
        if (std::filesystem::create_directories(dir, error)) {
            trace_dir_ = std::move(dir);
            trace_draw_ = trace_samples_ = trace_mesh_samples_ = trace_scene_draw_ = 0;
            trace_last_target_ = nullptr;
            trace_depth_captured_.clear();
            trace_relative_ = 0;
            trace_sky_window_ = false;
            trace_sky_saved_ = 0;
            trace_sky_mean_ = -1.0;
            trace_sky_on_ = 0;
            trace_strip_ = false;
            trace_strip_count_ = 0;
    trace_prev_pixels_.clear();
    trace_prev_rt_ = 0;
            LOGR("vulkan: frame-trace begin frame=%llu dir=%s",
                 static_cast<unsigned long long>(presents_), trace_dir_.string().c_str());
        }
    }
}

// the slot is not reused before the caller's wait: the title runs at most one present ahead
void VulkanBackend::WaitGpuFrame(uint64_t n) { WaitValue(present_value_[n % (kFrames + 1)].load()); }

VulkanBackend* g_vulkan = nullptr;

}  // namespace

std::unique_ptr<Backend> CreateBackend() {
    auto backend = std::make_unique<VulkanBackend>();
    g_vulkan = backend.get();
    return backend;
}

void WaitGpuFrame(uint64_t n) {
    if (g_vulkan) g_vulkan->WaitGpuFrame(n);
}

}  // namespace gpu
