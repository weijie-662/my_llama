// FPGA GEMM backend (shared-memory + UIO stub).
// Boilerplate only: replace shared-region and graph_compute FLASH_ATTN_EXT with real FPGA/DMA when ready.

#include "ggml-impl.h"
#include "ggml-fpga.h"
#include "ggml-backend-impl.h"
#include "ggml.h"

// Some ggml branches use GGML_OP_FLASH_ATTN, others use GGML_OP_FLASH_ATTN_EXT.
// Treat GGML_OP_FLASH_ATTN as an alias for GGML_OP_FLASH_ATTN_EXT when it is not defined.
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

/* new implementation */
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <vector>
#include <memory>

struct fpga_shared_region {
    xrt::device device;
    
    // Four dedicated BOs for Flash Attention
    struct {
        xrt::bo bo;
        void*   virt_addr;
        size_t  phys_addr;
        size_t  size;
    } q, k, v, o;

    bool initialized = false;
};

static std::unique_ptr<fpga_shared_region> g_fpga_reg = nullptr;

bool fpga_shared_region_init(size_t q_sz, size_t k_sz, size_t v_sz, size_t o_sz, int device_index = 0) {
    try {
        g_fpga_reg = std::make_unique<fpga_shared_region>();
        g_fpga_reg->device = xrt::device(device_index);

        auto alloc_bo = [&](auto& node, size_t size, const std::string& name) {
            node.size = (size + 4095) & ~4095; // Align to 4K
            node.bo = xrt::bo(g_fpga_reg->device, node.size, xrt::bo::flags::normal, 0);
            node.virt_addr = node.bo.map<void*>();
            node.phys_addr = node.bo.address();
            std::cout << "[FPGA] Allocated " << name << " at Phys: 0x" << std::hex << node.phys_addr << std::dec << std::endl;
        };

        alloc_bo(g_fpga_reg->q, q_sz, "Query");
        alloc_bo(g_fpga_reg->k, k_sz, "Key");
        alloc_bo(g_fpga_reg->v, v_sz, "Value");
        alloc_bo(g_fpga_reg->o, o_sz, "Output");

        g_fpga_reg->initialized = true;
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "FPGA Shared Region Init Error: %s\n", e.what());
        return false;
    }
}

enum fpga_buffer_type { Q_BUF, K_BUF, V_BUF, O_BUF };

void* fpga_shared_region_alloc(fpga_buffer_type type, size_t* out_phys_addr) {
    if (!g_fpga_reg || !g_fpga_reg->initialized) return nullptr;

    switch (type) {
        case Q_BUF: *out_phys_addr = g_fpga_reg->q.phys_addr; return g_fpga_reg->q.virt_addr;
        case K_BUF: *out_phys_addr = g_fpga_reg->k.phys_addr; return g_fpga_reg->k.virt_addr;
        case V_BUF: *out_phys_addr = g_fpga_reg->v.phys_addr; return g_fpga_reg->v.virt_addr;
        case O_BUF: *out_phys_addr = g_fpga_reg->o.phys_addr; return g_fpga_reg->o.virt_addr;
        default: return nullptr;
    }
}

void fpga_shared_region_free() {
    g_fpga_reg.reset(); // Automatically calls destructors for BOs and Device
}

// Buffer-level context
struct ggml_fpga_buffer_context {
    void *  virt_base;    // Virtual address for CPU access (mapped)
    uint64_t phys_base;   // Physical address for FPGA DMA
    size_t   size;
};

// Tensor-level metadata stored in tensor->extra
struct ggml_fpga_tensor_extra {
    uint64_t phys_addr;   // Exact physical address for this tensor
};

enum ggml_status ggml_fpga_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto * buf_ctx = (ggml_fpga_buffer_context *)buffer->context;

    // 1. Create extra metadata for this tensor
    auto * extra = new ggml_fpga_tensor_extra;
    
    // 2. Calculate physical address: Base + Offset
    // tensor->off is the offset within the buffer assigned by ggml-alloc
    extra->phys_addr = buf_ctx->phys_base + tensor->off;
    
    // 3. Attach to tensor
    tensor->extra = extra;

    // 4. Update the virtual pointer for CPU-side ops (e.g., loading weights)
    tensor->data = (char *)buf_ctx->virt_base + tensor->off;

    return GGML_STATUS_SUCCESS;
}

void ggml_fpga_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * buf_ctx = (ggml_fpga_buffer_context *)buffer->context;
    
    // Note: Individual tensor->extra cleanup usually happens 
    // when the graph is destroyed or during buffer reset.
    
    delete buf_ctx;
}

ggml_backend_buffer_t ggml_backend_fpga_buffer_alloc(size_t size) {
    auto * buf_ctx = new ggml_fpga_buffer_context();
    
    // Use the allocator we built in the previous step
    buf_ctx->virt_base = fpga_shared_region_alloc(size);
    
    // Retrieve the physical address from our global XRT context
    size_t offset = (uint8_t *)buf_ctx->virt_base - (uint8_t *)g_ctx->ptr;
    buf_ctx->phys_base = g_ctx->root_bo.address() + offset; 
    buf_ctx->size = size;

    return ggml_backend_buffer_init(
        /* buft  */ ggml_backend_fpga_buffer_type(),
        /* iface */ ggml_backend_fpga_buffer_interface,
        /* ctx   */ buf_ctx,
        /* size  */ size
    );
}

// The following is the obsolete original version. 
// ---------------------------------------------------------------------------
// "Shared region" (test-only): plain CPU memory allocation
// ---------------------------------------------------------------------------

struct ggml_fpga_shared_region {
    size_t alignment;       // allocation alignment (e.g. 4096 for DMA)
};

static bool fpga_shared_region_init(struct ggml_fpga_shared_region * region) {
    region->alignment = 4096;
    return true;
}

static void fpga_shared_region_free(struct ggml_fpga_shared_region * region) {
    GGML_UNUSED(region);
}

static void * fpga_shared_region_alloc(struct ggml_fpga_shared_region * region, size_t size, size_t * out_offset) {
    size_t aligned = (size + region->alignment - 1) & ~(region->alignment - 1);
    if (out_offset) {
        *out_offset = 0;
    }
    void * p = nullptr;
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    if (posix_memalign(&p, region->alignment, aligned) != 0) {
        p = nullptr;
    }
#else
    p = aligned_alloc(region->alignment, aligned);
#endif
    return p;
}

// ---------------------------------------------------------------------------
// Buffer context (one buffer = one allocation)
// ---------------------------------------------------------------------------

struct ggml_fpga_buffer_context {
    struct ggml_fpga_shared_region * region;
    void *   ptr;    // base pointer (into shared region)
    size_t   size;
    bool     owned;  // if true, we allocated it (bump); if false, external
};

static void ggml_fpga_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_fpga_buffer_context * ctx = static_cast<ggml_fpga_buffer_context *>(buffer->context);
    if (ctx->owned && ctx->ptr) {
        free(ctx->ptr);
    }
    delete ctx;
}

static void * ggml_fpga_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_fpga_buffer_context * ctx = static_cast<ggml_fpga_buffer_context *>(buffer->context);
    return ctx->ptr; 
}

static ggml_status ggml_fpga_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    return GGML_STATUS_SUCCESS;
}

static void ggml_fpga_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_fpga_buffer_context * ctx = static_cast<ggml_fpga_buffer_context *>(buffer->context);
    memset(static_cast<char *>(ctx->ptr) + (tensor->data ? static_cast<char *>(tensor->data) - static_cast<char *>(ctx->ptr) : 0) + offset, value, size);
}

static void ggml_fpga_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_fpga_buffer_context * ctx = static_cast<ggml_fpga_buffer_context *>(buffer->context);
    char * base = static_cast<char *>(ctx->ptr);
    size_t tensor_off = tensor->data ? static_cast<char *>(tensor->data) - base : 0;
    memcpy(base + tensor_off + offset, data, size);
}

static void ggml_fpga_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_fpga_buffer_context * ctx = static_cast<ggml_fpga_buffer_context *>(buffer->context);
    const char * base = static_cast<const char *>(ctx->ptr);
    size_t tensor_off = tensor->data ? static_cast<const char *>(tensor->data) - base : 0;
    memcpy(data, base + tensor_off + offset, size);
}

static bool ggml_fpga_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(src);
    GGML_ASSERT(dst);

    if (!ggml_are_same_layout(src, dst)) {
        return false;
    }

    const size_t nbytes = ggml_nbytes(src);
    if (nbytes == 0) {
        return true;
    }

    // dst is in this buffer (FPGA buffer is host-accessible). Try fast paths first.
    if (ggml_backend_buffer_is_host(src->buffer)) {
        buffer->iface.set_tensor(buffer, dst, src->data, 0, nbytes);
        return true;
    }

    // Fallback: pull via generic API then push into dst.
    void * tmp = malloc(nbytes);
    if (!tmp) {
        return false;
    }
    ggml_backend_tensor_get(src, tmp, 0, nbytes);
    buffer->iface.set_tensor(buffer, dst, tmp, 0, nbytes);
    free(tmp);
    return true;
}

static void ggml_fpga_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_fpga_buffer_context * ctx = static_cast<ggml_fpga_buffer_context *>(buffer->context);
    memset(ctx->ptr, value, ctx->size);
}

static struct ggml_backend_buffer_i const ggml_fpga_buffer_i = {
    /* .free_buffer   = */ ggml_fpga_buffer_free_buffer,
    /* .get_base      = */ ggml_fpga_buffer_get_base,
    /* .init_tensor   = */ ggml_fpga_buffer_init_tensor,
    /* .memset_tensor = */ ggml_fpga_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_fpga_buffer_set_tensor,
    /* .get_tensor    = */ ggml_fpga_buffer_get_tensor,
    /* .cpy_tensor    = */ ggml_fpga_buffer_cpy_tensor,
    /* .clear         = */ ggml_fpga_buffer_clear,
    /* .reset         = */ nullptr,
};

// ---------------------------------------------------------------------------
// Buffer type (allocates from shared region)
// ---------------------------------------------------------------------------

struct ggml_fpga_device_context {
    struct ggml_fpga_shared_region shared_region;
};

static const char * ggml_fpga_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_FPGA_NAME;
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_fpga_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_device * dev = buft->device;
    size_t align = 4096;
    if (dev && dev->context) {
        ggml_fpga_device_context * dev_ctx = static_cast<ggml_fpga_device_context *>(dev->context);
        align = dev_ctx->shared_region.alignment;
    }
    size_t aligned_size = (size + align - 1) & ~(align - 1);
    size_t offset = 0;
    struct ggml_fpga_shared_region stub_region = {};
    stub_region.alignment = align;
    if (dev && dev->context) {
        ggml_fpga_device_context * dev_ctx = static_cast<ggml_fpga_device_context *>(dev->context);
        void * ptr = fpga_shared_region_alloc(&dev_ctx->shared_region, aligned_size, &offset);
        if (!ptr) {
            return nullptr;
        }
        ggml_fpga_buffer_context * ctx = new ggml_fpga_buffer_context{};
        ctx->region = &dev_ctx->shared_region;
        ctx->ptr    = ptr;
        ctx->size   = aligned_size;
        ctx->owned  = true;
        return ggml_backend_buffer_init(buft, ggml_fpga_buffer_i, ctx, aligned_size);
    }
    void * ptr = fpga_shared_region_alloc(&stub_region, aligned_size, &offset);
    if (!ptr) {
        return nullptr;
    }
    ggml_fpga_buffer_context * ctx = new ggml_fpga_buffer_context{};
    ctx->region = nullptr;
    ctx->ptr    = ptr;
    ctx->size   = aligned_size;
    ctx->owned  = true;
    return ggml_backend_buffer_init(buft, ggml_fpga_buffer_i, ctx, aligned_size);
}

static size_t ggml_fpga_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_device * dev = buft->device;
    if (dev && dev->context) {
        ggml_fpga_device_context * dev_ctx = static_cast<ggml_fpga_device_context *>(dev->context);
        return dev_ctx->shared_region.alignment;
    }
    return 4096;
    GGML_UNUSED(buft);
}

static bool ggml_fpga_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;  // CPU can read/write shared region
}

// ---------------------------------------------------------------------------
// FPGA capture and stats (env: GGML_FPGA_CAPTURE=1, GGML_FPGA_LOG=1)
// ---------------------------------------------------------------------------

static int s_fpga_capture_index = 0;
static int s_fpga_ops_total     = 0;
static int s_fpga_log_written   = 0;  // 1 after first layout append
static int s_fpga_dump_index    = 0;

static int fpga_getenv_capture(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * v = getenv("GGML_FPGA_CAPTURE");
        cached = (v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y')) ? 1 : 0;
    }
    return cached;
}

static int fpga_getenv_log(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * v = getenv("GGML_FPGA_LOG");
        cached = (v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y')) ? 1 : 0;
    }
    return cached;
}

static void ggml_fpga_ensure_outputs_dir(void) {
    if (mkdir("outputs", 0755) != 0) {
        if (errno != EEXIST) {
            // Best-effort; ignore failures.
        }
    }
}

static void ggml_fpga_dump_tensor_to(const ggml_tensor * t, const char * path) {
    if (!t || !t->data) {
        return;
    }
    FILE * f = fopen(path, "wb");
    if (!f) {
        return;
    }
    fwrite(t->data, 1, ggml_nbytes(t), f);
    fclose(f);
}

static void ggml_fpga_dump_tensor_hex_to(const ggml_tensor * t, const char * path) {
    if (!t || !t->data) {
        return;
    }
    FILE * f = fopen(path, "w");
    if (!f) {
        return;
    }

    const unsigned char * p = (const unsigned char *) t->data;
    const size_t n = (size_t) ggml_nbytes(t);

    fprintf(f, "type=%s nbytes=%zu ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] contiguous=%d\n",
            ggml_type_name(t->type), n,
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3],
            ggml_is_contiguous(t) ? 1 : 0);

    // If this is a float tensor with a standard element size, also decode values.
    if (t->type == GGML_TYPE_F32) {
        const int64_t nelem = ggml_nelements(t);
        const size_t  step  = sizeof(float);
        const size_t  max_e = n / step;
        const size_t  n_e   = (size_t) (nelem < (int64_t) max_e ? nelem : (int64_t) max_e);

        fputs("index  hex32       value\n", f);
        for (size_t i = 0; i < n_e; ++i) {
            uint32_t u32 = 0;
            float    v   = 0.0f;
            memcpy(&u32, p + i*step, sizeof(uint32_t));
            memcpy(&v,   p + i*step, sizeof(float));
            fprintf(f, "%8zu  %08x  %.4f\n", i, u32, v);
        }

        fclose(f);
        return;
    }

    if (t->type == GGML_TYPE_F16) {
        const int64_t nelem = ggml_nelements(t);
        const size_t  step  = sizeof(ggml_fp16_t);
        const size_t  max_e = n / step;
        const size_t  n_e   = (size_t) (nelem < (int64_t) max_e ? nelem : (int64_t) max_e);

        fputs("index  hex16  value\n", f);
        for (size_t i = 0; i < n_e; ++i) {
            ggml_fp16_t h = 0;
            memcpy(&h, p + i*step, sizeof(ggml_fp16_t));
            const float v = GGML_FP16_TO_FP32(h);
            fprintf(f, "%8zu  %04x  %.4f\n", i, (unsigned) h, v);
        }

        fclose(f);
        return;
    }

    // Fallback: simple hexdump: offset + 16 bytes per line.
    for (size_t i = 0; i < n; i += 16) {
        fprintf(f, "%08zx  ", i);
        const size_t line = (n - i) < 16 ? (n - i) : 16;
        for (size_t j = 0; j < 16; ++j) {
            if (j < line) {
                fprintf(f, "%02x ", p[i + j]);
            } else {
                fputs("   ", f);
            }
        }
        fputs(" |", f);
        for (size_t j = 0; j < line; ++j) {
            const unsigned char c = p[i + j];
            fputc((c >= 32 && c <= 126) ? (int) c : '.', f);
        }
        fputs("|\n", f);
    }

    fclose(f);
}

static void ggml_fpga_dump_tensor_bin_and_ascii(const ggml_tensor * t, const char * bin_path) {
    ggml_fpga_dump_tensor_to(t, bin_path);

    char txt_path[512];
    snprintf(txt_path, sizeof(txt_path), "%s.txt", bin_path);
    ggml_fpga_dump_tensor_hex_to(t, txt_path);
}

static void ggml_fpga_dump_flash_attn_tensors(ggml_tensor * node, int n, bool is_before) {
    ggml_fpga_ensure_outputs_dir();

    char path[256];
    const ggml_tensor * q     = node->src[0];
    const ggml_tensor * k     = node->src[1];
    const ggml_tensor * v     = node->src[2];
    const ggml_tensor * mask  = node->src[3];
    const ggml_tensor * sinks = node->src[4];

    const char * tag = is_before ? "before" : "after";

    snprintf(path, sizeof(path), "outputs/fpga_flash_attn_%s_Q.bin", tag);
    ggml_fpga_dump_tensor_bin_and_ascii(q, path);
    snprintf(path, sizeof(path), "outputs/fpga_flash_attn_%s_K.bin", tag);
    ggml_fpga_dump_tensor_bin_and_ascii(k, path);
    snprintf(path, sizeof(path), "outputs/fpga_flash_attn_%s_V.bin", tag);
    ggml_fpga_dump_tensor_bin_and_ascii(v, path);

    if (mask) {
        snprintf(path, sizeof(path), "outputs/fpga_flash_attn_%s_mask.bin", tag);
        ggml_fpga_dump_tensor_bin_and_ascii(mask, path);
    }

    if (sinks) {
        snprintf(path, sizeof(path), "outputs/fpga_flash_attn_%s_sinks.bin", tag);
        ggml_fpga_dump_tensor_bin_and_ascii(sinks, path);
    }

    if (!is_before) {
        snprintf(path, sizeof(path), "outputs/fpga_flash_attn_%s_dst.bin", tag);
        ggml_fpga_dump_tensor_bin_and_ascii(node, path);
    }
}

static void ggml_fpga_fprint_tensor_layout(FILE * f, const ggml_tensor * t, const char * label) {
    if (!t) return;
    fprintf(f, "%s: ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] type=%s nbytes=%zu contiguous=%d\n",
            label,
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3],
            ggml_type_name(t->type), (size_t) ggml_nbytes(t), ggml_is_contiguous(t) ? 1 : 0);
}

static void ggml_fpga_log_flash_attn_layout(FILE * f, const ggml_tensor * node) {
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    const ggml_tensor * mask = node->src[3];
    ggml_fpga_fprint_tensor_layout(f, q, "Q");
    ggml_fpga_fprint_tensor_layout(f, k, "K");
    ggml_fpga_fprint_tensor_layout(f, v, "V");
    ggml_fpga_fprint_tensor_layout(f, node, "dst");
    fprintf(f, "op_params[0..7]: %d %d %d %d %d %d %d %d\n",
            ggml_get_op_params_i32(node, 0), ggml_get_op_params_i32(node, 1),
            ggml_get_op_params_i32(node, 2), ggml_get_op_params_i32(node, 3),
            ggml_get_op_params_i32(node, 4), ggml_get_op_params_i32(node, 5),
            ggml_get_op_params_i32(node, 6), ggml_get_op_params_i32(node, 7));
    if (mask) {
        ggml_fpga_fprint_tensor_layout(f, mask, "mask");
    }
}

static void ggml_fpga_print_stats_flash_attn(const ggml_tensor * node) {
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
}

static void ggml_fpga_capture_flash_attn_inputs(ggml_tensor * node, int n) {
    char path[256];
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    snprintf(path, sizeof(path), "fpga_capture_%d_Q.bin", n);
    FILE * fq = fopen(path, "wb");
    if (fq) { if (q->data) fwrite(q->data, 1, ggml_nbytes(q), fq); fclose(fq); }
    snprintf(path, sizeof(path), "fpga_capture_%d_K.bin", n);
    FILE * fk = fopen(path, "wb");
    if (fk) { if (k->data) fwrite(k->data, 1, ggml_nbytes(k), fk); fclose(fk); }
    snprintf(path, sizeof(path), "fpga_capture_%d_V.bin", n);
    FILE * fv = fopen(path, "wb");
    if (fv) { if (v->data) fwrite(v->data, 1, ggml_nbytes(v), fv); fclose(fv); }
}

static void ggml_fpga_capture_flash_attn_output(ggml_tensor * node, int n) {
    char path[256];
    snprintf(path, sizeof(path), "fpga_capture_%d_dst.bin", n);
    FILE * fd = fopen(path, "wb");
    if (fd) { if (node->data) fwrite(node->data, 1, ggml_nbytes(node), fd); fclose(fd); }
    snprintf(path, sizeof(path), "fpga_capture_%d_meta.txt", n);
    FILE * fm = fopen(path, "w");
    if (fm) {
        ggml_fpga_log_flash_attn_layout(fm, node);
        fclose(fm);
    }
}

// ---------------------------------------------------------------------------
// Test-only FLASH_ATTN_EXT reference implementation (C loops).
// This is correctness-first and single-threaded; replace with FPGA kernel later.
// ---------------------------------------------------------------------------

struct ggml_fpga_compute_params {
    int ith, nth;
    size_t wsize;
    void * wdata;
    struct ggml_threadpool * threadpool;
    bool use_ref;
};

static inline float ggml_fpga_load_f32(const ggml_tensor * t, const char * p) {
    switch (t->type) {
        case GGML_TYPE_F32: return *(const float *) p;
        case GGML_TYPE_F16: return GGML_FP16_TO_FP32(*(const ggml_fp16_t *) p);
        default: GGML_ABORT("ggml-fpga: unsupported type in flash_attn_ext ref");
    }
}

static void ggml_fpga_compute_forward_flash_attn_ext_ref(
    const struct ggml_fpga_compute_params * params,
    struct ggml_tensor * dst) {
    GGML_UNUSED(params);

    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    GGML_ASSERT(q && k && v);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(q->type == GGML_TYPE_F32 || q->type == GGML_TYPE_F16);
    GGML_ASSERT(k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16);
    GGML_ASSERT(v->type == GGML_TYPE_F32 || v->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t N  = neq1;

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);
    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t rk2 = neq2/nek2;
    const int64_t rk3 = neq3/nek3;
    const int64_t rv2 = neq2/nev2;
    const int64_t rv3 = neq3/nev3;

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = (uint32_t) neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2((double) n_head));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const int64_t nr = neq1 * neq2 * neq3; // total q rows

    for (int64_t ir = 0; ir < nr; ++ir) {
        const int iq3 = (int) (ir/(neq2*neq1));
        const int iq2 = (int) ((ir - (int64_t) iq3*neq2*neq1)/neq1);
        const int iq1 = (int) (ir - (int64_t) iq3*neq2*neq1 - (int64_t) iq2*neq1);

        const uint32_t h = (uint32_t) iq2;
        const float slope = (max_bias > 0.0f) ? (h < n_head_log2 ? powf(m0, (float) (h + 1)) : powf(m1, (float) (2*(h - n_head_log2) + 1))) : 1.0f;

        float M = -INFINITY;
        float S = 0.0f;

        float * VKQ = (float *) alloca((size_t) DV * sizeof(float));
        for (int64_t d = 0; d < DV; ++d) VKQ[d] = 0.0f;

        const ggml_fp16_t * mp = nullptr;
        if (mask) {
            // matches CPU: mask indexed by (q_pos, head%mask_heads, batch%mask_batch)
            mp = (const ggml_fp16_t *)((const char *) mask->data +
                    (int64_t) iq1*mask->nb[1] +
                    (int64_t) (iq2 % mask->ne[2])*mask->nb[2] +
                    (int64_t) (iq3 % mask->ne[3])*mask->nb[3]);
        }

        const int ik3 = iq3 / (int) rk3;
        const int ik2 = iq2 / (int) rk2;
        const int iv3 = iq3 / (int) rv3;
        const int iv2 = iq2 / (int) rv2;

        const char * q_row = (const char *) q->data + (int64_t) iq1*nbq1 + (int64_t) iq2*nbq2 + (int64_t) iq3*nbq3;

        // online softmax / attention over kv sequence (nek1)
        for (int64_t ic = 0; ic < nek1; ++ic) {
            const float mv = mp ? (slope * GGML_FP16_TO_FP32(mp[ic])) : 0.0f;
            if (mv == -INFINITY) {
                continue;
            }

            const char * k_row = (const char *) k->data + ic*nbk1 + (int64_t) ik2*nbk2 + (int64_t) ik3*nbk3;

            float dot = 0.0f;
            for (int64_t d = 0; d < DK; ++d) {
                const char * qp = q_row + d*nbq0;
                const char * kp = k_row + d*nbk0;
                dot += ggml_fpga_load_f32(q, qp) * ggml_fpga_load_f32(k, kp);
            }

            float s = dot * scale;
            if (logit_softcap != 0.0f) {
                s = logit_softcap * tanhf(s);
            }
            s += mv;

            const float Mold = M;
            float ms = 1.0f;
            float vs = 1.0f;

            if (s > M) {
                M = s;
                ms = expf(Mold - M);
                // VKQ *= ms
                for (int64_t d = 0; d < DV; ++d) {
                    VKQ[d] *= ms;
                }
            } else {
                vs = expf(s - M);
            }

            // VKQ += V[ic] * vs
            const char * v_row = (const char *) v->data + ic*nbv1 + (int64_t) iv2*nbv2 + (int64_t) iv3*nbv3;
            for (int64_t d = 0; d < DV; ++d) {
                const char * vp = v_row + d*nbv0;
                VKQ[d] += ggml_fpga_load_f32(v, vp) * vs;
            }

            S = S*ms + vs;
        }

        // sinks (apply as if they were included in softmax, but without contributing to VKQ)
        if (sinks) {
            const float s_sink = ((const float *) sinks->data)[h];
            float ms = 1.0f;
            float vs = 1.0f;

            if (s_sink > M) {
                ms = expf(M - s_sink);
                M = s_sink;
                for (int64_t d = 0; d < DV; ++d) {
                    VKQ[d] *= ms;
                }
            } else {
                vs = expf(s_sink - M);
            }

            S = S*ms + vs;
        }

        const float invS = (S == 0.0f) ? 0.0f : (1.0f / S);
        for (int64_t d = 0; d < DV; ++d) {
            VKQ[d] *= invS;
        }

        // dst indices: permute(0, 2, 1, 3) in CPU impl
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;
        memcpy((char *) dst->data + ((int64_t) i3*ne2*ne1 + i2 + (int64_t) i1*ne1)*nb1, VKQ, (size_t) nb1);
    }
}

static void ggml_fpga_flash_attn_ext_stub(ggml_tensor * node) {
    static const struct ggml_fpga_compute_params params = {
        /* .ith = */ 0,
        /* .nth = */ 1,
        /* .wsize = */ 0,
        /* .wdata = */ nullptr,
        /* .threadpool = */ nullptr,
        /* .use_ref = */ false,
    };
    ggml_fpga_compute_forward_flash_attn_ext_ref(&params, node);
}

// ---------------------------------------------------------------------------
// Backend context
// ---------------------------------------------------------------------------

struct ggml_backend_fpga_context {
    ggml_fpga_device_context * dev_ctx;
};

// ---------------------------------------------------------------------------
// Backend interface
// ---------------------------------------------------------------------------

static const char * ggml_backend_fpga_get_name(ggml_backend_t backend) {
    return GGML_FPGA_NAME;
    GGML_UNUSED(backend);
}

static void ggml_backend_fpga_free(ggml_backend_t backend) {
    ggml_backend_fpga_context * ctx = static_cast<ggml_backend_fpga_context *>(backend->context);
    delete ctx;
    delete backend;
}

static void ggml_backend_fpga_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_UNUSED(backend);
    ggml_backend_tensor_set(tensor, data, offset, size);
}

static void ggml_backend_fpga_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(backend);
    ggml_backend_tensor_get(tensor, data, offset, size);
}

static bool ggml_backend_fpga_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(backend_src);
    GGML_UNUSED(backend_dst);

    if (!ggml_are_same_layout(src, dst)) {
        return false;
    }

    // Our buffer is host-accessible; fast path when src is host.
    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
        return true;
    }

    // Otherwise let the generic copy path handle it.
    return false;
}

static ggml_status ggml_backend_fpga_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_fpga_context * ctx = static_cast<ggml_backend_fpga_context *>(backend->context);
    GGML_UNUSED(ctx);

    int ops_this_graph = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
            continue;
        }
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            ops_this_graph++;
            ggml_fpga_print_stats_flash_attn(node);
            if (fpga_getenv_log() && !s_fpga_log_written) {
                FILE * fl = fopen("fpga_flash_attn_layout.txt", "a");
                if (fl) {
                    ggml_fpga_log_flash_attn_layout(fl, node);
                    fclose(fl);
                    s_fpga_log_written = 1;
                }
            }

            const int dump_n = s_fpga_dump_index++;
            ggml_fpga_dump_flash_attn_tensors(node, dump_n, /* is_before = */ true);

            ggml_fpga_flash_attn_ext_stub(node);

            ggml_fpga_dump_flash_attn_tensors(node, dump_n, /* is_before = */ false);

            continue;
        }
        GGML_ASSERT(false && "FPGA backend: unsupported op");
    }
    s_fpga_ops_total += ops_this_graph;
    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i const ggml_backend_fpga_i = {
    /* .get_name           = */ ggml_backend_fpga_get_name,
    /* .free               = */ ggml_backend_fpga_free,
    /* .set_tensor_async   = */ ggml_backend_fpga_set_tensor_async,
    /* .get_tensor_async   = */ ggml_backend_fpga_get_tensor_async,
    /* .cpy_tensor_async   = */ ggml_backend_fpga_cpy_tensor_async,
    /* .synchronize        = */ nullptr,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ ggml_backend_fpga_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

static ggml_guid_t ggml_backend_fpga_guid(void) {
    static ggml_guid guid = { 0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x70, 0x81, 0x92, 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0x09 };
    return &guid;
}

ggml_backend_t ggml_backend_fpga_init(ggml_backend_dev_t dev) {
    if (!dev || !dev->context) {
        return nullptr;
    }
    ggml_fpga_device_context * dev_ctx = static_cast<ggml_fpga_device_context *>(dev->context);
    ggml_backend_fpga_context * ctx = new ggml_backend_fpga_context{};
    ctx->dev_ctx = dev_ctx;

    ggml_backend_t backend = new ggml_backend{
        /* .guid    = */ ggml_backend_fpga_guid(),
        /* .iface   = */ ggml_backend_fpga_i,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
    return backend;
}

bool ggml_backend_is_fpga(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_fpga_guid());
}

// ---------------------------------------------------------------------------
// Device interface
// ---------------------------------------------------------------------------

static ggml_backend_reg_t ggml_backend_fpga_reg_static(void);

static const char * ggml_backend_fpga_device_get_name(ggml_backend_dev_t dev) {
    return GGML_FPGA_NAME;
    GGML_UNUSED(dev);
}

static const char * ggml_backend_fpga_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "FPGA Flash Attention (shared-memory stub)";
}

static void ggml_backend_fpga_device_get_memory(ggml_backend_dev_t dev, size_t * free_bytes, size_t * total_bytes) {
    GGML_UNUSED(dev);
    *free_bytes = 0;
    *total_bytes = 0;
}

static enum ggml_backend_dev_type ggml_backend_fpga_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_fpga_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_fpga_device_get_name(dev);
    props->description = ggml_backend_fpga_device_get_description(dev);
    props->type        = ggml_backend_fpga_device_get_type(dev);
    ggml_backend_fpga_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps.async                 = false;
    props->caps.host_buffer           = true;
    props->caps.buffer_from_host_ptr  = false;
    props->caps.events                = false;
}

static ggml_backend_t ggml_backend_fpga_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    return ggml_backend_fpga_init(dev);
}

static ggml_backend_buffer_type_t ggml_backend_fpga_device_get_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_fpga_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    if (op->op == GGML_OP_FLASH_ATTN_EXT) {
        const ggml_tensor * q = op->src[0];
        const ggml_tensor * k = op->src[1];
        const ggml_tensor * v = op->src[2];

        if (!q || !k || !v) return false;

        // 检查输入是否为 F32 或 F16 
        auto is_supported_type = [](ggml_type type) {
            return type == GGML_TYPE_F32 || type == GGML_TYPE_F16;
        };

        bool q_ok = is_supported_type(q->type);
        bool k_ok = is_supported_type(k->type);
        bool v_ok = is_supported_type(v->type);

        // 如果你希望只要有 Q8_0 就能分发上来调试：
        return q_ok && k_ok && v_ok;
    }
    return false;
}

// Accept own buffer type and host (CPU) buffers so the scheduler can assign FLASH_ATTN_EXT to FPGA
// when Q/K/V live in CPU memory; the backend then reads from host (e.g. stub calls CPU compute).
static bool ggml_backend_fpga_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == ggml_backend_fpga_device_get_buffer_type(dev) || ggml_backend_buft_is_host(buft);
}

static bool ggml_backend_fpga_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return ggml_backend_fpga_device_supports_op(dev, op);
}

static struct ggml_backend_device_i const ggml_backend_fpga_device_i = {
    /* .get_name             = */ ggml_backend_fpga_device_get_name,
    /* .get_description      = */ ggml_backend_fpga_device_get_description,
    /* .get_memory           = */ ggml_backend_fpga_device_get_memory,
    /* .get_type             = */ ggml_backend_fpga_device_get_type,
    /* .get_props            = */ ggml_backend_fpga_device_get_props,
    /* .init_backend         = */ ggml_backend_fpga_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_fpga_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_fpga_device_supports_op,
    /* .supports_buft        = */ ggml_backend_fpga_device_supports_buft,
    /* .offload_op           = */ ggml_backend_fpga_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// Buffer type singleton (device is stored in reg context)
static ggml_backend_buffer_type_t ggml_fpga_buft_ptr = nullptr;
static ggml_backend_device * ggml_fpga_device_ptr   = nullptr;

static ggml_backend_buffer_type_t ggml_backend_fpga_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_fpga_buft_ptr;
    GGML_UNUSED(dev);
}

// ---------------------------------------------------------------------------
// Backend reg
// ---------------------------------------------------------------------------

static const char * ggml_backend_fpga_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_FPGA_NAME;
    GGML_UNUSED(reg);
}

static size_t ggml_backend_fpga_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_fpga_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    GGML_UNUSED(reg);
    GGML_UNUSED(index);
    return ggml_fpga_device_ptr;
}

static void * ggml_backend_fpga_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static struct ggml_backend_reg_i const ggml_backend_fpga_reg_i = {
    /* .get_name         = */ ggml_backend_fpga_reg_get_name,
    /* .get_device_count = */ ggml_backend_fpga_reg_get_device_count,
    /* .get_device       = */ ggml_backend_fpga_reg_get_device,
    /* .get_proc_address = */ ggml_backend_fpga_get_proc_address,
};

ggml_backend_reg_t ggml_backend_fpga_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_fpga_reg_i,
        /* .context     = */ nullptr,
    };

    // One-time device + buffer type setup (shared region lives in device context)
    if (!ggml_fpga_device_ptr) {
        ggml_fpga_device_context * dev_ctx = new ggml_fpga_device_context{};
        if (!fpga_shared_region_init(&dev_ctx->shared_region)) {
            delete dev_ctx;
            return nullptr;
        }

        static struct ggml_backend_device fpga_device = {
            /* .iface   = */ ggml_backend_fpga_device_i,
            /* .reg     = */ &reg,
            /* .context = */ nullptr,
        };
        fpga_device.context = dev_ctx;
        ggml_fpga_device_ptr = &fpga_device;

        static struct ggml_backend_buffer_type fpga_buft = {
            /* .iface   = */ {
                /* .get_name      = */ ggml_fpga_buffer_type_get_name,
                /* .alloc_buffer  = */ ggml_fpga_buffer_type_alloc_buffer,
                /* .get_alignment = */ ggml_fpga_buffer_type_get_alignment,
                /* .get_max_size  = */ nullptr,
                /* .get_alloc_size= */ nullptr,
                /* .is_host       = */ ggml_fpga_buffer_type_is_host,
            },
            /* .device  = */ ggml_fpga_device_ptr,
            /* .context= */ nullptr,
        };
        ggml_fpga_buft_ptr = &fpga_buft;
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_fpga_reg)
