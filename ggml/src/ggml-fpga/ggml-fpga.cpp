// FPGA GEMM backend (shared-memory + UIO stub).
// Boilerplate only: replace shared-region and graph_compute FLASH_ATTN_EXT with real FPGA/DMA when ready.

#include "ggml-impl.h"
#include "ggml-fpga.h"
#include "ggml-backend-impl.h"
#include "ggml.h"

#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#ifdef __linux__
#include <sys/sysmacros.h>
#endif

// ---------------------------------------------------------------------------
// Shared region (stub: malloc when UIO not available; replace with UIO mmap later)
// ---------------------------------------------------------------------------

struct ggml_fpga_shared_region {
    void *  base_virt;      // CPU virtual base
    size_t size;            // size in bytes
    size_t alignment;       // allocation alignment (e.g. 4096 for DMA)
    bool   use_uio;         // true if backed by UIO mmap, false if malloc stub
    int    uio_fd;          // -1 or fd for /dev/uioX
#ifdef __linux__
    uintptr_t base_phys;    // physical base (for DMA; 0 in stub)
#endif
};

static bool fpga_shared_region_init(struct ggml_fpga_shared_region * region) {
    region->base_virt = nullptr;
    region->size      = 0;
    region->alignment = 4096;
    region->use_uio   = false;
    region->uio_fd    = -1;
#ifdef __linux__
    region->base_phys = 0;
#endif

    // Stub: try /dev/uio0; if present you could mmap control + shared RAM here.
    int fd = open("/dev/uio0", O_RDWR);
    if (fd >= 0) {
        close(fd);
        // TODO: mmap UIO regions and set region->base_virt, region->size, region->base_phys from sysfs.
    }

    // Stub: no real shared region yet; alloc_buffer will use aligned_alloc per buffer.
    // When UIO is wired, set region->base_virt/size here and use bump alloc in fpga_shared_region_alloc.
    return true;
}

static void fpga_shared_region_free(struct ggml_fpga_shared_region * region) {
    if (region->base_virt && !region->use_uio) {
        free(region->base_virt);
    }
    if (region->uio_fd >= 0) {
        close(region->uio_fd);
    }
    region->base_virt = nullptr;
    region->size      = 0;
    region->uio_fd    = -1;
}

// Allocate from the shared region. Stub: use aligned_alloc per buffer (replace with bump from region when using UIO).
static void * fpga_shared_region_alloc(struct ggml_fpga_shared_region * region, size_t size, size_t * out_offset) {
    size_t aligned = (size + region->alignment - 1) & ~(region->alignment - 1);
    if (out_offset) {
        *out_offset = 0;
    }
    if (region->base_virt && region->size > 0) {
        // Real shared region: bump alloc (caller must reset between graphs if needed).
        static size_t s_offset = 0;
        if (s_offset + aligned > region->size) {
            return nullptr;
        }
        size_t off = s_offset;
        s_offset += aligned;
        if (out_offset) {
            *out_offset = off;
        }
        return static_cast<char *>(region->base_virt) + off;
    }
    // Stub: no region yet, allocate independently.
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

// Physical address for DMA (stub: 0; with UIO read from sysfs maps/mapX/addr).
static uintptr_t fpga_shared_region_virt_to_phys(struct ggml_fpga_shared_region const * region, void const * virt) {
    GGML_UNUSED(region);
    GGML_UNUSED(virt);
#ifdef __linux__
    return region->base_phys + (static_cast<char const *>(virt) - static_cast<char const *>(region->base_virt));
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Buffer context (one buffer = one allocation from shared region)
// ---------------------------------------------------------------------------

struct ggml_fpga_buffer_context {
    struct ggml_fpga_shared_region * region;
    void *   ptr;    // base pointer (into shared region)
    size_t   size;
    bool     owned;  // if true, we allocated it (bump); if false, external
};

static void ggml_fpga_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_fpga_buffer_context * ctx = static_cast<ggml_fpga_buffer_context *>(buffer->context);
    if (ctx->owned && ctx->ptr && (!ctx->region || !ctx->region->base_virt)) {
        free(ctx->ptr);  // Stub: we allocated with aligned_alloc.
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
    /* .cpy_tensor    = */ nullptr,
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
// Stub FLASH_ATTN_EXT: delegate to CPU (replace with FPGA when ready).
// Uses same layout as ggml_compute_params so CPU impl can be called when linked.
// ---------------------------------------------------------------------------

struct ggml_fpga_compute_params {
    int ith, nth;
    size_t wsize;
    void * wdata;
    struct ggml_threadpool * threadpool;
    bool use_ref;
};

extern "C" void ggml_compute_forward_flash_attn_ext(
    const struct ggml_fpga_compute_params * params,
    struct ggml_tensor * dst);

static void ggml_fpga_flash_attn_ext_stub(ggml_tensor * node) {
    static const struct ggml_fpga_compute_params params = {
        /* .ith = */ 0,
        /* .nth = */ 1,
        /* .wsize = */ 0,
        /* .wdata = */ nullptr,
        /* .threadpool = */ nullptr,
        /* .use_ref = */ false,
    };
    ggml_compute_forward_flash_attn_ext(&params, node);
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

static ggml_status ggml_backend_fpga_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_fpga_context * ctx = static_cast<ggml_backend_fpga_context *>(backend->context);
    GGML_UNUSED(ctx);

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
            ggml_fpga_flash_attn_ext_stub(node);
            continue;
        }
        GGML_ASSERT(false && "FPGA backend: unsupported op");
    }
    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i const ggml_backend_fpga_i = {
    /* .get_name           = */ ggml_backend_fpga_get_name,
    /* .free               = */ ggml_backend_fpga_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
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
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
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
        if (!q || !k || !v) {
            return false;
        }
        return ggml_is_contiguous(q) && ggml_is_contiguous(k) && ggml_is_contiguous(v) &&
               op->type == GGML_TYPE_F32;
    }
    return false;
}

static bool ggml_backend_fpga_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == ggml_backend_fpga_device_get_buffer_type(dev);
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
