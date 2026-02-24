#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_FPGA_NAME "FPGA"

// Backend API (dev is passed by the registry when initializing from device)
GGML_BACKEND_API ggml_backend_t ggml_backend_fpga_init(ggml_backend_dev_t dev);
GGML_BACKEND_API bool ggml_backend_is_fpga(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_fpga_reg(void);

#ifdef __cplusplus
}
#endif
