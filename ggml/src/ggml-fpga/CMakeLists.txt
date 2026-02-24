# FPGA GEMM backend (shared-memory + UIO stub).
# Enable with -DGGML_FPGA=ON.

ggml_add_backend_library(ggml-fpga
    ggml-fpga.cpp
)

target_link_libraries(ggml-fpga PRIVATE ggml-base)
target_include_directories(ggml-fpga PRIVATE ..)
