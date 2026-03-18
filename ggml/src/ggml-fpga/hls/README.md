# HLS Flash Attention Kernel

Flash Attention HLS kernel for the FPGA backend in `ggml-fpga.cpp`.

- **`flash_attn_kernel.cpp`** – Single-head, no-mask implementation: `O = softmax(Q K^T * scale) V`. Build with Vitis HLS or Vivado HLS and connect the IP to the ggml-fpga data path (shared region / DMA).

**Control (s_axilite):** `seq_len_q`, `seq_len_kv`, `dk`, `dv`, `scale`.

**Stream order:**  
1. All Q: `seq_len_q * dk` floats (row-major).  
2. All K: `seq_len_kv * dk` floats (row-major).  
3. All V: `seq_len_kv * dv` floats (row-major).

**Output:** `seq_len_q * dv` floats (row-major), one row per Q position.

**Limits:** Compile-time caps `FLASH_ATTN_MAX_SEQ_KV` (default 64), `FLASH_ATTN_MAX_DK` (128), `FLASH_ATTN_MAX_DV` (128). Increase only if your part has enough BRAM.
