// HLS kernel for Flash Attention, for use with ggml-fpga.
// Basic implementation: single head, no mask. Streams Q, K, V; outputs attention O = softmax(Q K^T * scale) V.

#include "hls_stream.h"
#include <cmath>

// Max sizes for on-chip buffers (tune for your part). Larger sizes need more BRAM.
#ifndef FLASH_ATTN_MAX_SEQ_KV
#define FLASH_ATTN_MAX_SEQ_KV 64
#endif
#ifndef FLASH_ATTN_MAX_DK
#define FLASH_ATTN_MAX_DK 128
#endif
#ifndef FLASH_ATTN_MAX_DV
#define FLASH_ATTN_MAX_DV 128
#endif

// Dot product of two vectors of length n.
static float dot(const float* a, const float* b, int n) {
#pragma HLS INLINE
    float sum = 0.f;
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        sum += a[i] * b[i];
    }
    return sum;
}

// Online softmax over `scores[0..n)`: find max, then compute exp(x-max) and sum, then normalize.
// Writes probabilities into scores and returns sum (for optional checks).
static float softmax_inplace(float* scores, int n) {
#pragma HLS INLINE off
    float row_max = -1e9f;
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        if (scores[i] > row_max) row_max = scores[i];
    }
    float sum = 0.f;
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        float e = expf(scores[i] - row_max);
        scores[i] = e;
        sum += e;
    }
    float inv_sum = (sum > 0.f) ? (1.f / sum) : 0.f;
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        scores[i] *= inv_sum;
    }
    return sum;
}

// Top-level flash attention kernel.
// Control (s_axilite): seq_len_q, seq_len_kv, dk, dv, scale.
// Stream order: all Q (seq_len_q * dk), then all K (seq_len_kv * dk), then all V (seq_len_kv * dv).
// Output: seq_len_q * dv in row-major order.
void flash_attn_kernel(
    hls::stream<float>& q_in,
    hls::stream<float>& k_in,
    hls::stream<float>& v_in,
    hls::stream<float>& out,
    int seq_len_q,
    int seq_len_kv,
    int dk,
    int dv,
    float scale
)
{
#pragma HLS INTERFACE mode=axis port=q_in
#pragma HLS INTERFACE mode=axis port=k_in
#pragma HLS INTERFACE mode=axis port=v_in
#pragma HLS INTERFACE mode=axis port=out
#pragma HLS INTERFACE mode=s_axilite port=seq_len_q bundle=control
#pragma HLS INTERFACE mode=s_axilite port=seq_len_kv bundle=control
#pragma HLS INTERFACE mode=s_axilite port=dk bundle=control
#pragma HLS INTERFACE mode=s_axilite port=dv bundle=control
#pragma HLS INTERFACE mode=s_axilite port=scale bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // On-chip buffers for K and V (one full seq_kv of keys/values).
    static float K_buf[FLASH_ATTN_MAX_SEQ_KV][FLASH_ATTN_MAX_DK];
#pragma HLS ARRAY_PARTITION variable=K_buf dim=2 cyclic factor=4
    static float V_buf[FLASH_ATTN_MAX_SEQ_KV][FLASH_ATTN_MAX_DV];
#pragma HLS ARRAY_PARTITION variable=V_buf dim=2 cyclic factor=4

    // Load full K [seq_len_kv][dk]
    for (int j = 0; j < seq_len_kv; j++) {
        for (int d = 0; d < dk; d++) {
#pragma HLS PIPELINE II=1
            float v;
            k_in.read(v);
            K_buf[j][d] = v;
        }
    }
    // Load full V [seq_len_kv][dv]
    for (int j = 0; j < seq_len_kv; j++) {
        for (int d = 0; d < dv; d++) {
#pragma HLS PIPELINE II=1
            float v;
            v_in.read(v);
            V_buf[j][d] = v;
        }
    }

    // Scratch for one Q row, scores, and one output row
    float q_row[FLASH_ATTN_MAX_DK];
#pragma HLS ARRAY_PARTITION variable=q_row cyclic factor=8
    float scores[FLASH_ATTN_MAX_SEQ_KV];
#pragma HLS ARRAY_PARTITION variable=scores cyclic factor=4
    float out_row[FLASH_ATTN_MAX_DV];
#pragma HLS ARRAY_PARTITION variable=out_row cyclic factor=8

    for (int i = 0; i < seq_len_q; i++) {
        // Read Q row
        for (int d = 0; d < dk; d++) {
#pragma HLS PIPELINE II=1
            float v;
            q_in.read(v);
            q_row[d] = v;
        }

        // Scores = Q[i] @ K^T * scale
        for (int j = 0; j < seq_len_kv; j++) {
#pragma HLS PIPELINE II=1
            scores[j] = scale * dot(q_row, K_buf[j], dk);
        }

        // Softmax in place
        softmax_inplace(scores, seq_len_kv);

        // out_row = scores @ V
        for (int d = 0; d < dv; d++) {
#pragma HLS PIPELINE II=1
            float acc = 0.f;
            for (int j = 0; j < seq_len_kv; j++) {
                acc += scores[j] * V_buf[j][d];
            }
            out_row[d] = acc;
        }

        // Write output row
        for (int d = 0; d < dv; d++) {
#pragma HLS PIPELINE II=1
            out.write(out_row[d]);
        }
    }
}
