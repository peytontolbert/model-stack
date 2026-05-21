#include <metal_stdlib>
using namespace metal;

struct Q4LinearKernelConstants {
    uint rows;
    uint in_dim;
    uint out_dim;
    uint packed_cols;
};

static inline float decode_q4_symmetric(uchar packed, uint lane) {
    uchar nibble = lane == 0 ? (packed & 0x0f) : (packed >> 4);
    int signed_value = nibble >= 8 ? int(nibble) - 16 : int(nibble);
    return float(signed_value);
}

kernel void q4_linear_f32_kernel(
    device const float* input [[buffer(0)]],
    device const uchar* packed_weight [[buffer(1)]],
    device const float* row_scales [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant Q4LinearKernelConstants& c [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint out_col = gid.x;
    const uint row = gid.y;
    if (row >= c.rows || out_col >= c.out_dim) {
        return;
    }

    const uint input_base = row * c.in_dim;
    const uint weight_base = out_col * c.packed_cols;
    float acc = 0.0f;

    uint col = 0;
    for (; col + 1 < c.in_dim; col += 2) {
        const uchar packed = packed_weight[weight_base + (col >> 1)];
        acc += input[input_base + col] * decode_q4_symmetric(packed, 0);
        acc += input[input_base + col + 1] * decode_q4_symmetric(packed, 1);
    }
    if (col < c.in_dim) {
        const uchar packed = packed_weight[weight_base + (col >> 1)];
        acc += input[input_base + col] * decode_q4_symmetric(packed, 0);
    }

    output[row * c.out_dim + out_col] = acc * row_scales[out_col] + bias[out_col];
}
