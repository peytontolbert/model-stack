use wasm_bindgen::prelude::*;

const HEADER_LEN: usize = 13;
const IDX_FORMAT_VERSION: usize = 0;
const IDX_TILE_N: usize = 1;
const IDX_TILE_K: usize = 2;
const IDX_LOGICAL_OUT: usize = 3;
const IDX_LOGICAL_IN: usize = 4;
const IDX_PADDED_IN: usize = 6;
const IDX_SCALE_GRANULARITY: usize = 7;
const IDX_SCALE_GROUP_SIZE: usize = 8;
const IDX_INTERLEAVE_MODE: usize = 9;
const IDX_SEGMENT_COUNT: usize = 11;
const OUT_TILE: usize = 8;

fn validate_header(layout_header: &[i32]) -> Result<(), JsValue> {
    if layout_header.len() < HEADER_LEN {
        return Err(JsValue::from_str("BitNet layout_header must contain at least 13 entries"));
    }
    if layout_header[IDX_FORMAT_VERSION] != 1
        || layout_header[IDX_TILE_N] != 16
        || layout_header[IDX_TILE_K] != 32
        || layout_header[IDX_INTERLEAVE_MODE] != 1
    {
        return Err(JsValue::from_str(
            "Unsupported BitNet WASM layout; expected v1 16x32 interleave mode 1",
        ));
    }
    Ok(())
}

fn resolve_scale(
    out_idx: usize,
    scale_values: &[f32],
    segment_offsets: &[i32],
    scale_granularity: usize,
    scale_group_size: usize,
    segment_count: usize,
) -> f32 {
    match scale_granularity {
        0 => scale_values.first().copied().unwrap_or(0.0),
        1 => {
            for seg in 0..segment_count {
                let start = segment_offsets.get(seg).copied().unwrap_or(0).max(0) as usize;
                let end = segment_offsets.get(seg + 1).copied().unwrap_or(0).max(0) as usize;
                if out_idx >= start && out_idx < end {
                    return scale_values.get(seg).copied().unwrap_or(0.0);
                }
            }
            0.0
        }
        2 => {
            if scale_group_size == 0 {
                0.0
            } else {
                scale_values.get(out_idx / scale_group_size).copied().unwrap_or(0.0)
            }
        }
        _ => 0.0,
    }
}

fn quant_max(bits: u32) -> f32 {
    if bits == 0 || bits >= 31 {
        return 127.0;
    }
    ((1u32 << (bits - 1)) - 1) as f32
}

fn input_value(
    input: &[f32],
    row: usize,
    col: usize,
    logical_in: usize,
    input_scales: &[f32],
    input_quant_mode: u32,
    input_quant_bits: u32,
    input_scale_rows: usize,
) -> f32 {
    let value = input[row * logical_in + col];
    if input_quant_mode == 0 {
        return value;
    }
    let scale_row = if input_scale_rows == 1 { 0 } else { row };
    let scale = input_scales.get(scale_row).copied().unwrap_or(1.0).max(1e-8);
    let qmax = quant_max(input_quant_bits);
    (value / scale).round().clamp(-qmax, qmax) * scale
}

fn decode_signed_ternary_code(code: u8) -> f32 {
    match code & 3 {
        0 => -1.0,
        2 => 1.0,
        _ => 0.0,
    }
}

fn dot_packed_row_quantized(
    input: &[f32],
    row: usize,
    logical_in: usize,
    packed_row: &[u8],
    input_scales: &[f32],
    input_quant_mode: u32,
    input_quant_bits: u32,
    input_scale_rows: usize,
) -> f32 {
    let mut acc = 0.0f32;
    for (packed_col, packed_byte) in packed_row.iter().enumerate() {
        let base_col = packed_col * 4;
        if base_col >= logical_in {
            break;
        }
        let byte = *packed_byte;
        for lane in 0..4 {
            let col = base_col + lane;
            if col >= logical_in {
                break;
            }
            let w = decode_signed_ternary_code(byte >> (lane * 2));
            if w != 0.0 {
                acc += input_value(
                    input,
                    row,
                    col,
                    logical_in,
                    input_scales,
                    input_quant_mode,
                    input_quant_bits,
                    input_scale_rows,
                ) * w;
            }
        }
    }
    acc
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn dot_packed_row_noquant_simd(input_row: &[f32], packed_row: &[u8], logical_in: usize) -> f32 {
    use core::arch::wasm32::*;

    let packed_cols = logical_in / 4;
    let mut sum = f32x4_splat(0.0);
    for packed_col in 0..packed_cols {
        let byte = *packed_row.get_unchecked(packed_col);
        let weights = f32x4(
            decode_signed_ternary_code(byte),
            decode_signed_ternary_code(byte >> 2),
            decode_signed_ternary_code(byte >> 4),
            decode_signed_ternary_code(byte >> 6),
        );
        let x = v128_load(input_row.as_ptr().add(packed_col * 4) as *const v128);
        sum = f32x4_add(sum, f32x4_mul(x, weights));
    }

    let mut acc = f32x4_extract_lane::<0>(sum)
        + f32x4_extract_lane::<1>(sum)
        + f32x4_extract_lane::<2>(sum)
        + f32x4_extract_lane::<3>(sum);
    let base_col = packed_cols * 4;
    if base_col < logical_in {
        let byte = *packed_row.get_unchecked(packed_cols);
        for lane in 0..(logical_in - base_col) {
            acc += input_row[base_col + lane] * decode_signed_ternary_code(byte >> (lane * 2));
        }
    }
    acc
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn dot_packed_row_noquant(input_row: &[f32], packed_row: &[u8], logical_in: usize) -> f32 {
    unsafe { dot_packed_row_noquant_simd(input_row, packed_row, logical_in) }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn dot_packed_row_noquant(input_row: &[f32], packed_row: &[u8], logical_in: usize) -> f32 {
    let mut acc = 0.0f32;
    for (packed_col, packed_byte) in packed_row.iter().enumerate() {
        let base_col = packed_col * 4;
        if base_col >= logical_in {
            break;
        }
        let byte = *packed_byte;
        let remaining = (logical_in - base_col).min(4);
        if remaining > 0 {
            acc += input_row[base_col] * decode_signed_ternary_code(byte);
        }
        if remaining > 1 {
            acc += input_row[base_col + 1] * decode_signed_ternary_code(byte >> 2);
        }
        if remaining > 2 {
            acc += input_row[base_col + 2] * decode_signed_ternary_code(byte >> 4);
        }
        if remaining > 3 {
            acc += input_row[base_col + 3] * decode_signed_ternary_code(byte >> 6);
        }
    }
    acc
}

#[wasm_bindgen]
pub fn bitnet_linear_f32(
    input: &[f32],
    packed_weight: &[u8],
    scale_values: &[f32],
    segment_offsets: &[i32],
    bias_values: &[f32],
    layout_header: &[i32],
    input_scales: &[f32],
    rows: usize,
    input_quant_mode: u32,
    input_quant_bits: u32,
    input_scale_rows: usize,
) -> Result<Vec<f32>, JsValue> {
    validate_header(layout_header)?;
    let logical_out = layout_header[IDX_LOGICAL_OUT].max(0) as usize;
    let logical_in = layout_header[IDX_LOGICAL_IN].max(0) as usize;
    let padded_in = layout_header[IDX_PADDED_IN].max(0) as usize;
    let scale_granularity = layout_header[IDX_SCALE_GRANULARITY].max(0) as usize;
    let scale_group_size = layout_header[IDX_SCALE_GROUP_SIZE].max(0) as usize;
    let segment_count = layout_header[IDX_SEGMENT_COUNT].max(0) as usize;
    if logical_in == 0 || logical_out == 0 || rows == 0 {
        return Ok(Vec::new());
    }
    if input.len() != rows * logical_in {
        return Err(JsValue::from_str("BitNet input length does not match rows * logical_in"));
    }
    let row_stride_bytes = padded_in / 4;
    if packed_weight.len() < logical_out * row_stride_bytes {
        return Err(JsValue::from_str("BitNet packed_weight is shorter than layout requires"));
    }

    let mut output = vec![0.0f32; rows * logical_out];
    let packed_cols = logical_in.div_ceil(4);
    for row in 0..rows {
        let input_row = &input[row * logical_in..(row + 1) * logical_in];
        for out_base in (0..logical_out).step_by(OUT_TILE) {
            let out_end = (out_base + OUT_TILE).min(logical_out);
            for out_idx in out_base..out_end {
                let row_base = out_idx * row_stride_bytes;
                let packed_row = &packed_weight[row_base..row_base + packed_cols];
                let acc = if input_quant_mode == 0 {
                    dot_packed_row_noquant(input_row, packed_row, logical_in)
                } else {
                    dot_packed_row_quantized(
                        input,
                        row,
                        logical_in,
                        packed_row,
                        input_scales,
                        input_quant_mode,
                        input_quant_bits,
                        input_scale_rows,
                    )
                };
                let scale = resolve_scale(
                    out_idx,
                    scale_values,
                    segment_offsets,
                    scale_granularity,
                    scale_group_size,
                    segment_count,
                );
                let bias = bias_values.get(out_idx).copied().unwrap_or(0.0);
                output[row * logical_out + out_idx] = acc * scale + bias;
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_tiny_linear() {
        let input = [2.0f32, 3.0, 5.0, 7.0];
        // Codes per row: [-1, 0, +1, 0].
        let mut packed_weight = [0u8; 8];
        packed_weight[0] = 0b01_10_01_00u8;
        let scales = [0.5f32];
        let offsets = [0i32, 1];
        let header = [1, 16, 32, 1, 4, 16, 32, 0, 1, 1, 0, 1, 0];
        let y = bitnet_linear_f32(
            &input,
            &packed_weight,
            &scales,
            &offsets,
            &[],
            &header,
            &[1.0],
            1,
            0,
            8,
            1,
        )
        .unwrap();
        assert_eq!(y.len(), 1);
        assert!((y[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn decodes_two_output_rows_with_tile_loop() {
        let input = [2.0f32, 3.0, 5.0, 7.0];
        let mut packed_weight = [0u8; 16];
        packed_weight[0] = 0b01_10_01_00u8; // -2 + 5 = 3
        packed_weight[8] = 0b10_00_10_01u8; // 0 + 3 - 5 + 7 = 5
        let scales = [1.0f32, 2.0];
        let offsets = [0i32, 1, 2];
        let header = [1, 16, 32, 2, 4, 16, 32, 1, 1, 1, 0, 2, 0];
        let y = bitnet_linear_f32(
            &input,
            &packed_weight,
            &scales,
            &offsets,
            &[0.0, 1.0],
            &header,
            &[1.0],
            1,
            0,
            8,
            1,
        )
        .unwrap();
        assert_eq!(y.len(), 2);
        assert!((y[0] - 3.0).abs() < 1e-6);
        assert!((y[1] - 11.0).abs() < 1e-6);
    }
}
