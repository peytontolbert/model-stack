/* tslint:disable */
/* eslint-disable */

export class AttentionKvCache {
    free(): void;
    [Symbol.dispose](): void;
    append_self_attention(q: Float32Array, k_new: Float32Array, v_new: Float32Array, q_len: number, causal: boolean): Float32Array;
    attention(q: Float32Array, q_len: number, causal: boolean, past_len: number): Float32Array;
    clear(): void;
    clone_cache(): AttentionKvCache;
    len(): number;
    constructor(n_heads: number, head_dim: number);
    set_cross(k: Float32Array, v: Float32Array, kv_len: number): void;
}

export class BitnetLinearHandle {
    free(): void;
    [Symbol.dispose](): void;
    constructor(packed_weight: Uint8Array, scale_values: Float32Array, segment_offsets: Int32Array, bias_values: Float32Array, layout_header: Int32Array, input_scales: Float32Array, input_quant_mode: number, input_quant_bits: number, input_scale_rows: number);
    run(input: Float32Array, rows: number): Float32Array;
}

export class DecoderLayerHandle {
    free(): void;
    [Symbol.dispose](): void;
    clone_cache(): DecoderLayerHandle;
    constructor(self_q: BitnetLinearHandle, self_k: BitnetLinearHandle, self_v: BitnetLinearHandle, self_o: BitnetLinearHandle, self_mlp_in: BitnetLinearHandle, self_mlp_out: BitnetLinearHandle, cross_q: BitnetLinearHandle, cross_k: BitnetLinearHandle, cross_v: BitnetLinearHandle, cross_o: BitnetLinearHandle, cross_mlp_in: BitnetLinearHandle, cross_mlp_out: BitnetLinearHandle, self_n1_weight: Float32Array, self_n1_bias: Float32Array, self_n2_weight: Float32Array, self_n2_bias: Float32Array, cross_n1_weight: Float32Array, cross_n1_bias: Float32Array, cross_n2_weight: Float32Array, cross_n2_bias: Float32Array, activation: string, d_model: number, n_heads: number, head_dim: number, rotary_base: number);
    next(input: Float32Array, memory: Float32Array, memory_len: number): Float32Array;
    self_len(): number;
}

export class TokenSample {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly probability: number;
    readonly rank: number;
    readonly token_id: number;
    readonly top_probability: number;
}

export function activate_f32(input: Float32Array, activation: string): Float32Array;

export function attention_f32(q: Float32Array, k: Float32Array, v: Float32Array, q_len: number, kv_len: number, n_heads: number, head_dim: number, causal: boolean, past_len: number): Float32Array;

export function bitnet_linear2_f32(first: BitnetLinearHandle, second: BitnetLinearHandle, input: Float32Array, rows: number): Float32Array;

export function bitnet_linear3_f32(first: BitnetLinearHandle, second: BitnetLinearHandle, third: BitnetLinearHandle, input: Float32Array, rows: number): Float32Array;

export function bitnet_linear_f32(input: Float32Array, packed_weight: Uint8Array, scale_values: Float32Array, segment_offsets: Int32Array, bias_values: Float32Array, layout_header: Int32Array, input_scales: Float32Array, rows: number, input_quant_mode: number, input_quant_bits: number, input_scale_rows: number): Float32Array;

export function bitnet_mlp_f32(w_in: BitnetLinearHandle, w_out: BitnetLinearHandle, input: Float32Array, rows: number, activation: string): Float32Array;

export function bitnet_sample_token_f32(lm_head: BitnetLinearHandle, hidden: Float32Array, generated_ids: Uint32Array, blocked_ids: Uint32Array, temperature: number, top_p: number, repetition_penalty: number, random_value: number): TokenSample;

export function gated_activation_f32(input: Float32Array, rows: number, cols: number, activation: string): Float32Array;

export function layer_norm_f32(input: Float32Array, weight: Float32Array, bias: Float32Array, rows: number, cols: number, eps: number): Float32Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly bitnet_linear_f32: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number) => [number, number, number, number];
    readonly __wbg_bitnetlinearhandle_free: (a: number, b: number) => void;
    readonly __wbg_decoderlayerhandle_free: (a: number, b: number) => void;
    readonly decoderlayerhandle_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: number, v: number, w: number, x: number, y: number, z: number, a1: number, b1: number, c1: number, d1: number, e1: number, f1: number, g1: number, h1: number) => number;
    readonly decoderlayerhandle_self_len: (a: number) => number;
    readonly decoderlayerhandle_clone_cache: (a: number) => number;
    readonly decoderlayerhandle_next: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly bitnetlinearhandle_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number) => [number, number, number];
    readonly bitnetlinearhandle_run: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly __wbg_tokensample_free: (a: number, b: number) => void;
    readonly tokensample_token_id: (a: number) => number;
    readonly tokensample_probability: (a: number) => number;
    readonly tokensample_top_probability: (a: number) => number;
    readonly tokensample_rank: (a: number) => number;
    readonly bitnet_sample_token_f32: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number, number];
    readonly bitnet_linear2_f32: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly bitnet_linear3_f32: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly bitnet_mlp_f32: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number, number];
    readonly layer_norm_f32: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number, number];
    readonly attention_f32: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number) => [number, number, number, number];
    readonly __wbg_attentionkvcache_free: (a: number, b: number) => void;
    readonly attentionkvcache_new: (a: number, b: number) => number;
    readonly attentionkvcache_len: (a: number) => number;
    readonly attentionkvcache_clear: (a: number) => void;
    readonly attentionkvcache_clone_cache: (a: number) => number;
    readonly attentionkvcache_set_cross: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly attentionkvcache_append_self_attention: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number, number];
    readonly attentionkvcache_attention: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly activate_f32: (a: number, b: number, c: number, d: number) => [number, number];
    readonly gated_activation_f32: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
