let wasmModulePromise = null;

function normalizeLayout(layoutHeader) {
  if (!layoutHeader || layoutHeader.length < 13) {
    throw new Error("BitNet layout_header must contain at least 13 entries");
  }
  const header = Int32Array.from(Array.from(layoutHeader, Number));
  if (header[0] !== 1 || header[1] !== 16 || header[2] !== 32 || header[9] !== 1) {
    throw new Error("Unsupported BitNet WASM layout; expected v1 16x32 interleave mode 1");
  }
  if (header[3] <= 0 || header[4] <= 0) {
    throw new Error("BitNet logical dimensions must be positive");
  }
  if (header[5] < header[3] || header[6] < header[4]) {
    throw new Error("BitNet padded dimensions must be at least the logical dimensions");
  }
  if (header[5] % header[1] !== 0) {
    throw new Error("BitNet padded output features must be tile aligned");
  }
  if (header[6] % 4 !== 0) {
    throw new Error("BitNet padded input features must be divisible by 4");
  }
  if (header[6] % header[2] !== 0) {
    throw new Error("BitNet padded input features must be tile aligned");
  }
  if (![0, 1, 2].includes(header[7])) {
    throw new Error(`Unsupported BitNet scale_granularity: ${header[7]}`);
  }
  if (header[7] === 2 && header[8] <= 0) {
    throw new Error("BitNet per-output-group scaling requires a positive scale_group_size");
  }
  if (header[11] <= 0) {
    throw new Error("BitNet segment_count must be positive");
  }
  return {
    header,
    logicalOut: header[3],
    logicalIn: header[4],
    paddedOut: header[5],
    paddedIn: header[6],
    scaleGranularity: header[7],
    scaleGroupSize: header[8],
    segmentCount: header[11],
  };
}

function validateSegmentOffsets(layout, segmentOffsets) {
  if (!(segmentOffsets instanceof Int32Array)) {
    throw new Error("BitNet segment_offsets must use int32 storage");
  }
  if (segmentOffsets.length !== layout.segmentCount + 1) {
    throw new Error("BitNet segment_offsets length mismatch");
  }
  if (segmentOffsets[0] !== 0) {
    throw new Error("BitNet segment_offsets must start at 0");
  }
  if (segmentOffsets[layout.segmentCount] !== layout.logicalOut) {
    throw new Error("BitNet segment_offsets must end at logical_out_features");
  }
  for (let idx = 1; idx < segmentOffsets.length; idx += 1) {
    if (segmentOffsets[idx] < segmentOffsets[idx - 1]) {
      throw new Error("BitNet segment_offsets must be non-decreasing");
    }
  }
}

function validateScaleValues(layout, scaleValues) {
  if (!(scaleValues instanceof Float32Array)) {
    throw new Error("BitNet scale_values must use float32 storage");
  }
  if (layout.scaleGranularity === 0 && scaleValues.length < 1) {
    throw new Error("BitNet per-matrix scaling requires at least one value");
  }
  if (layout.scaleGranularity === 1 && scaleValues.length !== layout.segmentCount) {
    throw new Error("BitNet per-segment scaling size mismatch");
  }
  if (layout.scaleGranularity === 2) {
    const expectedGroups = Math.ceil(layout.logicalOut / layout.scaleGroupSize);
    if (scaleValues.length !== expectedGroups) {
      throw new Error("BitNet per-output-group scaling size mismatch");
    }
  }
}

function validateInputQuantization(inputQuantMode, inputQuantBits, inputScaleRows, inputScales) {
  if (inputQuantMode === 0) {
    return;
  }
  if (inputQuantBits < 2 || inputQuantBits > 8) {
    throw new Error("BitNet input_quant_bits must be in [2, 8]");
  }
  if (inputScaleRows <= 0) {
    throw new Error("BitNet input_scale_rows must be positive when input quantization is enabled");
  }
  if (!(inputScales instanceof Float32Array) || inputScales.length < inputScaleRows) {
    throw new Error("BitNet input_scales length mismatch");
  }
}

function validateLayerTensors(layout, bundle) {
  const rowStrideBytes = layout.paddedIn / 4;
  const expectedPackedBytes = layout.logicalOut * rowStrideBytes;
  if (!(bundle.packedWeight instanceof Uint8Array) || bundle.packedWeight.length < expectedPackedBytes) {
    throw new Error("BitNet packed_weight is shorter than layout requires");
  }
  validateScaleValues(layout, bundle.scaleValues);
  validateSegmentOffsets(layout, bundle.segmentOffsets);
  if (bundle.bias && bundle.bias.length > 0 && (!(bundle.bias instanceof Float32Array) || bundle.bias.length !== layout.logicalOut)) {
    throw new Error("BitNet bias length mismatch");
  }
}

function resolveUrl(path, baseUrl) {
  return new URL(path, baseUrl).toString();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchWithRetry(url, options = {}) {
  const attempts = Math.max(1, Number(options.attempts || 5));
  let lastError = null;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await fetch(url);
      if (response.ok) return response;
      if (response.status < 500 && response.status !== 408 && response.status !== 429) {
        throw new Error(`failed to fetch ${url}: ${response.status}`);
      }
      lastError = new Error(`failed to fetch ${url}: ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    if (attempt < attempts - 1) {
      await sleep(Math.min(2000, 150 * 2 ** attempt));
    }
  }
  throw lastError || new Error(`failed to fetch ${url}`);
}

async function fetchTensor(entry, baseUrl, TypedArray) {
  const url = resolveUrl(entry.path, baseUrl);
  const response = await fetchWithRetry(url);
  if (!response.ok) {
    throw new Error(`failed to fetch ${entry.path}: ${response.status}`);
  }
  return new TypedArray(await response.arrayBuffer());
}

function tensorType(entry) {
  if (entry.dtype === "uint8") return Uint8Array;
  if (entry.dtype === "int32") return Int32Array;
  if (entry.dtype === "float32") return Float32Array;
  throw new Error(`unsupported tensor dtype: ${entry.dtype}`);
}

async function ensureBitNetWasm() {
  if (!wasmModulePromise) {
    wasmModulePromise = (async () => {
      let module;
      try {
        module = await import(new URL("model_stack_bitnet_wasm.js", import.meta.url).href);
      } catch (error) {
        module = await import(new URL("pkg/model_stack_bitnet_wasm.js", import.meta.url).href);
      }
      await module.default();
      return module;
    })();
  }
  return wasmModulePromise;
}

export class BitNetLinearWASM {
  constructor(bundle) {
    this.layout = normalizeLayout(bundle.layoutHeader);
    this.packedWeight = bundle.packedWeight instanceof Uint8Array
      ? bundle.packedWeight
      : new Uint8Array(bundle.packedWeight);
    this.scaleValues = bundle.scaleValues instanceof Float32Array
      ? bundle.scaleValues
      : new Float32Array(bundle.scaleValues);
    this.segmentOffsets = bundle.segmentOffsets instanceof Int32Array
      ? bundle.segmentOffsets
      : Int32Array.from(bundle.segmentOffsets || []);
    this.bias = bundle.bias
      ? (bundle.bias instanceof Float32Array ? bundle.bias : new Float32Array(bundle.bias))
      : new Float32Array(0);
    this.inputScales = bundle.inputScales
      ? (bundle.inputScales instanceof Float32Array ? bundle.inputScales : new Float32Array(bundle.inputScales))
      : new Float32Array([1]);
    this.inputQuantMode = bundle.inputQuantMode ?? 0;
    this.inputQuantBits = bundle.inputQuantBits ?? 8;
    this.inputScaleRows = bundle.inputScaleRows ?? 1;
    validateInputQuantization(this.inputQuantMode, this.inputQuantBits, this.inputScaleRows, this.inputScales);
    validateLayerTensors(this.layout, this);
    this.wasm = bundle.wasm || null;
    this.handle = bundle.handle || null;
  }

  static async fromManifestLayer(manifest, layer, manifestUrl, options = {}) {
    const progress = typeof options.progress === "function" ? options.progress : () => {};
    const index = Number(options.index || 0);
    const total = Number(options.total || 0);
    const name = String(options.name || layer.name || "layer");
    const label = total ? `${index}/${total}: ${name}` : name;
    const baseUrl = new URL(".", manifestUrl).toString();
    const tensors = layer.tensors;
    const layersBaseUrl = resolveUrl("layers/", baseUrl);
    progress({ phase: "layer_tensors", index, total, name, message: `Loading BitNet WASM tensors ${label}` });
    const [wasm, packedWeight, scaleValues, segmentOffsets, bias, inputScales] = await Promise.all([
      ensureBitNetWasm(),
      fetchTensor(tensors.packed_weight, layersBaseUrl, Uint8Array),
      fetchTensor(tensors.scale_values, layersBaseUrl, Float32Array),
      fetchTensor(tensors.segment_offsets, layersBaseUrl, Int32Array),
      tensors.bias ? fetchTensor(tensors.bias, layersBaseUrl, Float32Array) : Promise.resolve(null),
      fetchTensor(tensors.act_scale, layersBaseUrl, tensorType(tensors.act_scale)),
    ]);
    progress({ phase: "layer_ready", index, total, name, message: `BitNet WASM layer ${label} ready` });
    const layoutHeader = layer.layout_header;
    const inputQuantMode = layer.act_quant_mode === "none" ? 0 : 1;
    const inputQuantBits = layer.act_quant_bits;
    const inputScaleRows = layer.act_quant_mode === "static_int8" ? 1 : 1;
    const handle = wasm.BitnetLinearHandle
      ? new wasm.BitnetLinearHandle(
          packedWeight,
          scaleValues,
          segmentOffsets,
          bias || new Float32Array(0),
          Int32Array.from(Array.from(layoutHeader, Number)),
          inputScales,
          inputQuantMode,
          inputQuantBits,
          inputScaleRows,
        )
      : null;
    return new BitNetLinearWASM({
      layoutHeader,
      packedWeight,
      scaleValues,
      segmentOffsets,
      bias,
      inputScales,
      inputQuantMode,
      inputQuantBits,
      inputScaleRows,
      wasm,
      handle,
    });
  }

  run(input, rows = 1) {
    const x = input instanceof Float32Array ? input : new Float32Array(input);
    if (x.length !== rows * this.layout.logicalIn) {
      throw new Error(`BitNet input length mismatch: got ${x.length}, expected ${rows * this.layout.logicalIn}`);
    }
    const wasm = this.wasm;
    if (!wasm) {
      throw new Error("BitNet WASM module is not loaded");
    }
    if (this.handle?.run) {
      return this.handle.run(x, rows);
    }
    return wasm.bitnet_linear_f32(
      x,
      this.packedWeight,
      this.scaleValues,
      this.segmentOffsets,
      this.bias,
      this.layout.header,
      this.inputScales,
      rows,
      this.inputQuantMode,
      this.inputQuantBits,
      this.inputScaleRows,
    );
  }
}
