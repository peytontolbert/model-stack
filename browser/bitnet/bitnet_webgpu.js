const PARAM_U32_COUNT = 12;
const PARAM_BUFFER_BYTES = PARAM_U32_COUNT * 4;
const shaderTextCache = new Map();
const pipelineCache = new WeakMap();

function align4(value) {
  return (value + 3) & ~3;
}

function packedWeightToWords(packedWeight) {
  const bytes = packedWeight instanceof Uint8Array ? packedWeight : new Uint8Array(packedWeight);
  const padded = new Uint8Array(align4(bytes.byteLength));
  padded.set(bytes);
  return new Uint32Array(padded.buffer);
}

function createStorageBuffer(device, data, usage = GPUBufferUsage.STORAGE) {
  const source = ArrayBuffer.isView(data) ? data : new Uint8Array(data);
  const buffer = device.createBuffer({
    size: align4(source.byteLength),
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, source.buffer, source.byteOffset, source.byteLength);
  return buffer;
}

function createOutputBuffer(device, byteLength) {
  return device.createBuffer({
    size: align4(byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
}

function createReadbackBuffer(device, byteLength) {
  return device.createBuffer({
    size: align4(byteLength),
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
}

function normalizeLayout(layoutHeader) {
  if (!layoutHeader || layoutHeader.length < 13) {
    throw new Error("BitNet layout_header must contain at least 13 entries");
  }
  const header = Array.from(layoutHeader, Number);
  if (header[0] !== 1 || header[1] !== 16 || header[2] !== 32 || header[9] !== 1) {
    throw new Error("Unsupported BitNet browser layout; expected v1 16x32 interleave mode 1");
  }
  if (header[3] <= 0 || header[4] <= 0) {
    throw new Error("BitNet logical dimensions must be positive");
  }
  if (header[5] < header[3] || header[6] < header[4]) {
    throw new Error("BitNet padded dimensions must be greater than or equal to logical dimensions");
  }
  if (header[6] % 4 !== 0) {
    throw new Error("BitNet padded input features must be divisible by 4");
  }
  if (![0, 1, 2].includes(header[7])) {
    throw new Error("Unsupported BitNet scale_granularity");
  }
  if (header[7] === 1 && header[11] <= 0) {
    throw new Error("BitNet per-segment scaling requires segment_count > 0");
  }
  if (header[7] === 2 && header[8] <= 0) {
    throw new Error("BitNet per-output-group scaling requires scale_group_size > 0");
  }
  return {
    logicalOut: header[3],
    logicalIn: header[4],
    paddedOut: header[5],
    paddedIn: header[6],
    scaleGranularity: header[7],
    scaleGroupSize: header[8],
    segmentCount: header[11],
  };
}

function validateLayerTensors(layout, bundle) {
  const rowStrideBytes = layout.paddedIn / 4;
  const expectedPackedBytes = layout.logicalOut * rowStrideBytes;
  if (bundle.packedWeight.length < expectedPackedBytes) {
    throw new Error("BitNet packed_weight is shorter than layout requires");
  }
  if (bundle.bias && bundle.bias.length > 0 && bundle.bias.length < layout.logicalOut) {
    throw new Error("BitNet bias length must be zero or at least logicalOut");
  }
  if (layout.scaleGranularity === 0 && bundle.scaleValues.length < 1) {
    throw new Error("BitNet per-matrix scaling requires one scale value");
  }
  if (layout.scaleGranularity === 1) {
    if (bundle.scaleValues.length < layout.segmentCount) {
      throw new Error("BitNet per-segment scaling has too few scale values");
    }
    if (bundle.segmentOffsets.length < layout.segmentCount + 1) {
      throw new Error("BitNet per-segment scaling has too few segment offsets");
    }
    if (bundle.segmentOffsets[0] !== 0 || bundle.segmentOffsets[layout.segmentCount] !== layout.logicalOut) {
      throw new Error("BitNet segment offsets must span [0, logicalOut]");
    }
    for (let i = 0; i < layout.segmentCount; i += 1) {
      if (bundle.segmentOffsets[i] > bundle.segmentOffsets[i + 1]) {
        throw new Error("BitNet segment offsets must be non-decreasing");
      }
    }
  }
  if (layout.scaleGranularity === 2) {
    const expectedGroups = Math.ceil(layout.logicalOut / layout.scaleGroupSize);
    if (bundle.scaleValues.length < expectedGroups) {
      throw new Error("BitNet per-output-group scaling has too few scale values");
    }
  }
}

function validateInputQuantization(inputScales, rows, inputQuantMode, inputQuantBits, inputScaleRows) {
  if (inputQuantMode === 0) return;
  if (inputQuantMode !== 1) {
    throw new Error("Unsupported BitNet input_quant_mode");
  }
  if (inputQuantBits < 2 || inputQuantBits > 8) {
    throw new Error("BitNet input_quant_bits must be in [2, 8]");
  }
  if (inputScaleRows !== 1 && inputScaleRows !== rows) {
    throw new Error("BitNet input_scale_rows must be 1 or rows");
  }
  if (inputScales.length < inputScaleRows) {
    throw new Error("BitNet input_scales has too few values");
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

async function fetchJson(url) {
  const response = await fetchWithRetry(url);
  if (!response.ok) {
    throw new Error(`failed to fetch ${url}: ${response.status}`);
  }
  return response.json();
}

async function fetchText(url) {
  const response = await fetchWithRetry(url);
  if (!response.ok) {
    throw new Error(`failed to fetch ${url}: ${response.status}`);
  }
  return response.text();
}

async function fetchTextCached(url) {
  if (!shaderTextCache.has(url)) {
    shaderTextCache.set(url, fetchText(url));
  }
  return shaderTextCache.get(url);
}

async function getBitNetPipeline(device, shaderCode, cacheKey) {
  let deviceCache = pipelineCache.get(device);
  if (!deviceCache) {
    deviceCache = new Map();
    pipelineCache.set(device, deviceCache);
  }
  if (!deviceCache.has(cacheKey)) {
    deviceCache.set(cacheKey, (async () => {
      const module = device.createShaderModule({ code: shaderCode });
      const descriptor = {
        layout: "auto",
        compute: { module, entryPoint: "bitnet_linear_main" },
      };
      const pipeline = typeof device.createComputePipelineAsync === "function"
        ? await device.createComputePipelineAsync(descriptor)
        : device.createComputePipeline(descriptor);
      return { module, pipeline };
    })());
  }
  return deviceCache.get(cacheKey);
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
  if (entry.dtype === "uint8") {
    return Uint8Array;
  }
  if (entry.dtype === "int32") {
    return Int32Array;
  }
  if (entry.dtype === "float32") {
    return Float32Array;
  }
  throw new Error(`unsupported tensor dtype: ${entry.dtype}`);
}

export async function createBitNetWebGPUDevice() {
  if (!globalThis.navigator?.gpu) {
    throw new Error("WebGPU is not available in this browser");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("WebGPU adapter request failed");
  }
  const device = await adapter.requestDevice();
  return { adapter, device };
}

export class BitNetLinearWebGPU {
  constructor(device, bundle) {
    this.device = device;
    this.layout = normalizeLayout(bundle.layoutHeader);
    this.inputQuantMode = bundle.inputQuantMode ?? 0;
    this.inputQuantBits = bundle.inputQuantBits ?? 8;
    this.inputScaleRows = bundle.inputScaleRows ?? 1;
    this.inputScales = bundle.inputScales ? new Float32Array(bundle.inputScales) : new Float32Array([1]);

    if (!bundle.shaderCode && !bundle.pipeline) {
      throw new Error("BitNetLinearWebGPU requires shaderCode or pipeline; use fromManifestLayer() or fromManifestUrl()");
    }
    if (bundle.pipeline) {
      this.module = bundle.module || null;
      this.pipeline = bundle.pipeline;
    } else {
      this.module = device.createShaderModule({ code: bundle.shaderCode });
      this.pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: this.module, entryPoint: "bitnet_linear_main" },
      });
    }

    const packedWeight = bundle.packedWeight instanceof Uint8Array
      ? bundle.packedWeight
      : new Uint8Array(bundle.packedWeight);
    const scaleValues = bundle.scaleValues instanceof Float32Array
      ? bundle.scaleValues
      : new Float32Array(bundle.scaleValues);
    const segmentOffsets = bundle.segmentOffsets instanceof Int32Array
      ? bundle.segmentOffsets
      : Int32Array.from(bundle.segmentOffsets || []);
    const hasBiasInput = bundle.bias != null;
    const bias = hasBiasInput
      ? (bundle.bias instanceof Float32Array ? bundle.bias : new Float32Array(bundle.bias))
      : new Float32Array(0);
    this.hasBias = bias.length > 0;
    validateLayerTensors(this.layout, {
      packedWeight,
      scaleValues,
      segmentOffsets,
      bias,
    });

    this.packedWeightBuffer = createStorageBuffer(device, packedWeightToWords(packedWeight));
    this.scaleBuffer = createStorageBuffer(device, scaleValues);
    this.segmentOffsetBuffer = createStorageBuffer(device, new Uint32Array(segmentOffsets));
    this.biasBuffer = createStorageBuffer(
      device,
      this.hasBias ? bias : new Float32Array([0]),
    );
    this.inputScaleBuffer = createStorageBuffer(device, this.inputScales);
    this.paramsBuffer = device.createBuffer({
      size: PARAM_BUFFER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.runCache = new Map();
  }

  static async fromManifestLayer(device, manifest, layer, manifestUrl, options = {}) {
    const progress = typeof options.progress === "function" ? options.progress : () => {};
    const index = Number(options.index || 0);
    const total = Number(options.total || 0);
    const name = String(options.name || layer.name || "layer");
    const label = total ? `${index}/${total}: ${name}` : name;
    const baseUrl = new URL(".", manifestUrl).toString();
    const shaderUrl = resolveUrl(manifest.runtime.files.wgsl, baseUrl);
    const runtimeBaseUrl = resolveUrl(".", shaderUrl);
    progress({ phase: "layer_shader", index, total, name, message: `Loading shader for BitNet layer ${label}` });
    const shaderCode = options.shaderCode || await fetchTextCached(shaderUrl);
    progress({ phase: "layer_pipeline", index, total, name, message: `Preparing WebGPU pipeline for BitNet layer ${label}` });
    const pipelineBundle = options.pipeline
      ? { module: options.module || null, pipeline: options.pipeline }
      : await getBitNetPipeline(device, shaderCode, shaderUrl);
    const tensors = layer.tensors;
    const layersBaseUrl = resolveUrl("layers/", baseUrl);
    progress({ phase: "layer_tensors", index, total, name, message: `Loading tensors for BitNet layer ${label}` });
    const [packedWeight, scaleValues, segmentOffsets, bias, inputScales] = await Promise.all([
      fetchTensor(tensors.packed_weight, layersBaseUrl, Uint8Array),
      fetchTensor(tensors.scale_values, layersBaseUrl, Float32Array),
      fetchTensor(tensors.segment_offsets, layersBaseUrl, Int32Array),
      tensors.bias ? fetchTensor(tensors.bias, layersBaseUrl, Float32Array) : Promise.resolve(null),
      fetchTensor(tensors.act_scale, layersBaseUrl, tensorType(tensors.act_scale)),
    ]);
    progress({ phase: "layer_upload", index, total, name, message: `Uploading BitNet layer ${label}` });
    return new BitNetLinearWebGPU(device, {
      shaderCode,
      module: pipelineBundle.module,
      pipeline: pipelineBundle.pipeline,
      layoutHeader: layer.layout_header,
      packedWeight,
      scaleValues,
      segmentOffsets,
      bias,
      inputScales,
      inputQuantMode: layer.act_quant_mode === "none" ? 0 : 1,
      inputQuantBits: layer.act_quant_bits,
      inputScaleRows: layer.act_quant_mode === "static_int8" ? 1 : 1,
      runtimeBaseUrl,
    });
  }

  static async fromManifestUrl(device, manifestUrl, layerName) {
    const manifest = await fetchJson(manifestUrl);
    const layer = manifest.layers.find((candidate) => candidate.name === layerName);
    if (!layer) {
      throw new Error(`BitNet layer not found in manifest: ${layerName}`);
    }
    return BitNetLinearWebGPU.fromManifestLayer(device, manifest, layer, manifestUrl);
  }

  async run(input, rows = 1) {
    const x = input instanceof Float32Array ? input : new Float32Array(input);
    if (x.length !== rows * this.layout.logicalIn) {
      throw new Error(`BitNet input length mismatch: got ${x.length}, expected ${rows * this.layout.logicalIn}`);
    }
    validateInputQuantization(
      this.inputScales,
      rows,
      this.inputQuantMode,
      this.inputQuantBits,
      this.inputScaleRows,
    );

    const outputLength = rows * this.layout.logicalOut;
    const inputBytes = x.byteLength;
    const outputBytes = outputLength * Float32Array.BYTES_PER_ELEMENT;
    const cacheKey = `${rows}:${this.layout.logicalIn}:${this.layout.logicalOut}`;
    let cache = this.runCache.get(cacheKey);
    if (!cache) {
      const inputBuffer = this.device.createBuffer({
        size: align4(inputBytes),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const outputBuffer = createOutputBuffer(this.device, outputBytes);
      const readbackBuffer = createReadbackBuffer(this.device, outputBytes);
      const bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: this.packedWeightBuffer } },
          { binding: 2, resource: { buffer: this.scaleBuffer } },
          { binding: 3, resource: { buffer: this.segmentOffsetBuffer } },
          { binding: 4, resource: { buffer: this.biasBuffer } },
          { binding: 5, resource: { buffer: this.inputScaleBuffer } },
          { binding: 6, resource: { buffer: outputBuffer } },
          { binding: 7, resource: { buffer: this.paramsBuffer } },
        ],
      });
      cache = { inputBuffer, outputBuffer, readbackBuffer, bindGroup };
      this.runCache.set(cacheKey, cache);
    }
    this.device.queue.writeBuffer(cache.inputBuffer, 0, x.buffer, x.byteOffset, x.byteLength);

    const params = new Uint32Array([
      rows,
      this.layout.logicalIn,
      this.layout.logicalOut,
      this.layout.paddedIn,
      this.layout.scaleGranularity,
      this.layout.scaleGroupSize,
      this.layout.segmentCount,
      this.hasBias ? 1 : 0,
      this.inputQuantMode,
      this.inputQuantBits,
      this.inputScaleRows,
      0,
    ]);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, params);

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, cache.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.layout.logicalOut / 8), Math.ceil(rows / 8), 1);
    pass.end();
    encoder.copyBufferToBuffer(cache.outputBuffer, 0, cache.readbackBuffer, 0, outputBytes);
    this.device.queue.submit([encoder.finish()]);

    await cache.readbackBuffer.mapAsync(GPUMapMode.READ);
    const mapped = cache.readbackBuffer.getMappedRange();
    const result = new Float32Array(mapped.slice(0));
    cache.readbackBuffer.unmap();
    return result;
  }
}
