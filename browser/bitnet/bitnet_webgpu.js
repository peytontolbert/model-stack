const PARAM_U32_COUNT = 12;
const PARAM_BUFFER_BYTES = PARAM_U32_COUNT * 4;

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

function resolveUrl(path, baseUrl) {
  return new URL(path, baseUrl).toString();
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to fetch ${url}: ${response.status}`);
  }
  return response.json();
}

async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to fetch ${url}: ${response.status}`);
  }
  return response.text();
}

async function fetchTensor(entry, baseUrl, TypedArray) {
  const response = await fetch(resolveUrl(entry.path, baseUrl));
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
    this.hasBias = bundle.bias != null;
    this.inputQuantMode = bundle.inputQuantMode ?? 0;
    this.inputQuantBits = bundle.inputQuantBits ?? 8;
    this.inputScaleRows = bundle.inputScaleRows ?? 1;

    if (!bundle.shaderCode) {
      throw new Error("BitNetLinearWebGPU requires shaderCode; use fromManifestLayer() or fromManifestUrl()");
    }
    this.module = device.createShaderModule({ code: bundle.shaderCode });
    this.pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: this.module, entryPoint: "bitnet_linear_main" },
    });

    this.packedWeightBuffer = createStorageBuffer(device, packedWeightToWords(bundle.packedWeight));
    this.scaleBuffer = createStorageBuffer(device, new Float32Array(bundle.scaleValues));
    this.segmentOffsetBuffer = createStorageBuffer(device, new Uint32Array(bundle.segmentOffsets));
    this.biasBuffer = createStorageBuffer(
      device,
      this.hasBias ? new Float32Array(bundle.bias) : new Float32Array([0]),
    );
    this.inputScaleBuffer = createStorageBuffer(
      device,
      bundle.inputScales ? new Float32Array(bundle.inputScales) : new Float32Array([1]),
    );
    this.paramsBuffer = device.createBuffer({
      size: PARAM_BUFFER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  static async fromManifestLayer(device, manifest, layer, manifestUrl) {
    const baseUrl = new URL(".", manifestUrl).toString();
    const runtimeBaseUrl = resolveUrl(".", resolveUrl(manifest.runtime.files.wgsl, baseUrl));
    const shaderCode = await fetchText(resolveUrl(manifest.runtime.files.wgsl, baseUrl));
    const tensors = layer.tensors;
    return new BitNetLinearWebGPU(device, {
      shaderCode,
      layoutHeader: layer.layout_header,
      packedWeight: await fetchTensor(tensors.packed_weight, resolveUrl("layers/", baseUrl), Uint8Array),
      scaleValues: await fetchTensor(tensors.scale_values, resolveUrl("layers/", baseUrl), Float32Array),
      segmentOffsets: await fetchTensor(tensors.segment_offsets, resolveUrl("layers/", baseUrl), Int32Array),
      bias: tensors.bias ? await fetchTensor(tensors.bias, resolveUrl("layers/", baseUrl), Float32Array) : null,
      inputScales: await fetchTensor(tensors.act_scale, resolveUrl("layers/", baseUrl), tensorType(tensors.act_scale)),
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

    const outputLength = rows * this.layout.logicalOut;
    const inputBuffer = createStorageBuffer(this.device, x);
    const outputBuffer = createOutputBuffer(this.device, outputLength * Float32Array.BYTES_PER_ELEMENT);
    const readbackBuffer = createReadbackBuffer(this.device, outputLength * Float32Array.BYTES_PER_ELEMENT);

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

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.layout.logicalOut / 8), Math.ceil(rows / 8), 1);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
    this.device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const mapped = readbackBuffer.getMappedRange();
    const result = new Float32Array(mapped.slice(0));
    readbackBuffer.unmap();
    return result;
  }
}
