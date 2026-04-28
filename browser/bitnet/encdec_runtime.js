import { BitNetLinearWebGPU } from "./bitnet_webgpu.js";

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

async function fetchFloatTensor(entry, baseUrl) {
  const response = await fetch(resolveUrl(entry.path, baseUrl));
  if (!response.ok) {
    throw new Error(`failed to fetch ${entry.path}: ${response.status}`);
  }
  return { data: new Float32Array(await response.arrayBuffer()), shape: entry.shape };
}

function zeros(length) {
  return new Float32Array(length);
}

function addInPlace(dst, src) {
  for (let i = 0; i < dst.length; i += 1) {
    dst[i] += src[i];
  }
  return dst;
}

function layerNorm(x, rows, cols, weight, bias, eps = 1e-5) {
  const out = new Float32Array(x.length);
  for (let r = 0; r < rows; r += 1) {
    let mean = 0;
    for (let c = 0; c < cols; c += 1) mean += x[r * cols + c];
    mean /= cols;
    let variance = 0;
    for (let c = 0; c < cols; c += 1) {
      const d = x[r * cols + c] - mean;
      variance += d * d;
    }
    const inv = 1 / Math.sqrt(variance / cols + eps);
    for (let c = 0; c < cols; c += 1) {
      out[r * cols + c] = (x[r * cols + c] - mean) * inv * weight[c] + (bias ? bias[c] : 0);
    }
  }
  return out;
}

function silu(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i += 1) {
    out[i] = x[i] / (1 + Math.exp(-x[i]));
  }
  return out;
}

function gelu(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i += 1) {
    const v = x[i];
    out[i] = 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v * v * v)));
  }
  return out;
}

function activate(x, name) {
  const normalized = String(name || "silu").toLowerCase();
  if (normalized === "gelu") return gelu(x);
  return silu(x);
}

function gatedActivation(x, name) {
  const half = x.length / 2;
  const out = new Float32Array(half);
  const gateName = String(name || "swiglu").toLowerCase();
  const a = x.subarray(0, half);
  const b = x.subarray(half);
  const activated = gateName === "geglu" ? gelu(a) : silu(a);
  for (let i = 0; i < half; i += 1) {
    out[i] = activated[i] * b[i];
  }
  return out;
}

function embed(tokens, embedding, dModel) {
  const out = new Float32Array(tokens.length * dModel);
  for (let t = 0; t < tokens.length; t += 1) {
    const token = tokens[t];
    out.set(embedding.subarray(token * dModel, token * dModel + dModel), t * dModel);
  }
  return out;
}

function splitHeads(x, seqLen, nHeads, headDim) {
  const out = [];
  for (let h = 0; h < nHeads; h += 1) {
    const head = new Float32Array(seqLen * headDim);
    for (let t = 0; t < seqLen; t += 1) {
      const src = t * nHeads * headDim + h * headDim;
      head.set(x.subarray(src, src + headDim), t * headDim);
    }
    out.push(head);
  }
  return out;
}

function mergeHeads(heads, seqLen, nHeads, headDim) {
  const out = new Float32Array(seqLen * nHeads * headDim);
  for (let h = 0; h < nHeads; h += 1) {
    for (let t = 0; t < seqLen; t += 1) {
      out.set(heads[h].subarray(t * headDim, t * headDim + headDim), t * nHeads * headDim + h * headDim);
    }
  }
  return out;
}

function attention(q, k, v, qLen, kvLen, nHeads, headDim, causal) {
  const qh = splitHeads(q, qLen, nHeads, headDim);
  const kh = splitHeads(k, kvLen, nHeads, headDim);
  const vh = splitHeads(v, kvLen, nHeads, headDim);
  const outHeads = [];
  const scale = 1 / Math.sqrt(headDim);
  for (let h = 0; h < nHeads; h += 1) {
    const out = new Float32Array(qLen * headDim);
    for (let i = 0; i < qLen; i += 1) {
      const scores = new Float32Array(kvLen);
      let maxScore = -Infinity;
      for (let j = 0; j < kvLen; j += 1) {
        let score = causal && j > i ? -1e30 : 0;
        if (score > -1e20) {
          for (let d = 0; d < headDim; d += 1) {
            score += qh[h][i * headDim + d] * kh[h][j * headDim + d] * scale;
          }
        }
        scores[j] = score;
        maxScore = Math.max(maxScore, score);
      }
      let denom = 0;
      for (let j = 0; j < kvLen; j += 1) {
        scores[j] = Math.exp(scores[j] - maxScore);
        denom += scores[j];
      }
      for (let d = 0; d < headDim; d += 1) {
        let sum = 0;
        for (let j = 0; j < kvLen; j += 1) {
          sum += (scores[j] / denom) * vh[h][j * headDim + d];
        }
        out[i * headDim + d] = sum;
      }
    }
    outHeads.push(out);
  }
  return mergeHeads(outHeads, qLen, nHeads, headDim);
}

export class BitNetEncoderDecoderWebGPU {
  constructor(device, manifest, manifestUrl, denseTensors, linears) {
    if (manifest.graph?.architecture !== "encoder_decoder") {
      throw new Error("manifest is not an encoder_decoder browser BitNet bundle");
    }
    this.device = device;
    this.manifest = manifest;
    this.manifestUrl = manifestUrl;
    this.dense = denseTensors;
    this.linears = linears;
    this.graph = manifest.graph;
  }

  static async fromManifestUrl(device, manifestUrl) {
    const manifest = await fetchJson(manifestUrl);
    const baseUrl = new URL(".", manifestUrl).toString();
    const dense = {};
    for (const [name, entry] of Object.entries(manifest.dense_tensors || {})) {
      dense[name] = await fetchFloatTensor(entry, baseUrl);
    }
    const linears = {};
    for (const layer of manifest.layers) {
      linears[layer.name] = await BitNetLinearWebGPU.fromManifestLayer(device, manifest, layer, manifestUrl);
    }
    return new BitNetEncoderDecoderWebGPU(device, manifest, manifestUrl, dense, linears);
  }

  linear(name) {
    const layer = this.linears[name];
    if (!layer) throw new Error(`missing BitNet linear: ${name}`);
    return layer;
  }

  tensor(name) {
    const tensor = this.dense[name];
    if (!tensor) throw new Error(`missing dense tensor: ${name}`);
    return tensor.data;
  }

  async attentionBlock(prefix, x, seqLen, kv, kvLen, causal) {
    const dModel = this.graph.d_model;
    const nHeads = this.graph.n_heads;
    const headDim = this.graph.head_dim;
    const q = await this.linear(`${prefix}.w_q`).run(x, seqLen);
    const kInput = kv || x;
    const kRows = kvLen || seqLen;
    const k = await this.linear(`${prefix}.w_k`).run(kInput, kRows);
    const v = await this.linear(`${prefix}.w_v`).run(kInput, kRows);
    const merged = attention(q, k, v, seqLen, kRows, nHeads, headDim, causal);
    return this.linear(`${prefix}.w_o`).run(merged, seqLen);
  }

  async mlp(prefix, x, seqLen) {
    const hidden = await this.linear(`${prefix}.w_in`).run(x, seqLen);
    const activation = String(this.graph.activation || "silu").toLowerCase();
    const activated = ["swiglu", "gated-silu", "geglu", "reglu"].includes(activation)
      ? gatedActivation(hidden, activation)
      : activate(hidden, activation);
    return this.linear(`${prefix}.w_out`).run(activated, seqLen);
  }

  async encoderLayer(index, x, seqLen) {
    const dModel = this.graph.d_model;
    const n1 = layerNorm(
      x,
      seqLen,
      dModel,
      this.tensor(`encoder.${index}.n1.weight`),
      this.dense[`encoder.${index}.n1.bias`]?.data,
    );
    const attnOut = await this.attentionBlock(`encoder.${index}.attn`, n1, seqLen, null, null, false);
    x = addInPlace(x.slice(), attnOut);
    const n2 = layerNorm(
      x,
      seqLen,
      dModel,
      this.tensor(`encoder.${index}.n2.weight`),
      this.dense[`encoder.${index}.n2.bias`]?.data,
    );
    return addInPlace(x, await this.mlp(`encoder.${index}.mlp`, n2, seqLen));
  }

  async decoderLayer(index, x, seqLen, memory, memoryLen) {
    const dModel = this.graph.d_model;
    let n = layerNorm(
      x,
      seqLen,
      dModel,
      this.tensor(`decoder.${index}.self_attn_block.n1.weight`),
      this.dense[`decoder.${index}.self_attn_block.n1.bias`]?.data,
    );
    x = addInPlace(x.slice(), await this.attentionBlock(`decoder.${index}.self_attn_block.attn`, n, seqLen, null, null, true));
    n = layerNorm(
      x,
      seqLen,
      dModel,
      this.tensor(`decoder.${index}.self_attn_block.n2.weight`),
      this.dense[`decoder.${index}.self_attn_block.n2.bias`]?.data,
    );
    x = addInPlace(x, await this.mlp(`decoder.${index}.self_attn_block.mlp`, n, seqLen));
    n = layerNorm(
      x,
      seqLen,
      dModel,
      this.tensor(`decoder.${index}.cross_block.n1.weight`),
      this.dense[`decoder.${index}.cross_block.n1.bias`]?.data,
    );
    x = addInPlace(x.slice(), await this.attentionBlock(`decoder.${index}.cross_block.cross`, n, seqLen, memory, memoryLen, false));
    n = layerNorm(
      x,
      seqLen,
      dModel,
      this.tensor(`decoder.${index}.cross_block.n2.weight`),
      this.dense[`decoder.${index}.cross_block.n2.bias`]?.data,
    );
    return addInPlace(x, await this.mlp(`decoder.${index}.cross_block.mlp`, n, seqLen));
  }

  async encode(encInputIds) {
    let x = embed(encInputIds, this.tensor("enc_embed.weight"), this.graph.d_model);
    for (let i = 0; i < this.graph.n_layers; i += 1) {
      x = await this.encoderLayer(i, x, encInputIds.length);
    }
    return layerNorm(
      x,
      encInputIds.length,
      this.graph.d_model,
      this.tensor("enc_norm.weight"),
      this.dense["enc_norm.bias"]?.data,
    );
  }

  async decode(decInputIds, memory, memoryLen) {
    let x = embed(decInputIds, this.tensor("dec_embed.weight"), this.graph.d_model);
    for (let i = 0; i < this.graph.n_layers; i += 1) {
      x = await this.decoderLayer(i, x, decInputIds.length, memory, memoryLen);
    }
    return layerNorm(
      x,
      decInputIds.length,
      this.graph.d_model,
      this.tensor("dec_norm.weight"),
      this.dense["dec_norm.bias"]?.data,
    );
  }

  async forward(encInputIds, decInputIds) {
    const memory = await this.encode(encInputIds);
    const hidden = await this.decode(decInputIds, memory, encInputIds.length);
    return this.linear("lm_head").run(hidden, decInputIds.length);
  }
}
