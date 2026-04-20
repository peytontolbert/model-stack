#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kSmallRowThreads = 64;
constexpr int kMediumRowThreads = 128;
constexpr int kBeamSearchThreads = 128;
constexpr int kMaxBeamSearchBeamSize = 32;
constexpr int kSpeculativeAcceptStrict = 0;
constexpr int kSpeculativeAcceptRejectionSampler = 1;
constexpr int kSpeculativeAcceptTypical = 2;

struct SamplingLaunchPolicy {
  int elementwise_threads;
  int row_reduce_threads;
};

SamplingLaunchPolicy SelectSamplingLaunchPolicy(int64_t vocab_size) {
  if (vocab_size <= kSmallRowThreads) {
    return {kThreads, kSmallRowThreads};
  }
  if (vocab_size <= kMediumRowThreads) {
    return {kThreads, kMediumRowThreads};
  }
  return {kThreads, kThreads};
}

__device__ inline uint64_t SplitMix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

__device__ inline float Uniform01FromSeed(uint64_t seed) {
  const uint64_t mixed = SplitMix64(seed);
  const uint32_t mantissa = static_cast<uint32_t>((mixed >> 40) & 0xFFFFFFu);
  return (static_cast<float>(mantissa) + 0.5f) * (1.0f / 16777216.0f);
}

__device__ inline uint64_t SpeculativeSampleSeed(
    int64_t row,
    int64_t step,
    uint64_t salt) {
  return static_cast<uint64_t>(clock64()) ^
         (static_cast<uint64_t>(row + 1) * 0x9E3779B97F4A7C15ull) ^
         (static_cast<uint64_t>(step + 1) * 0xBF58476D1CE4E5B9ull) ^
         salt;
}

template <int Threads>
__device__ inline int64_t BlockArgmaxProbIndex(
    const float* probs,
    int64_t vocab_size,
    float* shared_values,
    int64_t* shared_indices) {
  float best_value = -std::numeric_limits<float>::infinity();
  int64_t best_index = 0;
  for (int64_t idx = threadIdx.x; idx < vocab_size; idx += Threads) {
    const float value = probs[idx];
    if (value > best_value || (value == best_value && idx < best_index)) {
      best_value = value;
      best_index = idx;
    }
  }

  shared_values[threadIdx.x] = best_value;
  shared_indices[threadIdx.x] = best_index;
  __syncthreads();

  for (int stride = Threads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      const float other_value = shared_values[threadIdx.x + stride];
      const int64_t other_index = shared_indices[threadIdx.x + stride];
      const float current_value = shared_values[threadIdx.x];
      const int64_t current_index = shared_indices[threadIdx.x];
      if (other_value > current_value ||
          (other_value == current_value && other_index < current_index)) {
        shared_values[threadIdx.x] = other_value;
        shared_indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }

  return shared_indices[0];
}

template <int Threads>
__device__ inline double BlockEntropy(
    const float* probs,
    int64_t vocab_size,
    double* shared_sums) {
  double entropy = 0.0;
  for (int64_t idx = threadIdx.x; idx < vocab_size; idx += Threads) {
    const float value = fmaxf(probs[idx], 0.0f);
    entropy -= static_cast<double>(value) * static_cast<double>(logf(value + 1.0e-5f));
  }
  shared_sums[threadIdx.x] = entropy;
  __syncthreads();

  for (int stride = Threads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_sums[threadIdx.x] += shared_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }

  return shared_sums[0];
}

__device__ inline int64_t ArgmaxProbIndex(
    const float* probs,
    int64_t vocab_size) {
  float best_value = -std::numeric_limits<float>::infinity();
  int64_t best_index = 0;
  for (int64_t idx = 0; idx < vocab_size; ++idx) {
    const float value = probs[idx];
    if (value > best_value || (value == best_value && idx < best_index)) {
      best_value = value;
      best_index = idx;
    }
  }
  return best_index;
}

__device__ inline int64_t SamplePositiveProbIndex(
    const float* probs,
    int64_t vocab_size,
    uint64_t seed) {
  double total = 0.0;
  int64_t fallback = ArgmaxProbIndex(probs, vocab_size);
  int64_t last_positive = fallback;
  for (int64_t idx = 0; idx < vocab_size; ++idx) {
    const float value = fmaxf(probs[idx], 0.0f);
    if (value > 0.0f) {
      last_positive = idx;
    }
    total += static_cast<double>(value);
  }
  if (total <= 1e-8) {
    return fallback;
  }
  const double threshold = static_cast<double>(Uniform01FromSeed(seed)) * total;
  double cumulative = 0.0;
  for (int64_t idx = 0; idx < vocab_size; ++idx) {
    const float value = fmaxf(probs[idx], 0.0f);
    if (value <= 0.0f) {
      continue;
    }
    cumulative += static_cast<double>(value);
    if (cumulative >= threshold) {
      return idx;
    }
  }
  return last_positive;
}

__device__ inline int64_t SampleResidualProbIndex(
    const float* target_probs,
    const float* draft_probs,
    int64_t vocab_size,
    uint64_t seed) {
  double total = 0.0;
  for (int64_t idx = 0; idx < vocab_size; ++idx) {
    total += static_cast<double>(fmaxf(target_probs[idx] - draft_probs[idx], 0.0f));
  }
  if (total <= 1e-8) {
    return SamplePositiveProbIndex(target_probs, vocab_size, seed);
  }
  const double threshold = static_cast<double>(Uniform01FromSeed(seed)) * total;
  double cumulative = 0.0;
  int64_t last_positive = ArgmaxProbIndex(target_probs, vocab_size);
  for (int64_t idx = 0; idx < vocab_size; ++idx) {
    const float value = fmaxf(target_probs[idx] - draft_probs[idx], 0.0f);
    if (value <= 0.0f) {
      continue;
    }
    last_positive = idx;
    cumulative += static_cast<double>(value);
    if (cumulative >= threshold) {
      return idx;
    }
  }
  return last_positive;
}

template <int Threads>
__global__ void speculative_accept_forward_kernel(
    const float* __restrict__ target_probs,
    const float* __restrict__ draft_probs,
    const int64_t* __restrict__ draft_token_ids,
    const float* __restrict__ bonus_probs,
    const bool* __restrict__ bonus_enabled,
    int64_t* __restrict__ out_tokens,
    int64_t* __restrict__ out_lengths,
    int64_t* __restrict__ accepted_counts,
    int64_t batch_size,
    int64_t draft_len,
    int64_t vocab_size,
    int method_code,
    float posterior_threshold,
    float posterior_alpha) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= batch_size) {
    return;
  }

  __shared__ float shared_values[Threads];
  __shared__ int64_t shared_indices[Threads];
  __shared__ double shared_sums[Threads];
  __shared__ int64_t shared_candidate;
  __shared__ int64_t shared_emitted;
  __shared__ int64_t shared_accepted;
  __shared__ int shared_stop;

  const int64_t out_base = row * (draft_len + 1);
  const int64_t token_base = row * draft_len;
  if (threadIdx.x == 0) {
    shared_emitted = 0;
    shared_accepted = 0;
    shared_stop = 0;
  }
  __syncthreads();

  for (int64_t step = 0; step < draft_len; ++step) {
    if (threadIdx.x == 0) {
      shared_candidate = draft_token_ids[token_base + step];
      shared_stop = 0;
    }
    __syncthreads();

    const int64_t candidate = shared_candidate;
    const float* target_row = target_probs + ((row * draft_len + step) * vocab_size);
    if (method_code == kSpeculativeAcceptStrict) {
      const int64_t expected =
          BlockArgmaxProbIndex<Threads>(target_row, vocab_size, shared_values, shared_indices);
      if (threadIdx.x == 0) {
        if (candidate == expected) {
          out_tokens[out_base + shared_emitted] = candidate;
          shared_emitted += 1;
          shared_accepted += 1;
        } else {
          out_tokens[out_base + shared_emitted] = expected;
          shared_emitted += 1;
          shared_stop = 1;
        }
      }
      __syncthreads();
      if (shared_stop != 0) {
        break;
      }
      continue;
    }
    if (method_code == kSpeculativeAcceptTypical) {
      const double entropy = BlockEntropy<Threads>(target_row, vocab_size, shared_sums);
      if (threadIdx.x == 0) {
        float candidate_prob = 0.0f;
        if (candidate >= 0 && candidate < vocab_size) {
          candidate_prob = target_row[candidate];
        }
        const float threshold =
            fminf(posterior_threshold, expf(static_cast<float>(-entropy)) * posterior_alpha);
        if (candidate_prob > threshold) {
          out_tokens[out_base + shared_emitted] = candidate;
          shared_emitted += 1;
          shared_accepted += 1;
        } else {
          out_tokens[out_base + shared_emitted] = SamplePositiveProbIndex(
              target_row,
              vocab_size,
              SpeculativeSampleSeed(row, step, 0xA5A5A5A5ull));
          shared_emitted += 1;
          shared_stop = 1;
        }
      }
      __syncthreads();
      if (shared_stop != 0) {
        break;
      }
      continue;
    }

    if (threadIdx.x == 0) {
      const float* draft_row = draft_probs + ((row * draft_len + step) * vocab_size);
      float px = 0.0f;
      float qx = 0.0f;
      if (candidate >= 0 && candidate < vocab_size) {
        px = target_row[candidate];
        qx = draft_row[candidate];
      }
      const float accept_prob = qx <= 1e-12f ? 1.0f : fminf(px / qx, 1.0f);
      if (Uniform01FromSeed(SpeculativeSampleSeed(row, step, 0x13579BDFull)) <= accept_prob) {
        out_tokens[out_base + shared_emitted] = candidate;
        shared_emitted += 1;
        shared_accepted += 1;
      } else {
        out_tokens[out_base + shared_emitted] = SampleResidualProbIndex(
            target_row,
            draft_row,
            vocab_size,
            SpeculativeSampleSeed(row, step, 0x2468ACE0ull));
        shared_emitted += 1;
        shared_stop = 1;
      }
    }
    __syncthreads();
    if (shared_stop != 0) {
      break;
    }
  }

  const bool emit_bonus =
      shared_accepted == draft_len && bonus_probs != nullptr && (bonus_enabled == nullptr || bonus_enabled[row]);
  if (emit_bonus && method_code == kSpeculativeAcceptStrict) {
    const float* bonus_row = bonus_probs + (row * vocab_size);
    const int64_t expected =
        BlockArgmaxProbIndex<Threads>(bonus_row, vocab_size, shared_values, shared_indices);
    if (threadIdx.x == 0) {
      out_tokens[out_base + shared_emitted] = expected;
      shared_emitted += 1;
    }
  } else if (emit_bonus && threadIdx.x == 0) {
    const float* bonus_row = bonus_probs + (row * vocab_size);
    out_tokens[out_base + shared_emitted] = SamplePositiveProbIndex(
        bonus_row,
        vocab_size,
        SpeculativeSampleSeed(row, draft_len, 0x11223344ull));
    shared_emitted += 1;
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    out_lengths[row] = shared_emitted;
    accepted_counts[row] = shared_accepted;
  }
}

template <typename scalar_t>
__global__ void temperature_forward_kernel(
    const scalar_t* __restrict__ logits,
    scalar_t* __restrict__ out,
    int64_t numel,
    float inv_tau) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }
  out[idx] = static_cast<scalar_t>(static_cast<float>(logits[idx]) * inv_tau);
}

template <typename scalar_t>
__global__ void prepare_sampling_logits_kernel(
    const scalar_t* __restrict__ logits,
    const bool* __restrict__ seen_mask,
    const float* __restrict__ seen_counts,
    scalar_t* __restrict__ out,
    int64_t numel,
    float repetition_penalty,
    float inv_tau,
    bool apply_repetition_penalty,
    bool apply_temperature) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }

  float value = static_cast<float>(logits[idx]);
  const bool seen =
      (seen_mask != nullptr && seen_mask[idx]) ||
      (seen_counts != nullptr && seen_counts[idx] > 0.0f);
  if (apply_repetition_penalty && seen) {
    value = value > 0.0f ? (value / repetition_penalty) : (value * repetition_penalty);
  }
  if (apply_temperature) {
    value *= inv_tau;
  }
  out[idx] = static_cast<scalar_t>(value);
}

template <typename scalar_t>
__global__ void presence_frequency_penalty_forward_kernel(
    const scalar_t* __restrict__ logits,
    const scalar_t* __restrict__ counts,
    scalar_t* __restrict__ out,
    int64_t numel,
    float alpha_presence,
    float alpha_frequency) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }
  const float count = static_cast<float>(counts[idx]);
  const float penalty =
      (count > 0.0f ? alpha_presence : 0.0f) + (alpha_frequency * count);
  out[idx] = static_cast<scalar_t>(static_cast<float>(logits[idx]) - penalty);
}

template <typename scalar_t>
__global__ void apply_sampling_mask_forward_kernel(
    const scalar_t* __restrict__ logits,
    const bool* __restrict__ topk_mask,
    const bool* __restrict__ topp_mask,
    const bool* __restrict__ no_repeat_mask,
    scalar_t* __restrict__ out,
    int64_t numel,
    scalar_t fill_value) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }
  const bool masked =
      (topk_mask != nullptr && topk_mask[idx]) ||
      (topp_mask != nullptr && topp_mask[idx]) ||
      (no_repeat_mask != nullptr && no_repeat_mask[idx]);
  out[idx] = masked ? fill_value : logits[idx];
}

template <typename scalar_t>
__global__ void finalize_sampling_logits_kernel(
    const scalar_t* __restrict__ logits,
    const bool* __restrict__ seen_mask,
    const float* __restrict__ seen_counts,
    const bool* __restrict__ combined_mask,
    const float* __restrict__ kth_values,
    scalar_t* __restrict__ out,
    int64_t numel,
    int64_t vocab_size,
    scalar_t fill_value,
    float alpha_presence,
    float alpha_frequency,
    bool apply_history_mask,
    bool apply_presence_frequency) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }

  const int64_t row = idx / vocab_size;
  const bool topk_masked =
      kth_values != nullptr && static_cast<float>(logits[idx]) < kth_values[row];
  const bool history_masked =
      apply_history_mask &&
      ((seen_mask != nullptr && seen_mask[idx]) ||
       (seen_counts != nullptr && seen_counts[idx] > 0.0f));
  float value =
      ((combined_mask != nullptr && combined_mask[idx]) || topk_masked || history_masked)
          ? static_cast<float>(fill_value)
          : static_cast<float>(logits[idx]);
  if (apply_presence_frequency) {
    const float count =
        seen_counts != nullptr
            ? seen_counts[idx]
            : ((seen_mask != nullptr && seen_mask[idx]) ? 1.0f : 0.0f);
    value -= (count > 0.0f ? alpha_presence : 0.0f) + (alpha_frequency * count);
  }
  out[idx] = static_cast<scalar_t>(value);
}

__global__ void token_counts_forward_kernel(
    const int64_t* __restrict__ token_ids,
    float* __restrict__ counts,
    int32_t* __restrict__ invalid_flags,
    int64_t batch_size,
    int64_t seq_len,
    int64_t vocab_size) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch_size * seq_len;
  if (idx >= total) {
    return;
  }

  const int64_t b = idx / seq_len;
  const int64_t token = token_ids[idx];
  if (token < 0) {
    atomicExch(&invalid_flags[0], 1);
    return;
  }
  if (token >= vocab_size) {
    atomicExch(&invalid_flags[1], 1);
    return;
  }
  atomicAdd(&counts[(b * vocab_size) + token], 1.0f);
}

__global__ void seen_mask_forward_kernel(
    const int64_t* __restrict__ token_ids,
    bool* __restrict__ seen_mask,
    int32_t* __restrict__ invalid_flags,
    int64_t batch_size,
    int64_t seq_len,
    int64_t vocab_size) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= batch_size || threadIdx.x != 0) {
    return;
  }

  const int64_t base = row * seq_len;
  const int64_t mask_base = row * vocab_size;
  for (int64_t pos = 0; pos < seq_len; ++pos) {
    const int64_t token = token_ids[base + pos];
    if (token < 0) {
      atomicExch(&invalid_flags[0], 1);
      continue;
    }
    if (token >= vocab_size) {
      atomicExch(&invalid_flags[1], 1);
      continue;
    }
    seen_mask[mask_base + token] = true;
  }
}

__global__ void fill_row_indices_kernel(
    int64_t* __restrict__ indices,
    int64_t batch_size,
    int64_t vocab_size) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch_size * vocab_size;
  if (idx >= total) {
    return;
  }
  indices[idx] = idx % vocab_size;
}

template <typename scalar_t>
__global__ void frequency_penalty_from_tokens_inplace_kernel(
    const int64_t* __restrict__ token_ids,
    scalar_t* __restrict__ logits,
    int64_t batch_size,
    int64_t seq_len,
    int64_t vocab_size,
    float alpha_frequency) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= batch_size || threadIdx.x != 0) {
    return;
  }

  const int64_t token_base = row * seq_len;
  const int64_t logits_base = row * vocab_size;
  for (int64_t pos = 0; pos < seq_len; ++pos) {
    const int64_t token = token_ids[token_base + pos];
    const int64_t idx = logits_base + token;
    logits[idx] = static_cast<scalar_t>(static_cast<float>(logits[idx]) - alpha_frequency);
  }
}

template <typename scalar_t>
__global__ void presence_penalty_from_seen_mask_inplace_kernel(
    const bool* __restrict__ seen_mask,
    scalar_t* __restrict__ logits,
    int64_t numel,
    float alpha_presence) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }
  if (!seen_mask[idx]) {
    return;
  }
  logits[idx] = static_cast<scalar_t>(static_cast<float>(logits[idx]) - alpha_presence);
}

__global__ void no_repeat_ngram_mask_forward_kernel(
    const int64_t* __restrict__ token_ids,
    bool* __restrict__ mask,
    int64_t batch_size,
    int64_t seq_len,
    int64_t vocab_size,
    int64_t n) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= batch_size || threadIdx.x != 0) {
    return;
  }

  if (n <= 0 || seq_len < n) {
    return;
  }

  const int64_t base = row * seq_len;
  const int64_t mask_base = row * vocab_size;
  if (n == 1) {
    for (int64_t pos = 0; pos < seq_len; ++pos) {
      const int64_t token = token_ids[base + pos];
      if (token >= 0 && token < vocab_size) {
        mask[mask_base + token] = true;
      }
    }
    return;
  }

  const int64_t prefix_len = n - 1;
  const int64_t recent_base = base + seq_len - prefix_len;
  for (int64_t start = 0; start <= seq_len - n; ++start) {
    bool match = true;
    for (int64_t offset = 0; offset < prefix_len; ++offset) {
      if (token_ids[base + start + offset] != token_ids[recent_base + offset]) {
        match = false;
        break;
      }
    }
    if (!match) {
      continue;
    }
    const int64_t token = token_ids[base + start + prefix_len];
    if (token >= 0 && token < vocab_size) {
      mask[mask_base + token] = true;
    }
  }
}

template <typename scalar_t>
__global__ void repetition_penalty_forward_kernel(
    const scalar_t* __restrict__ logits,
    const bool* __restrict__ seen_mask,
    scalar_t* __restrict__ out,
    int64_t numel,
    float penalty) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }
  const float value = static_cast<float>(logits[idx]);
  if (seen_mask[idx]) {
    out[idx] = static_cast<scalar_t>(value > 0.0f ? (value / penalty) : (value * penalty));
    return;
  }
  out[idx] = logits[idx];
}

template <typename scalar_t>
__global__ void topk_mask_forward_kernel(
    const scalar_t* __restrict__ logits,
    const float* __restrict__ kth_values,
    bool* __restrict__ mask,
    int64_t batch_size,
    int64_t vocab_size) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch_size * vocab_size;
  if (idx >= total) {
    return;
  }
  const int64_t row = idx / vocab_size;
  mask[idx] = mask[idx] || (static_cast<float>(logits[idx]) < kth_values[row]);
}

template <typename scalar_t, int Threads>
__global__ void greedy_next_token_forward_kernel(
    const scalar_t* __restrict__ logits,
    int64_t* __restrict__ next_token,
    int64_t batch_size,
    int64_t vocab_size) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= batch_size) {
    return;
  }

  __shared__ float shared_values[Threads];
  __shared__ int64_t shared_indices[Threads];

  float best_value = -std::numeric_limits<float>::infinity();
  int64_t best_index = 0;
  const int64_t base = row * vocab_size;

  for (int64_t col = threadIdx.x; col < vocab_size; col += Threads) {
    const float value = static_cast<float>(logits[base + col]);
    if (value > best_value || (value == best_value && col < best_index)) {
      best_value = value;
      best_index = col;
    }
  }

  shared_values[threadIdx.x] = best_value;
  shared_indices[threadIdx.x] = best_index;
  __syncthreads();

  for (int stride = Threads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      const float other_value = shared_values[threadIdx.x + stride];
      const int64_t other_index = shared_indices[threadIdx.x + stride];
      const float current_value = shared_values[threadIdx.x];
      const int64_t current_index = shared_indices[threadIdx.x];
      if (other_value > current_value ||
          (other_value == current_value && other_index < current_index)) {
        shared_values[threadIdx.x] = other_value;
        shared_indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    next_token[row] = shared_indices[0];
  }
}

template <typename scalar_t, int Threads>
__global__ void filtered_greedy_next_token_forward_kernel(
    const scalar_t* __restrict__ logits,
    const bool* __restrict__ seen_mask,
    const bool* __restrict__ blocked_mask,
    const float* __restrict__ kth_values,
    bool apply_history_mask,
    float alpha_presence,
    int64_t* __restrict__ next_token,
    int64_t batch_size,
    int64_t vocab_size) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= batch_size) {
    return;
  }

  __shared__ float shared_values[Threads];
  __shared__ int64_t shared_indices[Threads];
  __shared__ int shared_counts[Threads];

  const int64_t base = row * vocab_size;
  const float kth_value =
      kth_values != nullptr ? kth_values[row] : -std::numeric_limits<float>::infinity();

  float best_value = -std::numeric_limits<float>::infinity();
  int64_t best_index = 0;
  int allowed_count = 0;
  for (int64_t col = threadIdx.x; col < vocab_size; col += Threads) {
    const bool passes_topk =
        kth_values == nullptr || static_cast<float>(logits[base + col]) >= kth_value;
    const bool passes_history =
        !apply_history_mask || seen_mask == nullptr || !seen_mask[base + col];
    const bool passes_block_mask = blocked_mask == nullptr || !blocked_mask[base + col];
    if (!passes_topk || !passes_history || !passes_block_mask) {
      continue;
    }
    const bool seen = seen_mask != nullptr && seen_mask[base + col];
    const float value =
        static_cast<float>(logits[base + col]) - (seen ? alpha_presence : 0.0f);
    if (value > best_value || (value == best_value && col < best_index)) {
      best_value = value;
      best_index = col;
    }
    ++allowed_count;
  }

  shared_values[threadIdx.x] = best_value;
  shared_indices[threadIdx.x] = best_index;
  shared_counts[threadIdx.x] = allowed_count;
  __syncthreads();

  for (int stride = Threads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      const float other_value = shared_values[threadIdx.x + stride];
      const int64_t other_index = shared_indices[threadIdx.x + stride];
      const float current_value = shared_values[threadIdx.x];
      const int64_t current_index = shared_indices[threadIdx.x];
      if (other_value > current_value ||
          (other_value == current_value && other_index < current_index)) {
        shared_values[threadIdx.x] = other_value;
        shared_indices[threadIdx.x] = other_index;
      }
      shared_counts[threadIdx.x] += shared_counts[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    next_token[row] =
        shared_counts[0] > 0 ? shared_indices[0] : static_cast<int64_t>(0);
  }
}

template <int Threads>
__global__ void topp_mask_forward_kernel(
    const float* __restrict__ sorted_logits,
    const int64_t* __restrict__ sorted_indices,
    bool* __restrict__ mask,
    int64_t batch_size,
    int64_t vocab_size,
    float p) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= batch_size) {
    return;
  }

  __shared__ double shared_sums[Threads];
  const int64_t base = row * vocab_size;
  const float row_max = sorted_logits[base];

  double row_sum = 0.0;
  for (int64_t col = threadIdx.x; col < vocab_size; col += Threads) {
    row_sum += std::exp(static_cast<double>(sorted_logits[base + col] - row_max));
  }
  shared_sums[threadIdx.x] = row_sum;
  __syncthreads();

  for (int stride = Threads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_sums[threadIdx.x] += shared_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    row_sum = shared_sums[0];
    double cumulative = 0.0;
    for (int64_t col = 0; col < vocab_size; ++col) {
      cumulative += std::exp(static_cast<double>(sorted_logits[base + col] - row_max)) / row_sum;
      const int64_t token_index = sorted_indices[base + col];
      if (col > 0 && cumulative > static_cast<double>(p)) {
        mask[base + token_index] = true;
      }
    }
  }
}

template <typename scalar_t, int Threads>
__global__ void sorted_filtered_sample_kernel(
    const float* __restrict__ sorted_logits,
    const int64_t* __restrict__ sorted_indices,
    const scalar_t* __restrict__ sample_logits,
    const bool* __restrict__ seen_mask,
    const bool* __restrict__ blocked_mask,
    const float* __restrict__ kth_values,
    const float* __restrict__ row_sums,
    bool apply_history_mask,
    float alpha_presence,
    int64_t* __restrict__ next_token,
    int64_t batch_size,
    int64_t sorted_size,
    int64_t sample_vocab_size,
    float p) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= batch_size) {
    return;
  }

  __shared__ double shared_sums[Threads];
  const int64_t sorted_base = row * sorted_size;
  const int64_t sample_base = row * sample_vocab_size;
  const float row_max = sorted_logits[sorted_base];

  if (row_sums == nullptr) {
    double local_row_sum = 0.0;
    for (int64_t col = threadIdx.x; col < sorted_size; col += Threads) {
      local_row_sum += std::exp(static_cast<double>(sorted_logits[sorted_base + col] - row_max));
    }
    shared_sums[threadIdx.x] = local_row_sum;
    __syncthreads();

    for (int stride = Threads / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride) {
        shared_sums[threadIdx.x] += shared_sums[threadIdx.x + stride];
      }
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    double row_sum = 0.0;
    if (row_sums != nullptr) {
      row_sum = static_cast<double>(row_sums[row]);
    } else {
      for (int64_t col = 0; col < sorted_size; ++col) {
        row_sum += std::exp(static_cast<double>(sorted_logits[sorted_base + col] - row_max));
      }
    }
    const float kth_value =
        kth_values != nullptr ? kth_values[row] : -std::numeric_limits<float>::infinity();
    double cumulative = 0.0;
    float allowed_max = -std::numeric_limits<float>::infinity();
    for (int64_t col = 0; col < sorted_size; ++col) {
      const float logit = sorted_logits[sorted_base + col];
      const int64_t token = sorted_indices[sorted_base + col];
      const double prob = std::exp(static_cast<double>(logit - row_max)) / row_sum;
      cumulative += prob;
      const bool passes_topk = kth_values == nullptr || logit >= kth_value;
      const bool seen = seen_mask != nullptr && seen_mask[sample_base + token];
      const bool passes_history = !apply_history_mask || !seen;
      const bool passes_block_mask = blocked_mask == nullptr || !blocked_mask[sample_base + token];
      const bool passes_topp = (col == 0) ? true : (cumulative <= static_cast<double>(p));
      if (passes_topk && passes_history && passes_block_mask && passes_topp) {
        const float effective_logit =
            static_cast<float>(sample_logits[sample_base + token]) - (seen ? alpha_presence : 0.0f);
        allowed_max = fmaxf(allowed_max, effective_logit);
      }
    }

    double allowed_sum = 0.0;
    cumulative = 0.0;
    for (int64_t col = 0; col < sorted_size; ++col) {
      const float logit = sorted_logits[sorted_base + col];
      const int64_t token = sorted_indices[sorted_base + col];
      const double prob = std::exp(static_cast<double>(logit - row_max)) / row_sum;
      cumulative += prob;
      const bool passes_topk = kth_values == nullptr || logit >= kth_value;
      const bool seen = seen_mask != nullptr && seen_mask[sample_base + token];
      const bool passes_history = !apply_history_mask || !seen;
      const bool passes_block_mask = blocked_mask == nullptr || !blocked_mask[sample_base + token];
      const bool passes_topp = (col == 0) ? true : (cumulative <= static_cast<double>(p));
      if (!passes_topk || !passes_history || !passes_block_mask || !passes_topp) {
        continue;
      }
      const float effective_logit =
          static_cast<float>(sample_logits[sample_base + token]) - (seen ? alpha_presence : 0.0f);
      allowed_sum += std::exp(static_cast<double>(effective_logit - allowed_max));
    }

    const uint64_t seed =
        static_cast<uint64_t>(clock64()) ^
        (static_cast<uint64_t>(row + 1) * 0x9E3779B97F4A7C15ull) ^
        static_cast<uint64_t>(sample_vocab_size) ^
        0xD1B54A32D192ED03ull;
    const double threshold = static_cast<double>(Uniform01FromSeed(seed)) * allowed_sum;
    double sampled = 0.0;
    int64_t selected = sorted_indices[sorted_base];
    cumulative = 0.0;
    for (int64_t col = 0; col < sorted_size; ++col) {
      const float logit = sorted_logits[sorted_base + col];
      const int64_t token = sorted_indices[sorted_base + col];
      const double prob = std::exp(static_cast<double>(logit - row_max)) / row_sum;
      cumulative += prob;
      const bool passes_topk = kth_values == nullptr || logit >= kth_value;
      const bool seen = seen_mask != nullptr && seen_mask[sample_base + token];
      const bool passes_history = !apply_history_mask || !seen;
      const bool passes_block_mask = blocked_mask == nullptr || !blocked_mask[sample_base + token];
      const bool passes_topp = (col == 0) ? true : (cumulative <= static_cast<double>(p));
      if (!passes_topk || !passes_history || !passes_block_mask || !passes_topp) {
        continue;
      }
      const float effective_logit =
          static_cast<float>(sample_logits[sample_base + token]) - (seen ? alpha_presence : 0.0f);
      sampled += std::exp(static_cast<double>(effective_logit - allowed_max));
      selected = token;
      if (sampled >= threshold) {
        break;
      }
    }
    next_token[row] = selected;
  }
}

template <typename scalar_t, int Threads>
__global__ void multinomial_next_token_forward_kernel(
    const scalar_t* __restrict__ logits,
    int64_t* __restrict__ next_token,
    int64_t batch_size,
    int64_t vocab_size) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= batch_size) {
    return;
  }

  __shared__ float shared_values[Threads];

  float row_max = -std::numeric_limits<float>::infinity();
  const int64_t base = row * vocab_size;
  for (int64_t col = threadIdx.x; col < vocab_size; col += Threads) {
    row_max = fmaxf(row_max, static_cast<float>(logits[base + col]));
  }
  shared_values[threadIdx.x] = row_max;
  __syncthreads();

  for (int stride = Threads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_values[threadIdx.x] = fmaxf(shared_values[threadIdx.x], shared_values[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  row_max = shared_values[0];

  float row_sum = 0.0f;
  for (int64_t col = threadIdx.x; col < vocab_size; col += Threads) {
    row_sum += expf(static_cast<float>(logits[base + col]) - row_max);
  }
  shared_values[threadIdx.x] = row_sum;
  __syncthreads();

  for (int stride = Threads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_values[threadIdx.x] += shared_values[threadIdx.x + stride];
    }
    __syncthreads();
  }
  row_sum = shared_values[0];

  if (threadIdx.x == 0) {
    const uint64_t seed =
        static_cast<uint64_t>(clock64()) ^
        (static_cast<uint64_t>(row + 1) * 0x9E3779B97F4A7C15ull) ^
        static_cast<uint64_t>(vocab_size);
    const float threshold = Uniform01FromSeed(seed) * row_sum;
    float cumulative = 0.0f;
    int64_t selected = vocab_size - 1;
    for (int64_t col = 0; col < vocab_size; ++col) {
      cumulative += expf(static_cast<float>(logits[base + col]) - row_max);
      if (cumulative >= threshold) {
        selected = col;
        break;
      }
    }
    next_token[row] = selected;
  }
}

template <typename scalar_t, int Threads>
__global__ void filtered_multinomial_next_token_forward_kernel(
    const scalar_t* __restrict__ logits,
    const bool* __restrict__ seen_mask,
    const bool* __restrict__ blocked_mask,
    const float* __restrict__ kth_values,
    bool apply_history_mask,
    float alpha_presence,
    int64_t* __restrict__ next_token,
    int64_t batch_size,
    int64_t vocab_size) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= batch_size) {
    return;
  }

  __shared__ float shared_values[Threads];
  __shared__ int shared_counts[Threads];

  const int64_t base = row * vocab_size;
  const float kth_value =
      kth_values != nullptr ? kth_values[row] : -std::numeric_limits<float>::infinity();

  float row_max = -std::numeric_limits<float>::infinity();
  int allowed_count = 0;
  for (int64_t col = threadIdx.x; col < vocab_size; col += Threads) {
    const bool passes_topk =
        kth_values == nullptr || static_cast<float>(logits[base + col]) >= kth_value;
    const bool seen = seen_mask != nullptr && seen_mask[base + col];
    const bool passes_history = !apply_history_mask || !seen;
    const bool passes_block_mask = blocked_mask == nullptr || !blocked_mask[base + col];
    if (passes_topk && passes_history && passes_block_mask) {
      const float effective_value =
          static_cast<float>(logits[base + col]) - (seen ? alpha_presence : 0.0f);
      row_max = fmaxf(row_max, effective_value);
      ++allowed_count;
    }
  }
  shared_values[threadIdx.x] = row_max;
  shared_counts[threadIdx.x] = allowed_count;
  __syncthreads();

  for (int stride = Threads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_values[threadIdx.x] = fmaxf(shared_values[threadIdx.x], shared_values[threadIdx.x + stride]);
      shared_counts[threadIdx.x] += shared_counts[threadIdx.x + stride];
    }
    __syncthreads();
  }
  row_max = shared_values[0];
  const int total_allowed = shared_counts[0];

  if (total_allowed == 0) {
    if (threadIdx.x == 0) {
      const uint64_t seed =
          static_cast<uint64_t>(clock64()) ^
          (static_cast<uint64_t>(row + 1) * 0x9E3779B97F4A7C15ull) ^
          static_cast<uint64_t>(vocab_size) ^
          0xA24BAED4963EE407ull;
      const float threshold = Uniform01FromSeed(seed) * static_cast<float>(vocab_size);
      int64_t selected = static_cast<int64_t>(threshold);
      if (selected >= vocab_size) {
        selected = vocab_size - 1;
      }
      next_token[row] = selected;
    }
    return;
  }

  float row_sum = 0.0f;
  for (int64_t col = threadIdx.x; col < vocab_size; col += Threads) {
    const bool passes_topk =
        kth_values == nullptr || static_cast<float>(logits[base + col]) >= kth_value;
    const bool seen = seen_mask != nullptr && seen_mask[base + col];
    const bool passes_history = !apply_history_mask || !seen;
    const bool passes_block_mask = blocked_mask == nullptr || !blocked_mask[base + col];
    if (passes_topk && passes_history && passes_block_mask) {
      const float effective_value =
          static_cast<float>(logits[base + col]) - (seen ? alpha_presence : 0.0f);
      row_sum += expf(effective_value - row_max);
    }
  }
  shared_values[threadIdx.x] = row_sum;
  __syncthreads();

  for (int stride = Threads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_values[threadIdx.x] += shared_values[threadIdx.x + stride];
    }
    __syncthreads();
  }
  row_sum = shared_values[0];

  if (threadIdx.x == 0) {
    const uint64_t seed =
        static_cast<uint64_t>(clock64()) ^
        (static_cast<uint64_t>(row + 1) * 0x9E3779B97F4A7C15ull) ^
        static_cast<uint64_t>(vocab_size) ^
        0x94D049BB133111EBull;
    const float threshold = Uniform01FromSeed(seed) * row_sum;
    float cumulative = 0.0f;
    int64_t selected = vocab_size - 1;
    for (int64_t col = 0; col < vocab_size; ++col) {
      const bool passes_topk =
          kth_values == nullptr || static_cast<float>(logits[base + col]) >= kth_value;
      const bool seen = seen_mask != nullptr && seen_mask[base + col];
      const bool passes_history = !apply_history_mask || !seen;
      const bool passes_block_mask = blocked_mask == nullptr || !blocked_mask[base + col];
      if (!passes_topk || !passes_history || !passes_block_mask) {
        continue;
      }
      const float effective_value =
          static_cast<float>(logits[base + col]) - (seen ? alpha_presence : 0.0f);
      cumulative += expf(effective_value - row_max);
      selected = col;
      if (cumulative >= threshold) {
        break;
      }
    }
    next_token[row] = selected;
  }
}



template <int MaxBeam>
__device__ inline void init_beam_topk(
    float* scores,
    int64_t* tokens,
    int64_t* parents,
    int beam_size,
    int64_t default_token,
    int64_t default_parent) {
  for (int i = 0; i < beam_size; ++i) {
    scores[i] = -std::numeric_limits<float>::infinity();
    tokens[i] = default_token;
    parents[i] = default_parent;
  }
}

template <int MaxBeam>
__device__ inline void insert_beam_topk(
    float score,
    int64_t token,
    int64_t parent,
    float* scores,
    int64_t* tokens,
    int64_t* parents,
    int beam_size) {
  if (beam_size <= 0) {
    return;
  }
  if (score <= scores[beam_size - 1]) {
    return;
  }
  int pos = beam_size - 1;
  while (pos > 0 && score > scores[pos - 1]) {
    scores[pos] = scores[pos - 1];
    tokens[pos] = tokens[pos - 1];
    parents[pos] = parents[pos - 1];
    --pos;
  }
  scores[pos] = score;
  tokens[pos] = token;
  parents[pos] = parent;
}

template <typename scalar_t, int MaxBeam>
__global__ void beam_search_candidates_kernel(
    const scalar_t* __restrict__ logits,
    const float* __restrict__ raw_scores,
    const bool* __restrict__ finished,
    float* __restrict__ cand_scores,
    int64_t* __restrict__ cand_tokens,
    int64_t rows,
    int64_t vocab_size,
    int beam_size,
    int64_t pad_id) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  const int64_t cand_base = row * beam_size;
  if (finished[row]) {
    if (threadIdx.x == 0) {
      cand_scores[cand_base] = raw_scores[row];
      cand_tokens[cand_base] = pad_id;
      for (int i = 1; i < beam_size; ++i) {
        cand_scores[cand_base + i] = -std::numeric_limits<float>::infinity();
        cand_tokens[cand_base + i] = pad_id;
      }
    }
    return;
  }

  __shared__ float reduce_buffer[kBeamSearchThreads];
  const int64_t row_base = row * vocab_size;

  float local_max = -std::numeric_limits<float>::infinity();
  for (int64_t col = threadIdx.x; col < vocab_size; col += blockDim.x) {
    local_max = fmaxf(local_max, static_cast<float>(logits[row_base + col]));
  }
  reduce_buffer[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_buffer[threadIdx.x] = fmaxf(reduce_buffer[threadIdx.x], reduce_buffer[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float row_max = reduce_buffer[0];

  float local_sum = 0.0f;
  for (int64_t col = threadIdx.x; col < vocab_size; col += blockDim.x) {
    local_sum += expf(static_cast<float>(logits[row_base + col]) - row_max);
  }
  reduce_buffer[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_buffer[threadIdx.x] += reduce_buffer[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float log_denom = row_max + logf(fmaxf(reduce_buffer[0], 1.0e-20f));

  if (threadIdx.x == 0) {
    float top_scores[MaxBeam];
    int64_t top_tokens[MaxBeam];
    int64_t top_parents[MaxBeam];
    init_beam_topk<MaxBeam>(top_scores, top_tokens, top_parents, beam_size, pad_id, 0);
    const float base_score = raw_scores[row];
    for (int64_t col = 0; col < vocab_size; ++col) {
      const float score = base_score + static_cast<float>(logits[row_base + col]) - log_denom;
      insert_beam_topk<MaxBeam>(score, col, 0, top_scores, top_tokens, top_parents, beam_size);
    }
    for (int i = 0; i < beam_size; ++i) {
      cand_scores[cand_base + i] = top_scores[i];
      cand_tokens[cand_base + i] = top_tokens[i];
    }
  }
}

template <int MaxBeam>
__global__ void beam_search_merge_kernel(
    const float* __restrict__ cand_scores,
    const int64_t* __restrict__ cand_tokens,
    const bool* __restrict__ finished,
    const int64_t* __restrict__ lengths,
    float* __restrict__ best_scores,
    int64_t* __restrict__ parent_indices,
    int64_t* __restrict__ next_tokens,
    bool* __restrict__ next_finished,
    int64_t* __restrict__ next_lengths,
    int64_t batch_size,
    int beam_size,
    int64_t eos_id,
    int64_t pad_id) {
  const int64_t batch = static_cast<int64_t>(blockIdx.x);
  if (batch >= batch_size || threadIdx.x != 0) {
    return;
  }

  float top_scores[MaxBeam];
  int64_t top_tokens[MaxBeam];
  int64_t top_parents[MaxBeam];
  init_beam_topk<MaxBeam>(top_scores, top_tokens, top_parents, beam_size, pad_id, 0);

  const int64_t batch_base = batch * beam_size;
  for (int parent = 0; parent < beam_size; ++parent) {
    const int64_t cand_base = (batch_base + parent) * beam_size;
    for (int slot = 0; slot < beam_size; ++slot) {
      insert_beam_topk<MaxBeam>(
          cand_scores[cand_base + slot],
          cand_tokens[cand_base + slot],
          parent,
          top_scores,
          top_tokens,
          top_parents,
          beam_size);
    }
  }

  for (int i = 0; i < beam_size; ++i) {
    const int64_t out_idx = batch_base + i;
    const int64_t parent = top_parents[i];
    const bool parent_done = finished[batch_base + parent];
    const int64_t token = top_tokens[i];
    best_scores[out_idx] = top_scores[i];
    parent_indices[out_idx] = parent;
    next_tokens[out_idx] = token;
    next_finished[out_idx] = parent_done || (token == eos_id);
    next_lengths[out_idx] = lengths[batch_base + parent] + (parent_done ? 0 : 1);
  }
}

__global__ void beam_search_gather_append_kernel(
    const int64_t* __restrict__ beams,
    const int64_t* __restrict__ parent_indices,
    const int64_t* __restrict__ next_tokens,
    int64_t* __restrict__ next_beams,
    int64_t batch_size,
    int beam_size,
    int64_t seq_len) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t total_rows = batch_size * beam_size;
  if (row >= total_rows) {
    return;
  }

  const int64_t batch = row / beam_size;
  const int64_t parent = parent_indices[row];
  const int64_t src_row = batch * beam_size + parent;
  const int64_t src_base = src_row * seq_len;
  const int64_t dst_base = row * (seq_len + 1);

  for (int64_t col = threadIdx.x; col < seq_len; col += blockDim.x) {
    next_beams[dst_base + col] = beams[src_base + col];
  }
  if (threadIdx.x == 0) {
    next_beams[dst_base + seq_len] = next_tokens[row];
  }
}

torch::Tensor BuildSeenCounts(
    const torch::Tensor& token_ids_contig,
    int64_t vocab_size) {
  auto counts = torch::zeros(
      {token_ids_contig.size(0), vocab_size},
      torch::TensorOptions().dtype(torch::kFloat32).device(token_ids_contig.device()));
  auto invalid_flags = torch::zeros(
      {2},
      torch::TensorOptions().dtype(torch::kInt32).device(token_ids_contig.device()));
  if (token_ids_contig.numel() == 0) {
    return counts;
  }

  const int64_t total = token_ids_contig.numel();
  const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(token_ids_contig.get_device());
  token_counts_forward_kernel<<<blocks, threads, 0, stream.stream()>>>(
      token_ids_contig.data_ptr<int64_t>(),
      counts.data_ptr<float>(),
      invalid_flags.data_ptr<int32_t>(),
      token_ids_contig.size(0),
      token_ids_contig.size(1),
      vocab_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const auto flags_cpu = invalid_flags.cpu();
  TORCH_CHECK(flags_cpu[0].item<int32_t>() == 0, "CudaTokenCountsForward: token ids must be non-negative");
  TORCH_CHECK(flags_cpu[1].item<int32_t>() == 0, "CudaTokenCountsForward: token id out of range");
  return counts;
}

torch::Tensor BuildSeenMask(
    const torch::Tensor& token_ids_contig,
    int64_t vocab_size) {
  auto seen_mask = torch::zeros(
      {token_ids_contig.size(0), vocab_size},
      torch::TensorOptions().dtype(torch::kBool).device(token_ids_contig.device()));
  auto invalid_flags = torch::zeros(
      {2},
      torch::TensorOptions().dtype(torch::kInt32).device(token_ids_contig.device()));
  if (token_ids_contig.numel() == 0) {
    return seen_mask;
  }

  const dim3 blocks(static_cast<unsigned int>(token_ids_contig.size(0)));
  auto stream = c10::cuda::getCurrentCUDAStream(token_ids_contig.get_device());
  seen_mask_forward_kernel<<<blocks, 1, 0, stream.stream()>>>(
      token_ids_contig.data_ptr<int64_t>(),
      seen_mask.data_ptr<bool>(),
      invalid_flags.data_ptr<int32_t>(),
      token_ids_contig.size(0),
      token_ids_contig.size(1),
      vocab_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const auto flags_cpu = invalid_flags.cpu();
  TORCH_CHECK(flags_cpu[0].item<int32_t>() == 0, "CudaTokenCountsForward: token ids must be non-negative");
  TORCH_CHECK(flags_cpu[1].item<int32_t>() == 0, "CudaTokenCountsForward: token id out of range");
  return seen_mask;
}

torch::Tensor NormalizeTokenIdsLongContiguous(const torch::Tensor& token_ids) {
  if (token_ids.scalar_type() == torch::kLong && token_ids.is_contiguous()) {
    return token_ids;
  }
  return token_ids.to(torch::kLong).contiguous();
}

void ApplyNoRepeatNgramMaskFromContig(
    const torch::Tensor& token_ids_contig,
    int64_t vocab_size,
    int64_t n,
    const torch::Tensor& mask) {
  TORCH_CHECK(mask.is_cuda(), "ApplyNoRepeatNgramMaskFromContig: mask must be a CUDA tensor");
  TORCH_CHECK(mask.scalar_type() == torch::kBool, "ApplyNoRepeatNgramMaskFromContig: mask must be bool");
  TORCH_CHECK(mask.is_contiguous(), "ApplyNoRepeatNgramMaskFromContig: mask must be contiguous");
  TORCH_CHECK(mask.size(0) == token_ids_contig.size(0) && mask.size(1) == vocab_size,
              "ApplyNoRepeatNgramMaskFromContig: mask shape mismatch");
  if (token_ids_contig.numel() == 0 || n <= 0 || token_ids_contig.size(1) < n) {
    return;
  }

  const dim3 blocks(static_cast<unsigned int>(token_ids_contig.size(0)));
  auto stream = c10::cuda::getCurrentCUDAStream(token_ids_contig.get_device());
  no_repeat_ngram_mask_forward_kernel<<<blocks, 1, 0, stream.stream()>>>(
      token_ids_contig.data_ptr<int64_t>(),
      mask.data_ptr<bool>(),
      token_ids_contig.size(0),
      token_ids_contig.size(1),
      vocab_size,
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void LaunchGreedyNextTokenKernel(
    const torch::Tensor& logits_contig,
    torch::Tensor& next_token,
    cudaStream_t stream,
    int row_reduce_threads) {
  const dim3 blocks(static_cast<unsigned int>(logits_contig.size(0)));
  switch (row_reduce_threads) {
    case kSmallRowThreads:
      greedy_next_token_forward_kernel<scalar_t, kSmallRowThreads>
          <<<blocks, kSmallRowThreads, 0, stream>>>(
              logits_contig.data_ptr<scalar_t>(),
              next_token.data_ptr<int64_t>(),
              logits_contig.size(0),
              logits_contig.size(1));
      return;
    case kMediumRowThreads:
      greedy_next_token_forward_kernel<scalar_t, kMediumRowThreads>
          <<<blocks, kMediumRowThreads, 0, stream>>>(
              logits_contig.data_ptr<scalar_t>(),
              next_token.data_ptr<int64_t>(),
              logits_contig.size(0),
              logits_contig.size(1));
      return;
    default:
      greedy_next_token_forward_kernel<scalar_t, kThreads>
          <<<blocks, kThreads, 0, stream>>>(
              logits_contig.data_ptr<scalar_t>(),
              next_token.data_ptr<int64_t>(),
              logits_contig.size(0),
              logits_contig.size(1));
      return;
  }
}

template <typename scalar_t>
void LaunchFilteredGreedyNextTokenKernel(
    const torch::Tensor& logits_contig,
    torch::Tensor& next_token,
    const c10::optional<torch::Tensor>& seen_mask,
    const c10::optional<torch::Tensor>& blocked_mask,
    const c10::optional<torch::Tensor>& kth_values,
    bool apply_history_mask,
    double alpha_presence,
    cudaStream_t stream,
    int row_reduce_threads) {
  const dim3 blocks(static_cast<unsigned int>(logits_contig.size(0)));
  const bool* seen_ptr =
      (seen_mask.has_value() && seen_mask.value().defined())
          ? seen_mask.value().data_ptr<bool>()
          : nullptr;
  const bool* blocked_ptr =
      (blocked_mask.has_value() && blocked_mask.value().defined())
          ? blocked_mask.value().data_ptr<bool>()
          : nullptr;
  const float* kth_ptr =
      (kth_values.has_value() && kth_values.value().defined())
          ? kth_values.value().data_ptr<float>()
          : nullptr;
  switch (row_reduce_threads) {
    case kSmallRowThreads:
      filtered_greedy_next_token_forward_kernel<scalar_t, kSmallRowThreads>
          <<<blocks, kSmallRowThreads, 0, stream>>>(
              logits_contig.data_ptr<scalar_t>(),
              seen_ptr,
              blocked_ptr,
              kth_ptr,
              apply_history_mask,
              static_cast<float>(alpha_presence),
              next_token.data_ptr<int64_t>(),
              logits_contig.size(0),
              logits_contig.size(1));
      return;
    case kMediumRowThreads:
      filtered_greedy_next_token_forward_kernel<scalar_t, kMediumRowThreads>
          <<<blocks, kMediumRowThreads, 0, stream>>>(
              logits_contig.data_ptr<scalar_t>(),
              seen_ptr,
              blocked_ptr,
              kth_ptr,
              apply_history_mask,
              static_cast<float>(alpha_presence),
              next_token.data_ptr<int64_t>(),
              logits_contig.size(0),
              logits_contig.size(1));
      return;
    default:
      filtered_greedy_next_token_forward_kernel<scalar_t, kThreads>
          <<<blocks, kThreads, 0, stream>>>(
              logits_contig.data_ptr<scalar_t>(),
              seen_ptr,
              blocked_ptr,
              kth_ptr,
              apply_history_mask,
              static_cast<float>(alpha_presence),
              next_token.data_ptr<int64_t>(),
              logits_contig.size(0),
              logits_contig.size(1));
      return;
  }
}

template <typename scalar_t>
void LaunchMultinomialNextTokenKernel(
    const torch::Tensor& logits_contig,
    torch::Tensor& next_token,
    cudaStream_t stream,
    int row_reduce_threads) {
  const dim3 blocks(static_cast<unsigned int>(logits_contig.size(0)));
  switch (row_reduce_threads) {
    case kSmallRowThreads:
      multinomial_next_token_forward_kernel<scalar_t, kSmallRowThreads>
          <<<blocks, kSmallRowThreads, 0, stream>>>(
              logits_contig.data_ptr<scalar_t>(),
              next_token.data_ptr<int64_t>(),
              logits_contig.size(0),
              logits_contig.size(1));
      return;
    case kMediumRowThreads:
      multinomial_next_token_forward_kernel<scalar_t, kMediumRowThreads>
          <<<blocks, kMediumRowThreads, 0, stream>>>(
              logits_contig.data_ptr<scalar_t>(),
              next_token.data_ptr<int64_t>(),
              logits_contig.size(0),
              logits_contig.size(1));
      return;
    default:
      multinomial_next_token_forward_kernel<scalar_t, kThreads>
          <<<blocks, kThreads, 0, stream>>>(
              logits_contig.data_ptr<scalar_t>(),
              next_token.data_ptr<int64_t>(),
              logits_contig.size(0),
              logits_contig.size(1));
      return;
  }
}

template <typename scalar_t>
void LaunchFilteredMultinomialNextTokenKernel(
    const torch::Tensor& logits_contig,
    torch::Tensor& next_token,
    const c10::optional<torch::Tensor>& seen_mask,
    const c10::optional<torch::Tensor>& blocked_mask,
    const c10::optional<torch::Tensor>& kth_values,
    bool apply_history_mask,
    double alpha_presence,
    cudaStream_t stream,
    int row_reduce_threads) {
  const dim3 blocks(static_cast<unsigned int>(logits_contig.size(0)));
  const bool* seen_ptr =
      (seen_mask.has_value() && seen_mask.value().defined())
          ? seen_mask.value().data_ptr<bool>()
          : nullptr;
  const bool* blocked_ptr =
      (blocked_mask.has_value() && blocked_mask.value().defined())
          ? blocked_mask.value().data_ptr<bool>()
          : nullptr;
  const float* kth_ptr =
      (kth_values.has_value() && kth_values.value().defined())
          ? kth_values.value().data_ptr<float>()
          : nullptr;
  switch (row_reduce_threads) {
    case kSmallRowThreads:
      filtered_multinomial_next_token_forward_kernel<scalar_t, kSmallRowThreads>
          <<<blocks, kSmallRowThreads, 0, stream>>>(
              logits_contig.data_ptr<scalar_t>(),
              seen_ptr,
              blocked_ptr,
              kth_ptr,
              apply_history_mask,
              static_cast<float>(alpha_presence),
              next_token.data_ptr<int64_t>(),
              logits_contig.size(0),
              logits_contig.size(1));
      return;
    case kMediumRowThreads:
      filtered_multinomial_next_token_forward_kernel<scalar_t, kMediumRowThreads>
          <<<blocks, kMediumRowThreads, 0, stream>>>(
              logits_contig.data_ptr<scalar_t>(),
              seen_ptr,
              blocked_ptr,
              kth_ptr,
              apply_history_mask,
              static_cast<float>(alpha_presence),
              next_token.data_ptr<int64_t>(),
              logits_contig.size(0),
              logits_contig.size(1));
      return;
    default:
      filtered_multinomial_next_token_forward_kernel<scalar_t, kThreads>
          <<<blocks, kThreads, 0, stream>>>(
              logits_contig.data_ptr<scalar_t>(),
              seen_ptr,
              blocked_ptr,
              kth_ptr,
              apply_history_mask,
              static_cast<float>(alpha_presence),
              next_token.data_ptr<int64_t>(),
              logits_contig.size(0),
              logits_contig.size(1));
      return;
  }
}

void LaunchToppMaskKernel(
    const torch::Tensor& sorted_logits,
    const torch::Tensor& sorted_indices,
    const torch::Tensor& mask,
    double p,
    cudaStream_t stream,
    int row_reduce_threads) {
  const dim3 blocks(static_cast<unsigned int>(sorted_logits.size(0)));
  switch (row_reduce_threads) {
    case kSmallRowThreads:
      topp_mask_forward_kernel<kSmallRowThreads>
          <<<blocks, kSmallRowThreads, 0, stream>>>(
              sorted_logits.data_ptr<float>(),
              sorted_indices.data_ptr<int64_t>(),
              mask.data_ptr<bool>(),
              sorted_logits.size(0),
              sorted_logits.size(1),
              static_cast<float>(p));
      return;
    case kMediumRowThreads:
      topp_mask_forward_kernel<kMediumRowThreads>
          <<<blocks, kMediumRowThreads, 0, stream>>>(
              sorted_logits.data_ptr<float>(),
              sorted_indices.data_ptr<int64_t>(),
              mask.data_ptr<bool>(),
              sorted_logits.size(0),
              sorted_logits.size(1),
              static_cast<float>(p));
      return;
    default:
      topp_mask_forward_kernel<kThreads>
          <<<blocks, kThreads, 0, stream>>>(
              sorted_logits.data_ptr<float>(),
              sorted_indices.data_ptr<int64_t>(),
              mask.data_ptr<bool>(),
              sorted_logits.size(0),
              sorted_logits.size(1),
              static_cast<float>(p));
      return;
  }
}

template <typename scalar_t>
void LaunchSortedFilteredSampleKernel(
    const torch::Tensor& sample_logits,
    const torch::Tensor& sorted_logits,
    const torch::Tensor& sorted_indices,
    const torch::Tensor& next_token,
    const c10::optional<torch::Tensor>& seen_mask,
    const c10::optional<torch::Tensor>& blocked_mask,
    const c10::optional<torch::Tensor>& kth_values,
    const c10::optional<torch::Tensor>& row_sums,
    bool apply_history_mask,
    double alpha_presence,
    double p,
    cudaStream_t stream,
    int row_reduce_threads) {
  const dim3 blocks(static_cast<unsigned int>(sorted_logits.size(0)));
  const bool* seen_ptr =
      (seen_mask.has_value() && seen_mask.value().defined())
          ? seen_mask.value().data_ptr<bool>()
          : nullptr;
  const float* kth_ptr =
      (kth_values.has_value() && kth_values.value().defined())
          ? kth_values.value().data_ptr<float>()
          : nullptr;
  const float* row_sums_ptr =
      (row_sums.has_value() && row_sums.value().defined())
          ? row_sums.value().data_ptr<float>()
          : nullptr;
  const bool* blocked_ptr =
      (blocked_mask.has_value() && blocked_mask.value().defined())
          ? blocked_mask.value().data_ptr<bool>()
          : nullptr;
  switch (row_reduce_threads) {
    case kSmallRowThreads:
      sorted_filtered_sample_kernel<scalar_t, kSmallRowThreads>
          <<<blocks, kSmallRowThreads, 0, stream>>>(
              sorted_logits.data_ptr<float>(),
              sorted_indices.data_ptr<int64_t>(),
              sample_logits.data_ptr<scalar_t>(),
              seen_ptr,
              blocked_ptr,
              kth_ptr,
              row_sums_ptr,
              apply_history_mask,
              static_cast<float>(alpha_presence),
              next_token.data_ptr<int64_t>(),
              sorted_logits.size(0),
              sorted_logits.size(1),
              sample_logits.size(1),
              static_cast<float>(p));
      return;
    case kMediumRowThreads:
      sorted_filtered_sample_kernel<scalar_t, kMediumRowThreads>
          <<<blocks, kMediumRowThreads, 0, stream>>>(
              sorted_logits.data_ptr<float>(),
              sorted_indices.data_ptr<int64_t>(),
              sample_logits.data_ptr<scalar_t>(),
              seen_ptr,
              blocked_ptr,
              kth_ptr,
              row_sums_ptr,
              apply_history_mask,
              static_cast<float>(alpha_presence),
              next_token.data_ptr<int64_t>(),
              sorted_logits.size(0),
              sorted_logits.size(1),
              sample_logits.size(1),
              static_cast<float>(p));
      return;
    default:
      sorted_filtered_sample_kernel<scalar_t, kThreads>
          <<<blocks, kThreads, 0, stream>>>(
              sorted_logits.data_ptr<float>(),
              sorted_indices.data_ptr<int64_t>(),
              sample_logits.data_ptr<scalar_t>(),
              seen_ptr,
              blocked_ptr,
              kth_ptr,
              row_sums_ptr,
              apply_history_mask,
              static_cast<float>(alpha_presence),
              next_token.data_ptr<int64_t>(),
              sorted_logits.size(0),
              sorted_logits.size(1),
              sample_logits.size(1),
              static_cast<float>(p));
      return;
  }
}

std::pair<torch::Tensor, torch::Tensor> TopkLogitsWithIndicesDescending(
    const torch::Tensor& logits_contig,
    int64_t k) {
  const auto vocab_size = logits_contig.size(1);
  const auto bounded_k = std::min<int64_t>(k, vocab_size);
  auto topk = torch::topk(logits_contig, bounded_k, -1);
  return {
      std::get<0>(topk).to(torch::kFloat32).contiguous(),
      std::get<1>(topk).contiguous()};
}

torch::Tensor ComputeSoftmaxRowSumsFromMax(
    const torch::Tensor& logits_contig,
    const torch::Tensor& row_max) {
  auto logits_f = logits_contig.to(torch::kFloat32);
  return torch::exp(logits_f - row_max).sum(1).contiguous();
}

std::pair<torch::Tensor, torch::Tensor> SortLogitsWithIndicesDescending(
    const torch::Tensor& logits_contig) {
  auto sorted_logits = logits_contig.to(torch::kFloat32);
  const auto batch_size = logits_contig.size(0);
  const auto vocab_size = logits_contig.size(1);
  auto stream = c10::cuda::getCurrentCUDAStream(logits_contig.get_device());
  auto sorted_indices = torch::empty(
      {batch_size, vocab_size},
      torch::TensorOptions().dtype(torch::kLong).device(logits_contig.device()));
  const auto total = batch_size * vocab_size;
  const dim3 fill_blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads));
  const dim3 fill_threads(kThreads);
  fill_row_indices_kernel<<<fill_blocks, fill_threads, 0, stream.stream()>>>(
      sorted_indices.data_ptr<int64_t>(),
      batch_size,
      vocab_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto policy = thrust::cuda::par.on(stream.stream());
  auto* sorted_logits_ptr = sorted_logits.data_ptr<float>();
  auto* sorted_indices_ptr = sorted_indices.data_ptr<int64_t>();
  for (int64_t row = 0; row < batch_size; ++row) {
    thrust::device_ptr<float> key_begin(sorted_logits_ptr + (row * vocab_size));
    thrust::device_ptr<int64_t> value_begin(sorted_indices_ptr + (row * vocab_size));
    thrust::sort_by_key(policy, key_begin, key_begin + vocab_size, value_begin, thrust::greater<float>());
  }
  return {sorted_logits, sorted_indices};
}

torch::Tensor ComputeTopkThresholds(
    const torch::Tensor& logits_contig,
    int64_t k) {
  const auto vocab_size = logits_contig.size(1);
  const auto bounded_k = std::min<int64_t>(k, vocab_size);
  auto topk = std::get<0>(torch::topk(logits_contig, bounded_k, /*dim=*/-1));
  return topk.to(torch::kFloat32).select(1, bounded_k - 1).contiguous();
}

void ApplyTopkMaskWithThresholdInplace(
    const torch::Tensor& logits_contig,
    const torch::Tensor& kth_values,
    const torch::Tensor& mask) {
  TORCH_CHECK(mask.is_cuda(), "ApplyTopkMaskWithThresholdInplace: mask must be a CUDA tensor");
  TORCH_CHECK(mask.scalar_type() == torch::kBool, "ApplyTopkMaskWithThresholdInplace: mask must be bool");
  TORCH_CHECK(mask.is_contiguous(), "ApplyTopkMaskWithThresholdInplace: mask must be contiguous");
  TORCH_CHECK(kth_values.is_cuda(), "ApplyTopkMaskWithThresholdInplace: kth_values must be a CUDA tensor");
  TORCH_CHECK(kth_values.scalar_type() == torch::kFloat32,
              "ApplyTopkMaskWithThresholdInplace: kth_values must be float32");
  TORCH_CHECK(kth_values.dim() == 1 && kth_values.size(0) == logits_contig.size(0),
              "ApplyTopkMaskWithThresholdInplace: kth_values shape mismatch");
  TORCH_CHECK(mask.sizes() == logits_contig.sizes(),
              "ApplyTopkMaskWithThresholdInplace: mask shape mismatch");

  const auto batch_size = logits_contig.size(0);
  const auto vocab_size = logits_contig.size(1);
  const auto total = batch_size * vocab_size;
  const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(logits_contig.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      logits_contig.scalar_type(),
      "model_stack_cuda_topk_mask_forward",
      [&] {
        topk_mask_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            logits_contig.data_ptr<scalar_t>(),
            kth_values.data_ptr<float>(),
            mask.data_ptr<bool>(),
            batch_size,
            vocab_size);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ApplyToppMaskFromSortedInplace(
    const torch::Tensor& sorted_logits,
    const torch::Tensor& sorted_indices,
    double p,
    const torch::Tensor& mask) {
  TORCH_CHECK(sorted_logits.is_cuda(), "ApplyToppMaskFromSortedInplace: sorted_logits must be a CUDA tensor");
  TORCH_CHECK(sorted_indices.is_cuda(), "ApplyToppMaskFromSortedInplace: sorted_indices must be a CUDA tensor");
  TORCH_CHECK(sorted_logits.scalar_type() == torch::kFloat32,
              "ApplyToppMaskFromSortedInplace: sorted_logits must be float32");
  TORCH_CHECK(sorted_indices.scalar_type() == torch::kLong,
              "ApplyToppMaskFromSortedInplace: sorted_indices must be int64");
  TORCH_CHECK(mask.is_cuda(), "ApplyToppMaskFromSortedInplace: mask must be a CUDA tensor");
  TORCH_CHECK(mask.scalar_type() == torch::kBool, "ApplyToppMaskFromSortedInplace: mask must be bool");
  TORCH_CHECK(mask.is_contiguous(), "ApplyToppMaskFromSortedInplace: mask must be contiguous");
  TORCH_CHECK(sorted_logits.sizes() == sorted_indices.sizes(),
              "ApplyToppMaskFromSortedInplace: sorted tensor shape mismatch");
  TORCH_CHECK(mask.sizes() == sorted_logits.sizes(),
              "ApplyToppMaskFromSortedInplace: mask shape mismatch");

  const auto batch_size = sorted_logits.size(0);
  auto stream = c10::cuda::getCurrentCUDAStream(sorted_logits.get_device());
  const auto policy = SelectSamplingLaunchPolicy(sorted_logits.size(1));
  LaunchToppMaskKernel(
      sorted_logits,
      sorted_indices,
      mask,
      p,
      stream.stream(),
      policy.row_reduce_threads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

bool HasCudaSamplingKernel() {
  return true;
}

torch::Tensor CudaTemperatureForward(
    const torch::Tensor& logits,
    double tau) {
  TORCH_CHECK(logits.is_cuda(), "CudaTemperatureForward: logits must be a CUDA tensor");
  TORCH_CHECK(std::isfinite(tau), "CudaTemperatureForward: tau must be finite");

  c10::cuda::CUDAGuard device_guard{logits.device()};

  auto logits_contig = logits.contiguous();
  auto out = torch::empty_like(logits_contig);
  const auto numel = logits_contig.numel();
  const dim3 blocks(static_cast<unsigned int>((numel + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  const float inv_tau = 1.0f / static_cast<float>(std::max(tau, 1e-8));
  auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      logits_contig.scalar_type(),
      "model_stack_cuda_temperature_forward",
      [&] {
        temperature_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            logits_contig.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            numel,
            inv_tau);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.view(logits.sizes());
}

torch::Tensor CudaPresenceFrequencyPenaltyForward(
    const torch::Tensor& logits,
    const torch::Tensor& counts,
    double alpha_presence,
    double alpha_frequency) {
  TORCH_CHECK(logits.is_cuda() && counts.is_cuda(),
              "CudaPresenceFrequencyPenaltyForward: logits and counts must be CUDA tensors");
  TORCH_CHECK(logits.sizes() == counts.sizes(),
              "CudaPresenceFrequencyPenaltyForward: logits/counts shape mismatch");
  TORCH_CHECK(logits.scalar_type() == counts.scalar_type(),
              "CudaPresenceFrequencyPenaltyForward: logits/counts dtype mismatch");

  c10::cuda::CUDAGuard device_guard{logits.device()};

  auto logits_contig = logits.contiguous();
  auto counts_contig = counts.contiguous();
  auto out = torch::empty_like(logits_contig);
  const auto numel = logits_contig.numel();
  const dim3 blocks(static_cast<unsigned int>((numel + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      logits_contig.scalar_type(),
      "model_stack_cuda_presence_frequency_penalty_forward",
      [&] {
        presence_frequency_penalty_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            logits_contig.data_ptr<scalar_t>(),
            counts_contig.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            numel,
            static_cast<float>(alpha_presence),
            static_cast<float>(alpha_frequency));
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.view(logits.sizes());
}

torch::Tensor CudaApplySamplingMaskForward(
    const torch::Tensor& logits,
    const c10::optional<torch::Tensor>& topk_mask,
    const c10::optional<torch::Tensor>& topp_mask,
    const c10::optional<torch::Tensor>& no_repeat_mask) {
  TORCH_CHECK(logits.is_cuda(), "CudaApplySamplingMaskForward: logits must be a CUDA tensor");

  c10::cuda::CUDAGuard device_guard{logits.device()};

  auto logits_contig = logits.contiguous();
  auto out = torch::empty_like(logits_contig);
  const auto numel = logits_contig.numel();
  const dim3 blocks(static_cast<unsigned int>((numel + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device());

  const torch::Tensor topk_mask_tensor =
      (topk_mask.has_value() && topk_mask.value().defined()) ? topk_mask.value().contiguous() : torch::Tensor();
  const torch::Tensor topp_mask_tensor =
      (topp_mask.has_value() && topp_mask.value().defined()) ? topp_mask.value().contiguous() : torch::Tensor();
  const torch::Tensor no_repeat_mask_tensor =
      (no_repeat_mask.has_value() && no_repeat_mask.value().defined()) ? no_repeat_mask.value().contiguous() : torch::Tensor();

  if (topk_mask_tensor.defined()) {
    TORCH_CHECK(topk_mask_tensor.is_cuda(), "CudaApplySamplingMaskForward: topk_mask must be CUDA");
    TORCH_CHECK(topk_mask_tensor.sizes() == logits.sizes(), "CudaApplySamplingMaskForward: topk_mask shape mismatch");
    TORCH_CHECK(topk_mask_tensor.scalar_type() == torch::kBool, "CudaApplySamplingMaskForward: topk_mask must be bool");
  }
  if (topp_mask_tensor.defined()) {
    TORCH_CHECK(topp_mask_tensor.is_cuda(), "CudaApplySamplingMaskForward: topp_mask must be CUDA");
    TORCH_CHECK(topp_mask_tensor.sizes() == logits.sizes(), "CudaApplySamplingMaskForward: topp_mask shape mismatch");
    TORCH_CHECK(topp_mask_tensor.scalar_type() == torch::kBool, "CudaApplySamplingMaskForward: topp_mask must be bool");
  }
  if (no_repeat_mask_tensor.defined()) {
    TORCH_CHECK(no_repeat_mask_tensor.is_cuda(), "CudaApplySamplingMaskForward: no_repeat_mask must be CUDA");
    TORCH_CHECK(no_repeat_mask_tensor.sizes() == logits.sizes(),
                "CudaApplySamplingMaskForward: no_repeat_mask shape mismatch");
    TORCH_CHECK(no_repeat_mask_tensor.scalar_type() == torch::kBool,
                "CudaApplySamplingMaskForward: no_repeat_mask must be bool");
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      logits_contig.scalar_type(),
      "model_stack_cuda_apply_sampling_mask_forward",
      [&] {
        const scalar_t fill_value = std::numeric_limits<scalar_t>::lowest();
        apply_sampling_mask_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            logits_contig.data_ptr<scalar_t>(),
            topk_mask_tensor.defined() ? topk_mask_tensor.data_ptr<bool>() : nullptr,
            topp_mask_tensor.defined() ? topp_mask_tensor.data_ptr<bool>() : nullptr,
            no_repeat_mask_tensor.defined() ? no_repeat_mask_tensor.data_ptr<bool>() : nullptr,
            out.data_ptr<scalar_t>(),
            numel,
            fill_value);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.view(logits.sizes());
}

torch::Tensor CudaTopkMaskForward(
    const torch::Tensor& logits,
    int64_t k) {
  TORCH_CHECK(logits.is_cuda(), "CudaTopkMaskForward: logits must be a CUDA tensor");
  TORCH_CHECK(logits.dim() == 2, "CudaTopkMaskForward: logits must be rank-2 (B, V)");
  TORCH_CHECK(k > 0, "CudaTopkMaskForward: k must be positive");

  c10::cuda::CUDAGuard device_guard{logits.device()};

  auto logits_contig = logits.contiguous();
  const auto batch_size = logits_contig.size(0);
  const auto vocab_size = logits_contig.size(1);
  auto mask = torch::zeros(
      {batch_size, vocab_size},
      torch::TensorOptions().dtype(torch::kBool).device(logits.device()));
  auto kth_values = ComputeTopkThresholds(logits_contig, k);
  ApplyTopkMaskWithThresholdInplace(logits_contig, kth_values, mask);
  return mask;
}

void CudaTopkMaskForwardInplace(
    const torch::Tensor& logits,
    int64_t k,
    const torch::Tensor& mask) {
  TORCH_CHECK(logits.is_cuda(), "CudaTopkMaskForwardInplace: logits must be a CUDA tensor");
  TORCH_CHECK(logits.dim() == 2, "CudaTopkMaskForwardInplace: logits must be rank-2 (B, V)");
  TORCH_CHECK(k > 0, "CudaTopkMaskForwardInplace: k must be positive");
  TORCH_CHECK(mask.is_cuda(), "CudaTopkMaskForwardInplace: mask must be a CUDA tensor");
  TORCH_CHECK(mask.scalar_type() == torch::kBool, "CudaTopkMaskForwardInplace: mask must be bool");
  TORCH_CHECK(mask.is_contiguous(), "CudaTopkMaskForwardInplace: mask must be contiguous");

  c10::cuda::CUDAGuard device_guard{logits.device()};

  auto logits_contig = logits.contiguous();
  TORCH_CHECK(mask.sizes() == logits_contig.sizes(),
              "CudaTopkMaskForwardInplace: mask shape mismatch");

  auto kth_values = ComputeTopkThresholds(logits_contig, k);
  ApplyTopkMaskWithThresholdInplace(logits_contig, kth_values, mask);
}

torch::Tensor CudaToppMaskForward(
    const torch::Tensor& logits,
    double p) {
  TORCH_CHECK(logits.is_cuda(), "CudaToppMaskForward: logits must be a CUDA tensor");
  TORCH_CHECK(logits.dim() == 2, "CudaToppMaskForward: logits must be rank-2 (B, V)");

  c10::cuda::CUDAGuard device_guard{logits.device()};

  auto logits_contig = logits.contiguous();
  auto [sorted_logits, sorted_indices] = SortLogitsWithIndicesDescending(logits_contig);
  auto mask = torch::zeros(
      {logits_contig.size(0), logits_contig.size(1)},
      torch::TensorOptions().dtype(torch::kBool).device(logits.device()));
  ApplyToppMaskFromSortedInplace(sorted_logits, sorted_indices, p, mask);
  return mask;
}

void CudaToppMaskForwardInplace(
    const torch::Tensor& logits,
    double p,
    const torch::Tensor& mask) {
  TORCH_CHECK(logits.is_cuda(), "CudaToppMaskForwardInplace: logits must be a CUDA tensor");
  TORCH_CHECK(logits.dim() == 2, "CudaToppMaskForwardInplace: logits must be rank-2 (B, V)");
  TORCH_CHECK(mask.is_cuda(), "CudaToppMaskForwardInplace: mask must be a CUDA tensor");
  TORCH_CHECK(mask.scalar_type() == torch::kBool, "CudaToppMaskForwardInplace: mask must be bool");
  TORCH_CHECK(mask.is_contiguous(), "CudaToppMaskForwardInplace: mask must be contiguous");

  c10::cuda::CUDAGuard device_guard{logits.device()};

  auto logits_contig = logits.contiguous();
  TORCH_CHECK(mask.sizes() == logits_contig.sizes(),
              "CudaToppMaskForwardInplace: mask shape mismatch");
  auto [sorted_logits, sorted_indices] = SortLogitsWithIndicesDescending(logits_contig);
  ApplyToppMaskFromSortedInplace(sorted_logits, sorted_indices, p, mask);
}

torch::Tensor CudaTokenCountsForward(
    const torch::Tensor& token_ids,
    int64_t vocab_size,
    torch::ScalarType counts_dtype) {
  TORCH_CHECK(token_ids.is_cuda(), "CudaTokenCountsForward: token_ids must be a CUDA tensor");
  TORCH_CHECK(token_ids.dim() == 2, "CudaTokenCountsForward: token_ids must be rank-2");
  TORCH_CHECK(vocab_size > 0, "CudaTokenCountsForward: vocab_size must be positive");

  c10::cuda::CUDAGuard device_guard{token_ids.device()};

  auto token_ids_contig = NormalizeTokenIdsLongContiguous(token_ids);
  auto counts = BuildSeenCounts(token_ids_contig, vocab_size);
  return counts.to(torch::TensorOptions().dtype(counts_dtype).device(token_ids.device()));
}

torch::Tensor CudaNoRepeatNgramMaskForward(
    const torch::Tensor& token_ids,
    int64_t vocab_size,
    int64_t n) {
  TORCH_CHECK(token_ids.is_cuda(), "CudaNoRepeatNgramMaskForward: token_ids must be a CUDA tensor");
  TORCH_CHECK(token_ids.dim() == 2, "CudaNoRepeatNgramMaskForward: token_ids must be rank-2 (B, T)");
  TORCH_CHECK(vocab_size > 0, "CudaNoRepeatNgramMaskForward: vocab_size must be positive");

  c10::cuda::CUDAGuard device_guard{token_ids.device()};

  auto token_ids_contig = NormalizeTokenIdsLongContiguous(token_ids);
  auto mask = torch::zeros(
      {token_ids_contig.size(0), vocab_size},
      torch::TensorOptions().dtype(torch::kBool).device(token_ids.device()));
  ApplyNoRepeatNgramMaskFromContig(token_ids_contig, vocab_size, n, mask);
  return mask;
}

void CudaNoRepeatNgramMaskForwardInplace(
    const torch::Tensor& token_ids,
    int64_t vocab_size,
    int64_t n,
    const torch::Tensor& mask) {
  TORCH_CHECK(token_ids.is_cuda(), "CudaNoRepeatNgramMaskForwardInplace: token_ids must be a CUDA tensor");
  TORCH_CHECK(token_ids.dim() == 2, "CudaNoRepeatNgramMaskForwardInplace: token_ids must be rank-2 (B, T)");
  TORCH_CHECK(vocab_size > 0, "CudaNoRepeatNgramMaskForwardInplace: vocab_size must be positive");
  TORCH_CHECK(mask.is_cuda(), "CudaNoRepeatNgramMaskForwardInplace: mask must be a CUDA tensor");
  TORCH_CHECK(mask.scalar_type() == torch::kBool, "CudaNoRepeatNgramMaskForwardInplace: mask must be bool");
  TORCH_CHECK(mask.is_contiguous(), "CudaNoRepeatNgramMaskForwardInplace: mask must be contiguous");

  c10::cuda::CUDAGuard device_guard{token_ids.device()};

  auto token_ids_contig = NormalizeTokenIdsLongContiguous(token_ids);
  ApplyNoRepeatNgramMaskFromContig(token_ids_contig, vocab_size, n, mask);
}

torch::Tensor CudaRepetitionPenaltyForward(
    const torch::Tensor& logits,
    const torch::Tensor& token_ids,
    double penalty) {
  TORCH_CHECK(logits.is_cuda() && token_ids.is_cuda(),
              "CudaRepetitionPenaltyForward: logits and token_ids must be CUDA tensors");
  TORCH_CHECK(logits.dim() == 2, "CudaRepetitionPenaltyForward: logits must be rank-2 (B, V)");
  TORCH_CHECK(token_ids.dim() == 2, "CudaRepetitionPenaltyForward: token_ids must be rank-2 (B, T)");
  TORCH_CHECK(logits.size(0) == token_ids.size(0), "CudaRepetitionPenaltyForward: batch mismatch");
  TORCH_CHECK(std::isfinite(penalty) && penalty > 0.0,
              "CudaRepetitionPenaltyForward: penalty must be positive and finite");

  c10::cuda::CUDAGuard device_guard{logits.device()};

  auto logits_contig = logits.contiguous();
  auto out = torch::empty_like(logits_contig);
  auto token_ids_contig = NormalizeTokenIdsLongContiguous(token_ids);
  auto seen_mask = BuildSeenMask(token_ids_contig, logits_contig.size(1));
  const auto numel = logits_contig.numel();
  const dim3 blocks(static_cast<unsigned int>((numel + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      logits_contig.scalar_type(),
      "model_stack_cuda_repetition_penalty_forward",
      [&] {
        repetition_penalty_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            logits_contig.data_ptr<scalar_t>(),
            seen_mask.data_ptr<bool>(),
            out.data_ptr<scalar_t>(),
            numel,
            static_cast<float>(penalty));
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.view(logits.sizes());
}

torch::Tensor CudaGreedyNextTokenForward(const torch::Tensor& logits) {
  TORCH_CHECK(logits.is_cuda(), "CudaGreedyNextTokenForward: logits must be a CUDA tensor");
  TORCH_CHECK(logits.dim() == 2, "CudaGreedyNextTokenForward: logits must be rank-2 (B, V)");
  TORCH_CHECK(logits.size(1) > 0, "CudaGreedyNextTokenForward: vocab dimension must be non-empty");

  c10::cuda::CUDAGuard device_guard{logits.device()};

  auto logits_contig = logits.contiguous();
  auto next_token = torch::empty(
      {logits_contig.size(0), 1},
      torch::TensorOptions().dtype(torch::kLong).device(logits.device()));
  const auto policy = SelectSamplingLaunchPolicy(logits_contig.size(1));
  auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      logits_contig.scalar_type(),
      "model_stack_cuda_greedy_next_token_forward",
      [&] {
        LaunchGreedyNextTokenKernel<scalar_t>(
            logits_contig,
            next_token,
            stream.stream(),
            policy.row_reduce_threads);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return next_token;
}

torch::Tensor CudaMultinomialSampleForward(const torch::Tensor& logits) {
  TORCH_CHECK(logits.is_cuda(), "CudaMultinomialSampleForward: logits must be a CUDA tensor");
  TORCH_CHECK(logits.dim() == 2, "CudaMultinomialSampleForward: logits must be rank-2 (B, V)");
  TORCH_CHECK(logits.size(1) > 0, "CudaMultinomialSampleForward: vocab dimension must be non-empty");

  c10::cuda::CUDAGuard device_guard{logits.device()};

  auto logits_contig = logits.contiguous();
  auto next_token = torch::empty(
      {logits_contig.size(0), 1},
      torch::TensorOptions().dtype(torch::kLong).device(logits.device()));
  const auto policy = SelectSamplingLaunchPolicy(logits_contig.size(1));
  auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      logits_contig.scalar_type(),
      "model_stack_cuda_multinomial_next_token_forward",
      [&] {
        LaunchMultinomialNextTokenKernel<scalar_t>(
            logits_contig,
            next_token,
            stream.stream(),
            policy.row_reduce_threads);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return next_token;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CudaSpeculativeAcceptForward(
    const torch::Tensor& target_probs,
    const torch::Tensor& draft_probs,
    const torch::Tensor& draft_token_ids,
    const c10::optional<torch::Tensor>& bonus_probs,
    const c10::optional<torch::Tensor>& bonus_enabled,
    const std::string& method,
    double posterior_threshold,
    double posterior_alpha) {
  TORCH_CHECK(target_probs.is_cuda() && draft_probs.is_cuda() && draft_token_ids.is_cuda(),
              "CudaSpeculativeAcceptForward: target_probs, draft_probs, and draft_token_ids must be CUDA tensors");
  TORCH_CHECK(target_probs.dim() == 3, "CudaSpeculativeAcceptForward: target_probs must be rank-3 (B, K, V)");
  TORCH_CHECK(draft_probs.dim() == 3, "CudaSpeculativeAcceptForward: draft_probs must be rank-3 (B, K, V)");
  TORCH_CHECK(draft_token_ids.dim() == 2, "CudaSpeculativeAcceptForward: draft_token_ids must be rank-2 (B, K)");
  TORCH_CHECK(target_probs.sizes() == draft_probs.sizes(),
              "CudaSpeculativeAcceptForward: target_probs and draft_probs shape mismatch");
  TORCH_CHECK(target_probs.size(0) == draft_token_ids.size(0) && target_probs.size(1) == draft_token_ids.size(1),
              "CudaSpeculativeAcceptForward: draft token shape mismatch");

  c10::cuda::CUDAGuard device_guard{target_probs.device()};

  auto target_probs_contig = target_probs.to(torch::kFloat32).contiguous();
  auto draft_probs_contig = draft_probs.to(torch::kFloat32).contiguous();
  auto draft_token_ids_contig = NormalizeTokenIdsLongContiguous(draft_token_ids);
  torch::Tensor bonus_probs_contig;
  if (bonus_probs.has_value() && bonus_probs.value().defined()) {
    TORCH_CHECK(bonus_probs.value().is_cuda(), "CudaSpeculativeAcceptForward: bonus_probs must be a CUDA tensor");
    TORCH_CHECK(bonus_probs.value().dim() == 2, "CudaSpeculativeAcceptForward: bonus_probs must be rank-2 (B, V)");
    TORCH_CHECK(bonus_probs.value().size(0) == target_probs.size(0) && bonus_probs.value().size(1) == target_probs.size(2),
                "CudaSpeculativeAcceptForward: bonus_probs shape mismatch");
    bonus_probs_contig = bonus_probs.value().to(torch::kFloat32).contiguous();
  }
  torch::Tensor bonus_enabled_contig;
  if (bonus_enabled.has_value() && bonus_enabled.value().defined()) {
    TORCH_CHECK(bonus_enabled.value().is_cuda(), "CudaSpeculativeAcceptForward: bonus_enabled must be a CUDA tensor");
    TORCH_CHECK(bonus_enabled.value().numel() == target_probs.size(0),
                "CudaSpeculativeAcceptForward: bonus_enabled must have one flag per batch row");
    bonus_enabled_contig = bonus_enabled.value().to(torch::kBool).contiguous().view({-1});
  }

  int method_code = kSpeculativeAcceptRejectionSampler;
  std::string method_name = method;
  std::transform(method_name.begin(), method_name.end(), method_name.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (method_name.empty() || method_name == "rejection_sampler" || method_name == "probabilistic" || method_name == "rejection") {
    method_code = kSpeculativeAcceptRejectionSampler;
  } else if (method_name == "strict") {
    method_code = kSpeculativeAcceptStrict;
  } else if (method_name == "typical_acceptance_sampler" || method_name == "typical") {
    method_code = kSpeculativeAcceptTypical;
  } else {
    TORCH_CHECK(false, "CudaSpeculativeAcceptForward: unsupported method ", method);
  }

  const auto batch_size = target_probs_contig.size(0);
  const auto draft_len = target_probs_contig.size(1);
  const auto vocab_size = target_probs_contig.size(2);
  auto out_tokens = torch::full(
      {batch_size, draft_len + 1},
      -1,
      torch::TensorOptions().dtype(torch::kLong).device(target_probs.device()));
  auto out_lengths = torch::zeros(
      {batch_size},
      torch::TensorOptions().dtype(torch::kLong).device(target_probs.device()));
  auto accepted_counts = torch::zeros(
      {batch_size},
      torch::TensorOptions().dtype(torch::kLong).device(target_probs.device()));
  if (batch_size == 0) {
    return std::make_tuple(out_tokens, out_lengths, accepted_counts);
  }

  const auto policy = SelectSamplingLaunchPolicy(vocab_size);
  auto stream = c10::cuda::getCurrentCUDAStream(target_probs.get_device());
  switch (policy.row_reduce_threads) {
    case kSmallRowThreads:
      speculative_accept_forward_kernel<kSmallRowThreads><<<static_cast<unsigned int>(batch_size), kSmallRowThreads, 0, stream.stream()>>>(
          target_probs_contig.data_ptr<float>(),
          draft_probs_contig.data_ptr<float>(),
          draft_token_ids_contig.data_ptr<int64_t>(),
          bonus_probs_contig.defined() ? bonus_probs_contig.data_ptr<float>() : nullptr,
          bonus_enabled_contig.defined() ? bonus_enabled_contig.data_ptr<bool>() : nullptr,
          out_tokens.data_ptr<int64_t>(),
          out_lengths.data_ptr<int64_t>(),
          accepted_counts.data_ptr<int64_t>(),
          batch_size,
          draft_len,
          vocab_size,
          method_code,
          static_cast<float>(posterior_threshold),
          static_cast<float>(posterior_alpha));
      break;
    case kMediumRowThreads:
      speculative_accept_forward_kernel<kMediumRowThreads><<<static_cast<unsigned int>(batch_size), kMediumRowThreads, 0, stream.stream()>>>(
          target_probs_contig.data_ptr<float>(),
          draft_probs_contig.data_ptr<float>(),
          draft_token_ids_contig.data_ptr<int64_t>(),
          bonus_probs_contig.defined() ? bonus_probs_contig.data_ptr<float>() : nullptr,
          bonus_enabled_contig.defined() ? bonus_enabled_contig.data_ptr<bool>() : nullptr,
          out_tokens.data_ptr<int64_t>(),
          out_lengths.data_ptr<int64_t>(),
          accepted_counts.data_ptr<int64_t>(),
          batch_size,
          draft_len,
          vocab_size,
          method_code,
          static_cast<float>(posterior_threshold),
          static_cast<float>(posterior_alpha));
      break;
    default:
      speculative_accept_forward_kernel<kThreads><<<static_cast<unsigned int>(batch_size), kThreads, 0, stream.stream()>>>(
          target_probs_contig.data_ptr<float>(),
          draft_probs_contig.data_ptr<float>(),
          draft_token_ids_contig.data_ptr<int64_t>(),
          bonus_probs_contig.defined() ? bonus_probs_contig.data_ptr<float>() : nullptr,
          bonus_enabled_contig.defined() ? bonus_enabled_contig.data_ptr<bool>() : nullptr,
          out_tokens.data_ptr<int64_t>(),
          out_lengths.data_ptr<int64_t>(),
          accepted_counts.data_ptr<int64_t>(),
          batch_size,
          draft_len,
          vocab_size,
          method_code,
          static_cast<float>(posterior_threshold),
          static_cast<float>(posterior_alpha));
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(out_tokens, out_lengths, accepted_counts);
}


std::vector<torch::Tensor> CudaBeamSearchStepForward(
    const torch::Tensor& beams,
    const torch::Tensor& logits,
    const torch::Tensor& raw_scores,
    const torch::Tensor& finished,
    const torch::Tensor& lengths,
    int64_t beam_size,
    int64_t eos_id,
    int64_t pad_id) {
  TORCH_CHECK(beams.is_cuda() && logits.is_cuda() && raw_scores.is_cuda() && finished.is_cuda() && lengths.is_cuda(),
              "CudaBeamSearchStepForward: all tensors must be CUDA tensors");
  TORCH_CHECK(beams.dim() == 2, "CudaBeamSearchStepForward: beams must be rank-2 (B*beam, T)");
  TORCH_CHECK(logits.dim() == 2, "CudaBeamSearchStepForward: logits must be rank-2 (B*beam, V)");
  TORCH_CHECK(raw_scores.dim() == 2, "CudaBeamSearchStepForward: raw_scores must be rank-2 (B, beam)");
  TORCH_CHECK(finished.dim() == 2, "CudaBeamSearchStepForward: finished must be rank-2 (B, beam)");
  TORCH_CHECK(lengths.dim() == 2, "CudaBeamSearchStepForward: lengths must be rank-2 (B, beam)");
  TORCH_CHECK(beam_size > 0, "CudaBeamSearchStepForward: beam_size must be positive");
  TORCH_CHECK(beam_size <= kMaxBeamSearchBeamSize,
              "CudaBeamSearchStepForward: beam_size exceeds CUDA kernel limit of ",
              kMaxBeamSearchBeamSize);
  TORCH_CHECK(raw_scores.size(1) == beam_size, "CudaBeamSearchStepForward: raw_scores beam dimension mismatch");
  TORCH_CHECK(finished.sizes() == raw_scores.sizes(), "CudaBeamSearchStepForward: finished shape mismatch");
  TORCH_CHECK(lengths.sizes() == raw_scores.sizes(), "CudaBeamSearchStepForward: lengths shape mismatch");
  TORCH_CHECK(logits.size(0) == raw_scores.size(0) * beam_size, "CudaBeamSearchStepForward: logits batch mismatch");
  TORCH_CHECK(beams.size(0) == logits.size(0), "CudaBeamSearchStepForward: beams/logits batch mismatch");
  TORCH_CHECK(logits.size(1) > 0, "CudaBeamSearchStepForward: vocab dimension must be non-empty");
  TORCH_CHECK(pad_id >= 0 && pad_id < logits.size(1), "CudaBeamSearchStepForward: pad_id out of range");

  c10::cuda::CUDAGuard device_guard{logits.device()};

  auto beams_contig = NormalizeTokenIdsLongContiguous(beams);
  auto logits_contig = logits.contiguous();
  auto raw_scores_contig = raw_scores.to(torch::kFloat32).contiguous();
  auto finished_contig = finished.to(torch::kBool).contiguous();
  auto lengths_contig = lengths.to(torch::kLong).contiguous();

  const auto batch_size = raw_scores_contig.size(0);
  const auto rows = batch_size * beam_size;
  const auto seq_len = beams_contig.size(1);
  const auto vocab_size = logits_contig.size(1);
  auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device());

  auto cand_scores = torch::full(
      {rows, beam_size},
      -std::numeric_limits<float>::infinity(),
      torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));
  auto cand_tokens = torch::full(
      {rows, beam_size},
      pad_id,
      torch::TensorOptions().dtype(torch::kLong).device(logits.device()));

  const dim3 candidate_blocks(static_cast<unsigned int>(rows));
  const dim3 candidate_threads(kBeamSearchThreads);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      logits_contig.scalar_type(),
      "model_stack_cuda_beam_search_candidates",
      [&] {
        beam_search_candidates_kernel<scalar_t, kMaxBeamSearchBeamSize><<<candidate_blocks, candidate_threads, 0, stream.stream()>>>(
            logits_contig.data_ptr<scalar_t>(),
            raw_scores_contig.data_ptr<float>(),
            finished_contig.data_ptr<bool>(),
            cand_scores.data_ptr<float>(),
            cand_tokens.data_ptr<int64_t>(),
            rows,
            vocab_size,
            static_cast<int>(beam_size),
            pad_id);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto best_scores = torch::empty(
      {batch_size, beam_size},
      torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));
  auto parent_indices = torch::empty(
      {batch_size, beam_size},
      torch::TensorOptions().dtype(torch::kLong).device(logits.device()));
  auto next_tokens = torch::empty(
      {batch_size, beam_size},
      torch::TensorOptions().dtype(torch::kLong).device(logits.device()));
  auto next_finished = torch::empty(
      {batch_size, beam_size},
      torch::TensorOptions().dtype(torch::kBool).device(logits.device()));
  auto next_lengths = torch::empty(
      {batch_size, beam_size},
      torch::TensorOptions().dtype(torch::kLong).device(logits.device()));

  beam_search_merge_kernel<kMaxBeamSearchBeamSize><<<static_cast<unsigned int>(batch_size), 1, 0, stream.stream()>>>(
      cand_scores.data_ptr<float>(),
      cand_tokens.data_ptr<int64_t>(),
      finished_contig.data_ptr<bool>(),
      lengths_contig.data_ptr<int64_t>(),
      best_scores.data_ptr<float>(),
      parent_indices.data_ptr<int64_t>(),
      next_tokens.data_ptr<int64_t>(),
      next_finished.data_ptr<bool>(),
      next_lengths.data_ptr<int64_t>(),
      batch_size,
      static_cast<int>(beam_size),
      eos_id,
      pad_id);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto next_beams = torch::empty(
      {rows, seq_len + 1},
      torch::TensorOptions().dtype(torch::kLong).device(logits.device()));
  const int gather_threads = static_cast<int>(std::min<int64_t>(seq_len + 1, kThreads));
  beam_search_gather_append_kernel<<<static_cast<unsigned int>(rows), gather_threads, 0, stream.stream()>>>(
      beams_contig.data_ptr<int64_t>(),
      parent_indices.data_ptr<int64_t>(),
      next_tokens.data_ptr<int64_t>(),
      next_beams.data_ptr<int64_t>(),
      batch_size,
      static_cast<int>(beam_size),
      seq_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto parent_rows = (
      torch::arange(batch_size, parent_indices.options()).unsqueeze(1) * beam_size + parent_indices)
                          .reshape({rows});
  return {next_beams, best_scores, next_finished, next_lengths, parent_rows};
}

torch::Tensor CudaSampleWithPoliciesForward(
    const torch::Tensor& logits,
    const torch::Tensor& token_ids,
    bool do_sample,
    double temperature,
    const c10::optional<int64_t>& top_k,
    const c10::optional<double>& top_p,
    int64_t no_repeat_ngram,
    double repetition_penalty,
    double presence_penalty,
    double frequency_penalty) {
  TORCH_CHECK(logits.is_cuda() && token_ids.is_cuda(),
              "CudaSampleWithPoliciesForward: logits and token_ids must be CUDA tensors");
  TORCH_CHECK(logits.dim() == 2, "CudaSampleWithPoliciesForward: logits must be rank-2 (B, V)");
  TORCH_CHECK(token_ids.dim() == 2, "CudaSampleWithPoliciesForward: token_ids must be rank-2 (B, T)");
  TORCH_CHECK(logits.size(0) == token_ids.size(0), "CudaSampleWithPoliciesForward: batch mismatch");

  c10::cuda::CUDAGuard device_guard{logits.device()};

  auto logits_contig = logits.contiguous();
  auto token_ids_contig = NormalizeTokenIdsLongContiguous(token_ids);
  auto x = logits_contig;
  const auto batch_size = logits_contig.size(0);
  const auto vocab_size = logits_contig.size(1);
  const auto numel = logits_contig.numel();
  const dim3 blocks(static_cast<unsigned int>((numel + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device());

  const bool apply_repetition_penalty = repetition_penalty != 1.0;
  const bool apply_temperature = do_sample && temperature != 1.0;
  const bool apply_presence_penalty = presence_penalty != 0.0;
  const bool apply_frequency_penalty = frequency_penalty != 0.0;
  const bool use_history_mask = no_repeat_ngram == 1;
  const bool need_no_repeat_ngram_mask = no_repeat_ngram > 1;

  torch::Tensor seen_mask;
  if (apply_repetition_penalty || use_history_mask || apply_presence_penalty) {
    seen_mask = BuildSeenMask(token_ids_contig, vocab_size);
  }

  bool prepared_logits = false;
  if (apply_repetition_penalty || apply_temperature) {
    x = torch::empty_like(logits_contig);
    const float inv_tau = 1.0f / static_cast<float>(std::max(temperature, 1e-8));
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        logits_contig.scalar_type(),
        "model_stack_cuda_prepare_sampling_logits",
        [&] {
          prepare_sampling_logits_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
              logits_contig.data_ptr<scalar_t>(),
              seen_mask.defined() ? seen_mask.data_ptr<bool>() : nullptr,
              nullptr,
              x.data_ptr<scalar_t>(),
              numel,
              static_cast<float>(repetition_penalty),
              inv_tau,
              apply_repetition_penalty,
              apply_temperature);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    prepared_logits = true;
  }

  const auto bounded_top_k =
      top_k.has_value() ? std::min<int64_t>(top_k.value(), vocab_size) : int64_t{0};
  const bool deterministic_top1 = do_sample && bounded_top_k == 1;
  const bool need_topk_mask = do_sample && bounded_top_k > 0 && bounded_top_k < vocab_size;
  const bool need_topp_mask =
      do_sample &&
      !deterministic_top1 &&
      top_p.has_value() &&
      top_p.value() > 0.0 &&
      top_p.value() < 1.0;
  const bool deterministic_top1_needs_filter =
      deterministic_top1 && (use_history_mask || need_no_repeat_ngram_mask);
  const bool can_sample_from_sorted_filtered =
      do_sample &&
      need_topp_mask;
  const bool can_sample_from_filtered =
      do_sample &&
      !deterministic_top1 &&
      !need_topp_mask &&
      (need_topk_mask || use_history_mask || need_no_repeat_ngram_mask);
  const bool can_greedy_from_filtered =
      ((!do_sample) || deterministic_top1) &&
      (use_history_mask || need_no_repeat_ngram_mask);
  torch::Tensor no_repeat_mask;
  torch::Tensor topk_thresholds;
  torch::Tensor sorted_logits;
  torch::Tensor sorted_indices;
  torch::Tensor sorted_row_sums;
  if (apply_presence_penalty || apply_frequency_penalty) {
    if (!prepared_logits) {
      x = logits_contig.clone();
      prepared_logits = true;
    }
    if (apply_presence_penalty) {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          x.scalar_type(),
          "model_stack_cuda_presence_penalty_from_seen_mask_inplace",
          [&] {
            presence_penalty_from_seen_mask_inplace_kernel<scalar_t>
                <<<blocks, threads, 0, stream.stream()>>>(
                    seen_mask.data_ptr<bool>(),
                    x.data_ptr<scalar_t>(),
                    numel,
                    static_cast<float>(presence_penalty));
          });
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (apply_frequency_penalty) {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          x.scalar_type(),
          "model_stack_cuda_frequency_penalty_from_tokens_inplace_preselection",
          [&] {
            frequency_penalty_from_tokens_inplace_kernel<scalar_t>
                <<<static_cast<unsigned int>(batch_size), 1, 0, stream.stream()>>>(
                    token_ids_contig.data_ptr<int64_t>(),
                    x.data_ptr<scalar_t>(),
                    batch_size,
                    token_ids_contig.size(1),
                    vocab_size,
                    static_cast<float>(frequency_penalty));
          });
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  if (deterministic_top1 && !deterministic_top1_needs_filter) {
    return CudaGreedyNextTokenForward(x);
  }
  if (need_topp_mask) {
    if (need_topk_mask) {
      std::tie(sorted_logits, sorted_indices) =
          TopkLogitsWithIndicesDescending(x, bounded_top_k);
      sorted_row_sums = ComputeSoftmaxRowSumsFromMax(
          x,
          sorted_logits.select(1, 0).unsqueeze(1));
    } else {
      std::tie(sorted_logits, sorted_indices) = SortLogitsWithIndicesDescending(x);
    }
  }
  if (need_topk_mask &&
      !need_topp_mask &&
      (!deterministic_top1 || deterministic_top1_needs_filter)) {
    if (sorted_logits.defined()) {
      topk_thresholds = sorted_logits.select(1, bounded_top_k - 1).contiguous();
    } else {
      topk_thresholds = ComputeTopkThresholds(x, bounded_top_k);
    }
  }
  if (can_sample_from_sorted_filtered) {
    if (need_no_repeat_ngram_mask) {
      no_repeat_mask = torch::zeros(
          {batch_size, vocab_size},
          torch::TensorOptions().dtype(torch::kBool).device(logits.device()));
      ApplyNoRepeatNgramMaskFromContig(token_ids_contig, vocab_size, no_repeat_ngram, no_repeat_mask);
    }
    auto next_token = torch::empty(
        {batch_size, 1},
        torch::TensorOptions().dtype(torch::kLong).device(logits.device()));
    const auto policy = SelectSamplingLaunchPolicy(sorted_logits.size(1));
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "model_stack_cuda_sorted_filtered_next_token_forward",
        [&] {
          LaunchSortedFilteredSampleKernel<scalar_t>(
              x,
              sorted_logits,
              sorted_indices,
              next_token,
              seen_mask.defined() ? c10::optional<torch::Tensor>(seen_mask) : c10::nullopt,
              no_repeat_mask.defined() ? c10::optional<torch::Tensor>(no_repeat_mask) : c10::nullopt,
              topk_thresholds.defined() ? c10::optional<torch::Tensor>(topk_thresholds) : c10::nullopt,
              sorted_row_sums.defined() ? c10::optional<torch::Tensor>(sorted_row_sums) : c10::nullopt,
              use_history_mask,
              0.0,
              top_p.value(),
              stream.stream(),
              policy.row_reduce_threads);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return next_token;
  }
  if (can_sample_from_filtered) {
    if (need_no_repeat_ngram_mask) {
      no_repeat_mask = torch::zeros(
          {batch_size, vocab_size},
          torch::TensorOptions().dtype(torch::kBool).device(logits.device()));
      ApplyNoRepeatNgramMaskFromContig(token_ids_contig, vocab_size, no_repeat_ngram, no_repeat_mask);
    }
    auto next_token = torch::empty(
        {batch_size, 1},
        torch::TensorOptions().dtype(torch::kLong).device(logits.device()));
    const auto policy = SelectSamplingLaunchPolicy(vocab_size);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "model_stack_cuda_filtered_multinomial_next_token_forward",
        [&] {
          LaunchFilteredMultinomialNextTokenKernel<scalar_t>(
              x,
              next_token,
              seen_mask.defined() ? c10::optional<torch::Tensor>(seen_mask) : c10::nullopt,
              no_repeat_mask.defined() ? c10::optional<torch::Tensor>(no_repeat_mask) : c10::nullopt,
              topk_thresholds.defined() ? c10::optional<torch::Tensor>(topk_thresholds) : c10::nullopt,
              use_history_mask,
              0.0,
              stream.stream(),
              policy.row_reduce_threads);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return next_token;
  }
  if (can_greedy_from_filtered) {
    if (need_no_repeat_ngram_mask) {
      no_repeat_mask = torch::zeros(
          {batch_size, vocab_size},
          torch::TensorOptions().dtype(torch::kBool).device(logits.device()));
      ApplyNoRepeatNgramMaskFromContig(token_ids_contig, vocab_size, no_repeat_ngram, no_repeat_mask);
    }
    auto next_token = torch::empty(
        {batch_size, 1},
        torch::TensorOptions().dtype(torch::kLong).device(logits.device()));
    const auto policy = SelectSamplingLaunchPolicy(vocab_size);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "model_stack_cuda_filtered_greedy_next_token_forward",
        [&] {
          LaunchFilteredGreedyNextTokenKernel<scalar_t>(
              x,
              next_token,
              seen_mask.defined() ? c10::optional<torch::Tensor>(seen_mask) : c10::nullopt,
              no_repeat_mask.defined() ? c10::optional<torch::Tensor>(no_repeat_mask) : c10::nullopt,
              topk_thresholds.defined() ? c10::optional<torch::Tensor>(topk_thresholds) : c10::nullopt,
              use_history_mask,
              0.0,
              stream.stream(),
              policy.row_reduce_threads);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return next_token;
  }

  return do_sample ? CudaMultinomialSampleForward(x) : CudaGreedyNextTokenForward(x);
}
