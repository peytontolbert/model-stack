#pragma once

namespace t10::bitnet {

template <typename scalar_t>
__device__ inline scalar_t CastOutput(float value) {
  return static_cast<scalar_t>(value);
}

}  // namespace t10::bitnet
