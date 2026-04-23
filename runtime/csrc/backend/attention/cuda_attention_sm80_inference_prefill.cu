#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "cuda_attention_sm80_inference_prefill.cuh"

#ifdef MODEL_STACK_WITH_PYTORCH_MEMEFF_FMHA
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>
#include <cutlass/arch/arch.h>
#endif

namespace t10::cuda::attention {

#ifdef MODEL_STACK_WITH_PYTORCH_MEMEFF_FMHA

template <typename scalar_t>
struct ModelStackSm80InferenceScalar;

template <>
struct ModelStackSm80InferenceScalar<at::Half> {
  using type = cutlass::half_t;
};

template <>
struct ModelStackSm80InferenceScalar<at::BFloat16> {
  using type = cutlass::bfloat16_t;
};

template <typename scalar_t>
using ModelStackSm80InferenceScalarT = typename ModelStackSm80InferenceScalar<scalar_t>::type;

template <typename scalar_t, int QueriesPerBlock, int KeysPerBlock, int MaxK>
struct ModelStackSm80CausalPrefillKernel {
  // Reuse the proven tensor-core building blocks, but own the top-level prefill
  // contract and strip it down to the causal BHSD MHA inference case.
  using BaseKernel = PyTorchMemEffAttention::AttentionKernel<
      scalar_t,
      cutlass::arch::Sm80,
      true,
      QueriesPerBlock,
      KeysPerBlock,
      MaxK,
      false,
      false>;

  using element_t = typename BaseKernel::scalar_t;
  using accum_t = typename BaseKernel::accum_t;
  using output_t = typename BaseKernel::output_t;
  using MM0 = typename BaseKernel::MM0;
  using MM1 = typename BaseKernel::MM1;
  using SharedStorage = typename BaseKernel::SharedStorage;
  using MM1DefaultMmaFromSmem = typename BaseKernel::MM1::DefaultMmaFromSmem;

  template <typename IteratorBase>
  struct MM1FullTileIteratorB {
    using Shape = typename IteratorBase::Shape;
    using Element = typename IteratorBase::Element;
    using Layout = typename IteratorBase::Layout;
    static int const kAdvanceRank = IteratorBase::kAdvanceRank;
    using ThreadMap = typename IteratorBase::ThreadMap;
    using AccessType = typename IteratorBase::AccessType;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename IteratorBase::TensorCoord;
    using Pointer = typename IteratorBase::Pointer;
    using RegularIterator = cutlass::transform::threadblock::RegularTileAccessIterator<
        Shape,
        Element,
        Layout,
        kAdvanceRank,
        ThreadMap>;
    using Fragment = cutlass::Array<
        Element,
        ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
    static int const kAccessesPerVector =
        ThreadMap::kElementsPerAccess / AccessType::kElements;

    struct Params {
      Layout layout;
      CUTLASS_HOST_DEVICE
      Params() = default;
      CUTLASS_HOST_DEVICE
      Params(Layout const& layout_) : layout(layout_) {}
    };

    RegularIterator iterator_;

    CUTLASS_HOST_DEVICE
    MM1FullTileIteratorB(
        Params const& params,
        Pointer pointer,
        TensorCoord,
        int thread_id,
        TensorCoord const& threadblock_offset)
        : iterator_(typename RegularIterator::TensorRef(pointer, params.layout), thread_id) {
      iterator_.add_tile_offset(threadblock_offset);
    }

    CUTLASS_HOST_DEVICE
    MM1FullTileIteratorB(
        Params const& params,
        Pointer pointer,
        TensorCoord extent,
        int thread_id)
        : MM1FullTileIteratorB(params, pointer, extent, thread_id, cutlass::make_Coord(0, 0)) {}

    CUTLASS_HOST_DEVICE
    void set_iteration_index(int index) {
      iterator_.set_iteration_index(index);
    }

    CUTLASS_HOST_DEVICE
    void set_residual_tile(bool) {}

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
      iterator_.add_pointer_offset(pointer_offset);
    }

    CUTLASS_DEVICE
    void add_tile_offset(TensorCoord const& tile_offset) {
      iterator_.add_tile_offset(tile_offset);
    }

    CUTLASS_HOST_DEVICE
    AccessType* get() const {
      return reinterpret_cast<AccessType*>(iterator_.get());
    }

    CUTLASS_HOST_DEVICE
    MM1FullTileIteratorB& operator++() {
      ++iterator_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    MM1FullTileIteratorB operator++(int) {
      MM1FullTileIteratorB prev(*this);
      ++(*this);
      return prev;
    }

    CUTLASS_HOST_DEVICE
    void clear_mask(bool = true) {}

    CUTLASS_HOST_DEVICE
    bool valid() const {
      return true;
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) const {
      static_assert(kAccessesPerVector == 1, "full-tile iterator expects one access per vector");
      auto iter = iterator_;
      auto* frag_ptr = reinterpret_cast<AccessType*>(&frag);

      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
          int access_idx = c + s * ThreadMap::Iterations::kContiguous;
          frag_ptr[access_idx] = *iter.get();
          ++iter;
        }
      }
    }
  };

  template <typename RegularMma, typename WarpIteratorA, int MM1MaxK>
  struct MM1FullTileMmaFromSmem;

  template <
      typename Shape_,
      typename IteratorA_,
      typename SmemIteratorA_,
      typename IteratorB_,
      typename SmemIteratorB_,
      typename ElementC_,
      typename LayoutC_,
      typename Policy_,
      typename TransformA_,
      typename TransformB_,
      typename WarpIteratorA,
      int MM1MaxK>
  struct MM1FullTileMmaFromSmem<
      cutlass::gemm::threadblock::MmaPipelined<
          Shape_,
          IteratorA_,
          SmemIteratorA_,
          IteratorB_,
          SmemIteratorB_,
          ElementC_,
          LayoutC_,
          Policy_,
          TransformA_,
          TransformB_>,
      WarpIteratorA,
      MM1MaxK> {
    using IteratorB = MM1FullTileIteratorB<IteratorB_>;
    using Mma = cutlass::gemm::threadblock::MmaPipelinedFromSharedMemory<
        Shape_,
        WarpIteratorA,
        false,
        MM1MaxK,
        IteratorB,
        SmemIteratorB_,
        ElementC_,
        LayoutC_,
        Policy_,
        TransformB_>;
  };

  template <
      typename Shape_,
      typename IteratorA_,
      typename SmemIteratorA_,
      cutlass::arch::CacheOperation::Kind CacheOpA,
      typename IteratorB_,
      typename SmemIteratorB_,
      cutlass::arch::CacheOperation::Kind CacheOpB,
      typename ElementC_,
      typename LayoutC_,
      typename Policy_,
      int Stages,
      cutlass::gemm::SharedMemoryClearOption SharedMemoryClear,
      typename WarpIteratorA,
      int MM1MaxK>
  struct MM1FullTileMmaFromSmem<
      cutlass::gemm::threadblock::MmaMultistage<
          Shape_,
          IteratorA_,
          SmemIteratorA_,
          CacheOpA,
          IteratorB_,
          SmemIteratorB_,
          CacheOpB,
          ElementC_,
          LayoutC_,
          Policy_,
          Stages,
          SharedMemoryClear>,
      WarpIteratorA,
      MM1MaxK> {
    static constexpr int kStagesMax =
        (MM1MaxK + int(Shape_::kK) - 1) / int(Shape_::kK);
    static constexpr int kReducedStages = cutlass::const_min(Stages, kStagesMax);
    using IteratorB = MM1FullTileIteratorB<IteratorB_>;
    using Mma = cutlass::gemm::threadblock::MmaMultistageFromSharedMemory<
        Shape_,
        WarpIteratorA,
        false,
        IteratorB,
        SmemIteratorB_,
        CacheOpB,
        ElementC_,
        LayoutC_,
        Policy_,
        kReducedStages,
        MM1MaxK>;
  };

  using MM1FullTileMmaConfig = MM1FullTileMmaFromSmem<
      typename MM1DefaultMmaFromSmem::RegularMma,
      typename MM1DefaultMmaFromSmem::WarpIteratorA,
      MM0::AccumulatorSharedStorage::Shape::kN>;
  using MM1FullTileMma = typename MM1FullTileMmaConfig::Mma;
  using MM1FullTileIterator = typename MM1FullTileMmaConfig::IteratorB;

  static constexpr int kQueriesPerBlock = BaseKernel::kQueriesPerBlock;
  static constexpr int kKeysPerBlock = BaseKernel::kKeysPerBlock;
  static constexpr int kMaxK = BaseKernel::kMaxK;
  static constexpr int kAlignLSE = BaseKernel::kAlignLSE;
  static constexpr bool kIsHalf = BaseKernel::kIsHalf;
  static constexpr bool kPreloadV = BaseKernel::kPreloadV;
  static constexpr bool kKeepOutputInRF = BaseKernel::kKeepOutputInRF;
  static constexpr bool kNeedsOutputAccumulatorBuffer = BaseKernel::kNeedsOutputAccumulatorBuffer;
  static constexpr int kNumWarpsPerBlock = BaseKernel::kNumWarpsPerBlock;
  static constexpr int kWarpSize = BaseKernel::kWarpSize;
  static constexpr int kNumThreads = BaseKernel::kNumThreads;
  static constexpr int kMinBlocksPerSm = BaseKernel::kMinBlocksPerSm;
  static constexpr int64_t kAlignmentQ = BaseKernel::kAlignmentQ;
  static constexpr int64_t kAlignmentK = BaseKernel::kAlignmentK;
  static constexpr int64_t kAlignmentV = BaseKernel::kAlignmentV;
  static constexpr int kWarpColumns = MM0::MmaCore::WarpCount::kN;

  static_assert(kKeepOutputInRF, "SM80 causal prefill fast lanes are expected to keep output in registers.");
  static_assert(!kNeedsOutputAccumulatorBuffer, "SM80 causal prefill fast lanes should not allocate output accumulators.");
  static_assert(kPreloadV, "SM80 causal prefill fast lanes expect preloaded V tiles.");

  struct Params {
    const scalar_t* query_ptr = nullptr;
    const scalar_t* key_ptr = nullptr;
    const scalar_t* value_ptr = nullptr;
    output_t* output_ptr = nullptr;

    accum_t scale = 0.0f;

    int32_t head_dim = 0;
    int32_t head_dim_value = 0;
    int32_t num_queries = 0;
    int32_t num_keys = 0;

    int32_t q_strideM = 0;
    int32_t k_strideM = 0;
    int32_t v_strideM = 0;
    int32_t o_strideM = 0;

    int64_t q_strideB = 0;
    int64_t k_strideB = 0;
    int64_t v_strideB = 0;
    int64_t o_strideB = 0;

    int32_t num_batches = 0;

    CUTLASS_DEVICE bool advance_to_block() {
      const auto batch_id = blockIdx.z;
      const auto query_start = blockIdx.x * kQueriesPerBlock;
      if (batch_id >= num_batches || query_start >= num_queries) {
        return false;
      }

      query_ptr += batch_id * q_strideB + int64_t(query_start) * q_strideM;
      key_ptr += batch_id * k_strideB;
      value_ptr += batch_id * v_strideB;
      output_ptr += batch_id * o_strideB + int64_t(query_start) * o_strideM;

      num_keys = cutlass::fast_min(
          int32_t(query_start + kQueriesPerBlock),
          num_keys);
      num_queries -= query_start;

      query_ptr = warp_uniform(query_ptr);
      key_ptr = warp_uniform(key_ptr);
      value_ptr = warp_uniform(value_ptr);
      output_ptr = warp_uniform(output_ptr);
      num_queries = warp_uniform(num_queries);
      num_keys = warp_uniform(num_keys);
      return true;
    }

    __host__ dim3 getBlocksGrid() const {
      return dim3(
          ceil_div(num_queries, int32_t(kQueriesPerBlock)),
          1,
          num_batches);
    }

    __host__ dim3 getThreadsGrid() const {
      return dim3(kWarpSize, kNumWarpsPerBlock, 1);
    }
  };

  static bool __host__ check_supported(const Params& p) {
    CHECK_ALIGNED_PTR(p.query_ptr, kAlignmentQ);
    CHECK_ALIGNED_PTR(p.key_ptr, kAlignmentK);
    CHECK_ALIGNED_PTR(p.value_ptr, kAlignmentV);
    TORCH_CHECK(p.head_dim == 64, "SM80 causal prefill fast lanes require head_dim == 64");
    TORCH_CHECK(p.head_dim_value == 64, "SM80 causal prefill fast lanes require head_dim_value == 64");
    TORCH_CHECK(p.head_dim <= kMaxK, "SM80 causal prefill fast lane exceeded kMaxK");
    TORCH_CHECK(p.q_strideM % kAlignmentQ == 0, "query is not correctly aligned (strideM)");
    TORCH_CHECK(p.k_strideM % kAlignmentK == 0, "key is not correctly aligned (strideM)");
    TORCH_CHECK(p.v_strideM % kAlignmentV == 0, "value is not correctly aligned (strideM)");
    return p.num_queries > 0 && p.num_keys > 0 && p.num_batches > 0;
  }

  static CUTLASS_DEVICE int8_t lane_id() {
    return threadIdx.x;
  }

  static CUTLASS_DEVICE int8_t warp_id() {
    return threadIdx.y;
  }

  static CUTLASS_DEVICE int16_t thread_id() {
    return threadIdx.x + threadIdx.y * blockDim.x;
  }

  template <bool kFullTile, bool kFirstTile, typename WarpIteratorC>
  static CUTLASS_DEVICE void iterative_softmax_64x64_rf(
      typename WarpIteratorC::Fragment& frag_o,
      typename WarpIteratorC::Fragment& frag,
      cutlass::Array<accum_t, kQueriesPerBlock>& mi,
      cutlass::Array<accum_t, kQueriesPerBlock>& m_prime,
      cutlass::Array<accum_t, kQueriesPerBlock>& s_prime,
      cutlass::Array<accum_t, kQueriesPerBlock>& out_rescale,
      cutlass::Array<accum_t, kQueriesPerBlock * MM0::MmaCore::WarpCount::kN>&
          reduction_storage,
      int8_t lane_id,
      int8_t thread_id,
      int8_t warp_id,
      int max_col,
      bool,
      typename WarpIteratorC::TensorCoord const& tile_offset,
      float scaling) {
    using Fragment = typename WarpIteratorC::Fragment;
    using LambdaIterator = typename DefaultMmaAccumLambdaIterator<
        WarpIteratorC,
        accum_t,
        kWarpSize>::Iterator;
    constexpr float kLog2e = 1.4426950408889634074f;

    static_assert(kQueriesPerBlock == 64, "specialized softmax expects 64 query rows");
    static_assert(kWarpColumns == 2, "specialized softmax expects two warp columns");
    static_assert(kQueriesPerBlock % kNumWarpsPerBlock == 0, "specialized softmax expects even row ownership");

    frag = cutlass::multiplies<Fragment>()(scaling * kLog2e, frag);

    auto lane_offset =
        LambdaIterator::get_lane_offset(lane_id, warp_id, tile_offset);
    const int warp_col = tile_offset.column();
    const int owner_row = thread_id;
    accum_t* warp_column_scratch = reduction_storage.data();

    {
      accum_t row_max;
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) {
            row_max = -cutlass::platform::numeric_limits<accum_t>::infinity();
          },
          [&](int accum_m, int accum_n, int idx) {
            if constexpr (kFullTile) {
              row_max = cutlass::fast_max(row_max, frag[idx]);
            } else if (accum_n < max_col) {
              row_max = cutlass::fast_max(row_max, frag[idx]);
            }
          },
          [&](int accum_m) {
            if (LambdaIterator::reduceSameRow(
                    lane_id, row_max, [](accum_t a, accum_t b) {
                      return cutlass::fast_max(a, b);
                    })) {
              warp_column_scratch[accum_m + kQueriesPerBlock * warp_col] =
                  row_max;
            }
          });
    }

    __syncthreads();

    if (owner_row < kQueriesPerBlock) {
      // Pack the row-owner work onto the first 64 threads so the merge/update
      // phase runs on full warps instead of half-active warps.
      const accum_t new_m = cutlass::fast_max(
          mi[owner_row],
          cutlass::fast_max(
              warp_column_scratch[owner_row],
              warp_column_scratch[owner_row + kQueriesPerBlock]));
      mi[owner_row] = new_m;
      if constexpr (!kFirstTile) {
        const accum_t rescale = (m_prime[owner_row] < new_m)
            ? exp2f(m_prime[owner_row] - new_m)
            : accum_t(1.0f);
        out_rescale[owner_row] = rescale;
        s_prime[owner_row] *= rescale;
      }
    }

    __syncthreads();

    if constexpr (!kFirstTile) {
      accum_t line_rescale;
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) { line_rescale = out_rescale[accum_m]; },
          [&](int accum_m, int accum_n, int idx) {
            frag_o[idx] = frag_o[idx] * line_rescale;
          },
          [&](int accum_m) {});
    }

    {
      accum_t mi_row, total_row;
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) {
            mi_row = mi[accum_m];
            total_row = accum_t(0.0f);
          },
          [&](int accum_m, int accum_n, int idx) {
            if constexpr (kFullTile) {
              frag[idx] = exp2f(frag[idx] - mi_row);
            } else {
              frag[idx] =
                  (accum_n < max_col) ? exp2f(frag[idx] - mi_row) : accum_t(0.0f);
            }
            total_row += frag[idx];
          },
          [&](int accum_m) {
            if (LambdaIterator::reduceSameRow(
                    lane_id, total_row, [](accum_t a, accum_t b) {
                      return a + b;
                    })) {
              warp_column_scratch[accum_m + kQueriesPerBlock * warp_col] =
                  total_row;
            }
          });
    }

    __syncthreads();

    if (owner_row < kQueriesPerBlock) {
      s_prime[owner_row] +=
          warp_column_scratch[owner_row] +
          warp_column_scratch[owner_row + kQueriesPerBlock];
      m_prime[owner_row] = mi[owner_row];
    }
  }

  template <bool kFullTileExtent>
  static CUTLASS_DEVICE typename MM0::IteratorA make_q_iterator(
      Params& p,
      int16_t tid,
      cutlass::MatrixCoord const& tb_offset_a,
      int32_t problem_size_0_m) {
    const int32_t query_extent = kFullTileExtent ? int32_t(kQueriesPerBlock) : problem_size_0_m;
    return typename MM0::IteratorA(
        typename MM0::IteratorA::Params(
            typename MM0::MmaCore::LayoutA(p.q_strideM)),
        const_cast<scalar_t*>(p.query_ptr),
        {query_extent, kMaxK},
        tid,
        tb_offset_a);
  }

  template <bool kFullTileExtent>
  static CUTLASS_DEVICE typename MM0::IteratorB make_k_iterator(
      Params& p,
      int32_t iter_key_start,
      int16_t tid,
      cutlass::MatrixCoord const& tb_offset_b,
      int32_t problem_size_0_n) {
    const int32_t key_extent = kFullTileExtent ? int32_t(kKeysPerBlock) : problem_size_0_n;
    return typename MM0::IteratorB(
        typename MM0::IteratorB::Params(
            typename MM0::MmaCore::LayoutB(p.k_strideM)),
        const_cast<scalar_t*>(p.key_ptr + iter_key_start * p.k_strideM),
        {kMaxK, key_extent},
        tid,
        tb_offset_b);
  }

  template <bool kFullTileExtent>
  static CUTLASS_DEVICE void prologue_v_tile(
      SharedStorage& shared_storage,
      Params& p,
      int32_t iter_key_start,
      int16_t tid,
      int32_t problem_size_1_k) {
    const int32_t value_extent = kFullTileExtent ? int32_t(kKeysPerBlock) : problem_size_1_k;
    typename MM1::Mma::IteratorB iterator_v(
        typename MM1::IteratorB::Params{MM1::LayoutB(p.v_strideM)},
        const_cast<scalar_t*>(p.value_ptr + iter_key_start * p.v_strideM),
        {value_extent, kMaxK},
        tid,
        cutlass::MatrixCoord{0, 0});
    MM1::Mma::prologue(
        shared_storage.after_mm0.mm1,
        iterator_v,
        tid,
        value_extent);
  }

  template <bool kFullTileExtent>
  static CUTLASS_DEVICE void mma_pv_tile(
      SharedStorage& shared_storage,
      Params& p,
      int32_t iter_key_start,
      int16_t tid,
      int8_t my_warp_id,
      int8_t my_lane_id,
      int32_t problem_size_1_k,
      typename MM1::Mma::FragmentC& accum_o) {
    const int32_t value_extent = kFullTileExtent ? int32_t(kKeysPerBlock) : problem_size_1_k;
    typename MM1::Mma::IteratorB iterator_v(
        typename MM1::IteratorB::Params{MM1::LayoutB(p.v_strideM)},
        const_cast<scalar_t*>(p.value_ptr + iter_key_start * p.v_strideM),
        {value_extent, kMaxK},
        tid,
        cutlass::MatrixCoord{0, 0});
    typename MM1::Mma mma_pv(
        shared_storage.after_mm0.si.accum_ref(),
        shared_storage.after_mm0.mm1.operand_B_ref(),
        (int)tid,
        (int)my_warp_id,
        (int)my_lane_id);
    mma_pv.set_prologue_done(kPreloadV);
    if constexpr (kFullTileExtent) {
      static_assert(
          kKeysPerBlock % MM1::Mma::Shape::kK == 0,
          "SM80 causal prefill fast lane expects full PV tiles to map cleanly onto MM1");
      constexpr int kFullPvIterations = kKeysPerBlock / MM1::Mma::Shape::kK;
      mma_pv(kFullPvIterations, accum_o, iterator_v, accum_o);
    } else {
      const auto gemm_k_iterations_pv =
          (problem_size_1_k + MM1::Mma::Shape::kK - 1) / MM1::Mma::Shape::kK;
      mma_pv(gemm_k_iterations_pv, accum_o, iterator_v, accum_o);
    }
  }

  static CUTLASS_DEVICE void prologue_v_tile_full_64x64_rf(
      SharedStorage& shared_storage,
      Params& p,
      int32_t iter_key_start,
      int16_t tid) {
    typename MM1FullTileIterator::Params params{MM1::LayoutB(p.v_strideM)};
    MM1FullTileIterator iterator_v(
        params,
        const_cast<scalar_t*>(p.value_ptr + iter_key_start * p.v_strideM),
        {int32_t(kKeysPerBlock), kMaxK},
        tid,
        cutlass::MatrixCoord{0, 0});
    MM1FullTileMma::prologue(
        shared_storage.after_mm0.mm1,
        iterator_v,
        tid,
        int32_t(kKeysPerBlock));
  }

  static CUTLASS_DEVICE void mma_pv_tile_full_64x64_rf(
      SharedStorage& shared_storage,
      Params& p,
      int32_t iter_key_start,
      int16_t tid,
      int8_t my_warp_id,
      int8_t my_lane_id,
      typename MM1::Mma::FragmentC& accum_o) {
    typename MM1FullTileIterator::Params params{MM1::LayoutB(p.v_strideM)};
    MM1FullTileIterator iterator_v(
        params,
        const_cast<scalar_t*>(p.value_ptr + iter_key_start * p.v_strideM),
        {int32_t(kKeysPerBlock), kMaxK},
        tid,
        cutlass::MatrixCoord{0, 0});
    MM1FullTileMma mma_pv(
        shared_storage.after_mm0.si.accum_ref(),
        shared_storage.after_mm0.mm1.operand_B_ref(),
        (int)tid,
        (int)my_warp_id,
        (int)my_lane_id);
    mma_pv.set_prologue_done(kPreloadV);
    static_assert(
        kKeysPerBlock % MM1FullTileMma::Shape::kK == 0,
        "full-tile MM1 fast lane expects 64-wide V tiles to map cleanly onto MM1");
    constexpr int kFullPvIterations = kKeysPerBlock / MM1FullTileMma::Shape::kK;
    mma_pv(kFullPvIterations, accum_o, iterator_v, accum_o);
  }

  static void CUTLASS_DEVICE attention_kernel(Params& p) {
    extern __shared__ char smem_buffer[];
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buffer);
    auto& m_prime = shared_storage.m_prime;
    auto& s_prime = shared_storage.s_prime;
    auto& mi = shared_storage.mi;
    auto& out_rescale = shared_storage.out_rescale;
    const uint32_t query_start = blockIdx.x * kQueriesPerBlock;

    static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize, "");
    if (thread_id() < kQueriesPerBlock) {
      s_prime[thread_id()] = accum_t(0);
      out_rescale[thread_id()] = accum_t(1.0);
      mi[thread_id()] = -cutlass::platform::numeric_limits<accum_t>::infinity();
      m_prime[thread_id()] = -cutlass::platform::numeric_limits<accum_t>::infinity();
    }

    typename MM1::Mma::FragmentC accum_o;
    accum_o.clear();

    auto create_output_iter = [&](int col) -> typename MM1::OutputTileIterator {
      using OutputTileIterator = typename MM1::OutputTileIterator;
      return OutputTileIterator(
          typename OutputTileIterator::Params{(int32_t)p.o_strideM},
          p.output_ptr,
          typename OutputTileIterator::TensorCoord{
              p.num_queries,
              kMaxK},
          thread_id(),
          {0, col});
    };

    const int16_t tid = thread_id();
    const int8_t my_warp_id = warp_uniform(warp_id());
    const int8_t my_lane_id = lane_id();
    const int32_t problem_size_0_m =
        cutlass::fast_min(int32_t(kQueriesPerBlock), p.num_queries);
    const bool full_query_tile = problem_size_0_m == kQueriesPerBlock;

    cutlass::gemm::GemmCoord tb_tile_offset = {0, 0, 0};
    cutlass::MatrixCoord tb_offset_a{
        tb_tile_offset.m() * MM0::Mma::Shape::kM,
        tb_tile_offset.k()};
    cutlass::MatrixCoord tb_offset_b{
        tb_tile_offset.k(),
        tb_tile_offset.n() * MM0::Mma::Shape::kN};

    typename MM0::Mma::Operator::IteratorC::TensorCoord iterator_c_tile_offset = {
        (tb_tile_offset.m() * MM0::Mma::WarpCount::kM) +
            (my_warp_id % MM0::Mma::WarpCount::kM),
        (tb_tile_offset.n() * MM0::Mma::WarpCount::kN) +
            (my_warp_id / MM0::Mma::WarpCount::kM)};

    const int warp_idx_mn_0 = my_warp_id %
        (MM0::Mma::Base::WarpCount::kM * MM0::Mma::Base::WarpCount::kN);
    const auto output_tile_coords = cutlass::MatrixCoord{
        warp_idx_mn_0 % MM0::Mma::Base::WarpCount::kM,
        warp_idx_mn_0 / MM0::Mma::Base::WarpCount::kM};

    for (int32_t iter_key_start = 0; iter_key_start < p.num_keys; iter_key_start += kKeysPerBlock) {
      const int32_t remaining_keys = p.num_keys - iter_key_start;
      const bool full_key_tile = remaining_keys >= int32_t(kKeysPerBlock);
      const int32_t problem_size_0_n =
          full_key_tile ? int32_t(kKeysPerBlock) : remaining_keys;
      const int32_t problem_size_1_k = problem_size_0_n;
      const bool full_tile_extent = full_query_tile && full_key_tile;
      const bool diagonal_tile =
          iter_key_start >= int32_t(query_start);

      auto iterator_a = full_query_tile
          ? make_q_iterator<true>(p, tid, tb_offset_a, problem_size_0_m)
          : make_q_iterator<false>(p, tid, tb_offset_a, problem_size_0_m);

      auto iterator_b = full_key_tile
          ? make_k_iterator<true>(p, iter_key_start, tid, tb_offset_b, problem_size_0_n)
          : make_k_iterator<false>(p, iter_key_start, tid, tb_offset_b, problem_size_0_n);

      typename MM0::Mma mma(
          shared_storage.mm0,
          tid,
          my_warp_id,
          my_lane_id);
      typename MM0::Mma::FragmentC accum;
      accum.clear();

      static_assert(
          kMaxK % MM0::Mma::Shape::kK == 0,
          "SM80 causal prefill fast lane expects head_dim == 64 to map cleanly onto MM0");
      constexpr int kQkGemmIterations = kMaxK / MM0::Mma::Shape::kK;
      mma(kQkGemmIterations, accum, iterator_a, iterator_b, accum);
      __syncthreads();

      if (full_tile_extent) {
        if constexpr (kQueriesPerBlock == 64 && kKeysPerBlock == 64 && kMaxK == 64) {
          if (!diagonal_tile) {
            prologue_v_tile_full_64x64_rf(
                shared_storage,
                p,
                iter_key_start,
                tid);
          } else {
            prologue_v_tile<true>(
                shared_storage,
                p,
                iter_key_start,
                tid,
                problem_size_1_k);
          }
        } else {
          prologue_v_tile<true>(
              shared_storage,
              p,
              iter_key_start,
              tid,
              problem_size_1_k);
        }
      } else {
        prologue_v_tile<false>(
            shared_storage,
            p,
            iter_key_start,
            tid,
            problem_size_1_k);
      }

      if (diagonal_tile) {
        auto lane_offset = MM0::AccumLambdaIterator::get_lane_offset(
            my_lane_id,
            my_warp_id,
            iterator_c_tile_offset);
        int32_t last_col;
        MM0::AccumLambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_m) {
              last_col = query_start + accum_m - iter_key_start;
            },
            [&](int accum_m, int accum_n, int idx) {
              if (accum_n > last_col) {
                accum[idx] = -cutlass::platform::numeric_limits<accum_t>::infinity();
              }
            },
            [&](int accum_m) {});
      }

      if constexpr (kQueriesPerBlock == 64 && kKeysPerBlock == 64 && kMaxK == 64) {
        if (!diagonal_tile && problem_size_0_n == kKeysPerBlock) {
          if (iter_key_start == 0) {
            iterative_softmax_64x64_rf<true, true, typename MM0::Mma::Operator::IteratorC>(
                accum_o,
                accum,
                mi,
                m_prime,
                s_prime,
                out_rescale,
                shared_storage.addition_storage,
                my_lane_id,
                tid,
                my_warp_id,
                kKeysPerBlock,
                true,
                iterator_c_tile_offset,
                p.scale);
          } else {
            iterative_softmax_64x64_rf<true, false, typename MM0::Mma::Operator::IteratorC>(
                accum_o,
                accum,
                mi,
                m_prime,
                s_prime,
                out_rescale,
                shared_storage.addition_storage,
                my_lane_id,
                tid,
                my_warp_id,
                kKeysPerBlock,
                false,
                iterator_c_tile_offset,
                p.scale);
          }
        } else {
          if (iter_key_start == 0) {
            iterative_softmax_64x64_rf<false, true, typename MM0::Mma::Operator::IteratorC>(
                accum_o,
                accum,
                mi,
                m_prime,
                s_prime,
                out_rescale,
                shared_storage.addition_storage,
                my_lane_id,
                tid,
                my_warp_id,
                remaining_keys,
                true,
                iterator_c_tile_offset,
                p.scale);
          } else {
            iterative_softmax_64x64_rf<false, false, typename MM0::Mma::Operator::IteratorC>(
                accum_o,
                accum,
                mi,
                m_prime,
                s_prime,
                out_rescale,
                shared_storage.addition_storage,
                my_lane_id,
                tid,
                my_warp_id,
                remaining_keys,
                false,
                iterator_c_tile_offset,
                p.scale);
          }
        }
      } else {
        BaseKernel::template iterative_softmax<typename MM0::Mma::Operator::IteratorC>(
            accum_o,
            accum,
            mi,
            m_prime,
            s_prime,
            out_rescale,
            shared_storage.addition_storage,
            my_lane_id,
            tid,
            my_warp_id,
            remaining_keys,
            iter_key_start == 0,
            iterator_c_tile_offset,
            p.scale);
      }

      MM0::B2bGemm::accumToSmem(
          shared_storage.after_mm0.si,
          accum,
          my_lane_id,
          output_tile_coords);
      __syncthreads();

      if (full_tile_extent) {
        if constexpr (kQueriesPerBlock == 64 && kKeysPerBlock == 64 && kMaxK == 64) {
          if (!diagonal_tile) {
            mma_pv_tile_full_64x64_rf(
                shared_storage,
                p,
                iter_key_start,
                tid,
                my_warp_id,
                my_lane_id,
                accum_o);
          } else {
            mma_pv_tile<true>(
                shared_storage,
                p,
                iter_key_start,
                tid,
                my_warp_id,
                my_lane_id,
                problem_size_1_k,
                accum_o);
          }
        } else {
          mma_pv_tile<true>(
              shared_storage,
              p,
              iter_key_start,
              tid,
              my_warp_id,
              my_lane_id,
              problem_size_1_k,
              accum_o);
        }
      } else {
        mma_pv_tile<false>(
            shared_storage,
            p,
            iter_key_start,
            tid,
            my_warp_id,
            my_lane_id,
            problem_size_1_k,
            accum_o);
      }
      __syncthreads();
    }

    using DefaultEpilogue = typename MM1::DefaultEpilogue;
    using DefaultOp = typename MM1::DefaultConfig::EpilogueOutputOp;
    using ElementCompute = typename DefaultOp::ElementCompute;
    using EpilogueOutputOp =
        typename cutlass::epilogue::thread::MemoryEfficientAttentionNormalize<
            output_t,
            accum_t,
            DefaultOp::kCount,
            typename DefaultOp::ElementAccumulator,
            accum_t,
            true,
            true,
            cutlass::Array<ElementCompute, kQueriesPerBlock>>;
    using Epilogue =
        typename cutlass::epilogue::threadblock::EpiloguePipelined<
            typename DefaultEpilogue::Shape,
            typename MM1::Mma::Operator,
            DefaultEpilogue::kPartitionsK,
            typename MM1::OutputTileIterator,
            typename DefaultEpilogue::AccumulatorFragmentIterator,
            typename DefaultEpilogue::WarpTileIterator,
            typename DefaultEpilogue::SharedLoadIterator,
            EpilogueOutputOp,
            typename DefaultEpilogue::Padding,
            DefaultEpilogue::kFragmentsPerIteration,
            true,
            typename MM1::OutputTileIteratorAccum>;

    auto dest_iter = create_output_iter(0);
    EpilogueOutputOp rescale(s_prime, out_rescale);
    Epilogue epilogue(
        shared_storage.epilogue_shared_storage(),
        tid,
        my_warp_id,
        my_lane_id);
    epilogue(rescale, dest_iter, accum_o);
  }
};

template <typename Kernel>
__global__ void __launch_bounds__(Kernel::kNumThreads, Kernel::kMinBlocksPerSm)
ModelStackSm80CausalPrefillForwardKernel(typename Kernel::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ <= 1210
  if (!params.advance_to_block()) {
    return;
  }
  Kernel::attention_kernel(params);
  return;
#endif
#endif
  printf(
      "FATAL: model-stack SM80 inference attention kernel was built for sm80-sm121, but was built for sm%d\n",
      int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template <typename Kernel, typename scalar_t>
inline bool TryLaunchModelStackSm80InferenceAttentionKernel(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream) {
  const auto* props = at::cuda::getDeviceProperties(q_contig.get_device());
  if (props == nullptr) {
    return false;
  }

  const size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
  if (smem_bytes > static_cast<size_t>(props->sharedMemPerBlockOptin)) {
    return false;
  }

  typename Kernel::Params params;
  params.query_ptr =
      reinterpret_cast<const typename Kernel::element_t*>(q_contig.data_ptr<scalar_t>());
  params.key_ptr =
      reinterpret_cast<const typename Kernel::element_t*>(k_contig.data_ptr<scalar_t>());
  params.value_ptr =
      reinterpret_cast<const typename Kernel::element_t*>(v_contig.data_ptr<scalar_t>());
  params.output_ptr = reinterpret_cast<typename Kernel::output_t*>(out.data_ptr<scalar_t>());
  params.scale = scale_value;
  params.head_dim = static_cast<int32_t>(desc.head_dim);
  params.head_dim_value = static_cast<int32_t>(desc.head_dim);
  params.num_queries = static_cast<int32_t>(desc.q_len);
  params.num_keys = static_cast<int32_t>(desc.kv_len);
  params.q_strideM = static_cast<int32_t>(desc.head_dim);
  params.k_strideM = static_cast<int32_t>(desc.head_dim);
  params.v_strideM = static_cast<int32_t>(desc.head_dim);
  params.o_strideM = static_cast<int32_t>(desc.head_dim);
  params.q_strideB = static_cast<int64_t>(desc.q_len * desc.head_dim);
  params.k_strideB = static_cast<int64_t>(desc.kv_len * desc.head_dim);
  params.v_strideB = static_cast<int64_t>(desc.kv_len * desc.head_dim);
  params.o_strideB = static_cast<int64_t>(desc.q_len * desc.head_dim);
  params.num_batches = static_cast<int32_t>(desc.batch * desc.q_heads);

  if (!Kernel::check_supported(params)) {
    return false;
  }

  const dim3 blocks = params.getBlocksGrid();
  if (blocks.x == 0 || blocks.y == 0 || blocks.z == 0) {
    return true;
  }

  constexpr auto kernel_fn = ModelStackSm80CausalPrefillForwardKernel<Kernel>;
  if (smem_bytes > 0xc000) {
    (void)cudaFuncSetAttribute(
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  }
  kernel_fn<<<blocks, params.getThreadsGrid(), smem_bytes, stream>>>(params);
  return true;
}

template <typename scalar_t>
using ModelStackSm80InferenceKernel64x64 = ModelStackSm80CausalPrefillKernel<
    ModelStackSm80InferenceScalarT<scalar_t>,
    64,
    64,
    64>;

template <typename scalar_t>
inline bool TryLaunchModelStackSm80InferenceAttentionPrefillImpl(
    ModelStackSm80InferencePrefillKernelKind kind,
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream) {
  const auto try_64x64 = [&]() {
    return TryLaunchModelStackSm80InferenceAttentionKernel<
        ModelStackSm80InferenceKernel64x64<scalar_t>,
        scalar_t>(q_contig, k_contig, v_contig, out, desc, scale_value, stream);
  };

  switch (kind) {
    case ModelStackSm80InferencePrefillKernelKind::k64x64Rf:
      return try_64x64();
    case ModelStackSm80InferencePrefillKernelKind::kAuto:
    default:
      break;
  }

  // The 64x64 register-resident lane remains the only supported SM80
  // inference-prefill override. Wider query/key tiles have not beaten it on
  // real SM80/Ada workloads, and the 64x128 lane is unstable at long context.
  return try_64x64();
}

bool TryLaunchModelStackSm80InferenceAttentionPrefillF16(
    ModelStackSm80InferencePrefillKernelKind kind,
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream) {
  return TryLaunchModelStackSm80InferenceAttentionPrefillImpl<at::Half>(
      kind, q_contig, k_contig, v_contig, out, desc, scale_value, stream);
}

bool TryLaunchModelStackSm80InferenceAttentionPrefillBF16(
    ModelStackSm80InferencePrefillKernelKind kind,
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream) {
  return TryLaunchModelStackSm80InferenceAttentionPrefillImpl<at::BFloat16>(
      kind, q_contig, k_contig, v_contig, out, desc, scale_value, stream);
}

#else

bool TryLaunchModelStackSm80InferenceAttentionPrefillF16(
    ModelStackSm80InferencePrefillKernelKind,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const t10::desc::AttentionDesc&,
    float,
    cudaStream_t) {
  return false;
}

bool TryLaunchModelStackSm80InferenceAttentionPrefillBF16(
    ModelStackSm80InferencePrefillKernelKind,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const t10::desc::AttentionDesc&,
    float,
    cudaStream_t) {
  return false;
}

#endif

}  // namespace t10::cuda::attention
