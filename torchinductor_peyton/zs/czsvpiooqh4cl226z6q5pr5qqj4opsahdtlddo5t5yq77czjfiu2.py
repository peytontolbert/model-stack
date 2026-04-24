# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cpp_fused___rshift____to_copy_bitwise_and_sub_zeros_0 = async_compile.cpp_pybinding(['const uint8_t*', 'at::BFloat16*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const uint8_t* in_ptr0,
                       at::BFloat16* out_ptr1)
{
    #pragma omp parallel num_threads(24)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2560L); x0+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(6912L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = x0;
                            auto tmp1 = c10::convert<int64_t>(tmp0);
                            auto tmp2 = static_cast<int64_t>(1280);
                            auto tmp3 = tmp1 >= tmp2;
                            auto tmp4 = static_cast<int64_t>(1920);
                            auto tmp5 = tmp1 < tmp4;
                            auto tmp6 = tmp3 & tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr0[static_cast<int64_t>((-8847360L) + x1 + 6912L*x0)];
                                auto tmp9 = static_cast<uint8_t>(48);
                                auto tmp10 = decltype(tmp8)(tmp8 & tmp9);
                                auto tmp11 = static_cast<uint8_t>(4);
                                auto tmp12 =
                                [&]()
                                {
                                    constexpr decltype(tmp11) max_shift = sizeof(uint8_t) * CHAR_BIT - std::is_signed_v<uint8_t>;
                                    if ((static_cast<std::make_signed_t<uint8_t>>(tmp11) < 0) || (tmp11 >= max_shift))
                                    {
                                        return decltype(tmp10)(tmp10 >> max_shift);
                                    }
                                    return decltype(tmp10)(tmp10 >> tmp11);
                                }
                                ()
                                ;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0);
                            auto tmp14 = static_cast<int64_t>(640);
                            auto tmp15 = tmp1 >= tmp14;
                            auto tmp16 = tmp1 < tmp2;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = in_ptr0[static_cast<int64_t>((-4423680L) + x1 + 6912L*x0)];
                                auto tmp20 = static_cast<uint8_t>(12);
                                auto tmp21 = decltype(tmp19)(tmp19 & tmp20);
                                auto tmp22 = static_cast<uint8_t>(2);
                                auto tmp23 =
                                [&]()
                                {
                                    constexpr decltype(tmp22) max_shift = sizeof(uint8_t) * CHAR_BIT - std::is_signed_v<uint8_t>;
                                    if ((static_cast<std::make_signed_t<uint8_t>>(tmp22) < 0) || (tmp22 >= max_shift))
                                    {
                                        return decltype(tmp21)(tmp21 >> max_shift);
                                    }
                                    return decltype(tmp21)(tmp21 >> tmp22);
                                }
                                ()
                                ;
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp17 ? tmp18() : static_cast<decltype(tmp18())>(0);
                            auto tmp25 = tmp1 < tmp14;
                            auto tmp26 = [&]
                            {
                                auto tmp27 = in_ptr0[static_cast<int64_t>(x1 + 6912L*x0)];
                                auto tmp28 = static_cast<uint8_t>(3);
                                auto tmp29 = decltype(tmp27)(tmp27 & tmp28);
                                auto tmp30 = static_cast<uint8_t>(0);
                                auto tmp31 =
                                [&]()
                                {
                                    constexpr decltype(tmp30) max_shift = sizeof(uint8_t) * CHAR_BIT - std::is_signed_v<uint8_t>;
                                    if ((static_cast<std::make_signed_t<uint8_t>>(tmp30) < 0) || (tmp30 >= max_shift))
                                    {
                                        return decltype(tmp29)(tmp29 >> max_shift);
                                    }
                                    return decltype(tmp29)(tmp29 >> tmp30);
                                }
                                ()
                                ;
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp25 ? tmp26() : static_cast<decltype(tmp26())>(0);
                            auto tmp33 = static_cast<uint8_t>(0);
                            auto tmp34 = tmp25 ? tmp32 : tmp33;
                            auto tmp35 = tmp17 ? tmp24 : tmp34;
                            auto tmp36 = tmp6 ? tmp13 : tmp35;
                            auto tmp37 = tmp1 >= tmp4;
                            auto tmp38 = [&]
                            {
                                auto tmp39 = in_ptr0[static_cast<int64_t>((-13271040L) + x1 + 6912L*x0)];
                                auto tmp40 = static_cast<uint8_t>(192);
                                auto tmp41 = decltype(tmp39)(tmp39 & tmp40);
                                auto tmp42 = static_cast<uint8_t>(6);
                                auto tmp43 =
                                [&]()
                                {
                                    constexpr decltype(tmp42) max_shift = sizeof(uint8_t) * CHAR_BIT - std::is_signed_v<uint8_t>;
                                    if ((static_cast<std::make_signed_t<uint8_t>>(tmp42) < 0) || (tmp42 >= max_shift))
                                    {
                                        return decltype(tmp41)(tmp41 >> max_shift);
                                    }
                                    return decltype(tmp41)(tmp41 >> tmp42);
                                }
                                ()
                                ;
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp37 ? tmp38() : static_cast<decltype(tmp38())>(0);
                            auto tmp45 = tmp37 ? tmp44 : tmp36;
                            auto tmp46 = c10::convert<float>(tmp45);
                            auto tmp47 = static_cast<float>(1.0);
                            auto tmp48 = float(tmp46 - tmp47);
                            auto tmp49 = c10::convert<at::BFloat16>(tmp48);
                            out_ptr1[static_cast<int64_t>(x1 + 6912L*x0)] = tmp49;
                        }
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, = args
        args.clear()
        assert_size_stride(arg0_1, (640, 6912), (6912, 1))
        buf1 = empty_strided_cpu((2560, 6912), (6912, 1), torch.bfloat16)
        cpp_fused___rshift____to_copy_bitwise_and_sub_zeros_0(arg0_1, buf1)
        del arg0_1
        return (buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((640, 6912), (6912, 1), device='cpu', dtype=torch.uint8)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
