
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

// Python bindings to call kernel():
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <sstream>
#include <cstdlib>

#ifndef _MSC_VER
#if __cplusplus < 202002L
// C++20 (earlier) code
// https://en.cppreference.com/w/cpp/language/attributes/likely
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#endif
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

// This is defined in guards.cpp so we don't need to import PyTorch headers that are slooow.
// We manually link it below to workaround issues with fbcode build.
static void* (*_torchinductor_pyobject_tensor_data_ptr)(PyObject* obj);

template <typename T> static inline T parse_arg(PyObject* args, size_t n) {
    static_assert(std::is_pointer_v<T>, "arg type must be pointer or long");
    return static_cast<T>(_torchinductor_pyobject_tensor_data_ptr(PyTuple_GET_ITEM(args, n)));
}
template <> inline int64_t parse_arg<int64_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, n));
    if(unlikely(result == -1 && PyErr_Occurred()))
        throw std::runtime_error("expected int arg");
    return result;
}
template <> inline uintptr_t parse_arg<uintptr_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsVoidPtr(PyTuple_GET_ITEM(args, n));
    if(unlikely(result == reinterpret_cast<void*>(-1) && PyErr_Occurred()))
        throw std::runtime_error("expected int arg");
    return reinterpret_cast<uintptr_t>(result);
}
template <> inline float parse_arg<float>(PyObject* args, size_t n) {
    auto result = PyFloat_AsDouble(PyTuple_GET_ITEM(args, n));
    if(unlikely(result == -1.0 && PyErr_Occurred()))
        throw std::runtime_error("expected float arg");
    return static_cast<float>(result);
}



static PyObject* kernel_py(PyObject* self, PyObject* args) {
    try {
        if(unlikely(!PyTuple_CheckExact(args)))
            throw std::runtime_error("tuple args required");
        if(unlikely(PyTuple_GET_SIZE(args) != 2))
            throw std::runtime_error("requires 2 args");
        kernel(parse_arg<uint8_t*>(args, 0), parse_arg<at::BFloat16*>(args, 1)); Py_RETURN_NONE;
    } catch(std::exception const& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch(...) {
        PyErr_SetString(PyExc_RuntimeError, "unhandled error");
        return nullptr;
    }
}

static PyMethodDef py_methods[] = {
    {"kernel", kernel_py, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef py_module =
    {PyModuleDef_HEAD_INIT, "kernel", NULL, -1, py_methods};

PyMODINIT_FUNC PyInit_kernel(void) {
    const char* str_addr = std::getenv("_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR");
    if(!str_addr) {
        PyErr_SetString(PyExc_RuntimeError, "_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR must be set");
        return nullptr;
    }
    std::istringstream iss(str_addr);
    uintptr_t addr = 0;
    iss >> addr;
    _torchinductor_pyobject_tensor_data_ptr =
        reinterpret_cast<decltype(_torchinductor_pyobject_tensor_data_ptr)>(addr);
    PyObject* module = PyModule_Create(&py_module);
    if (module == NULL) {
        return NULL;
    }
    #ifdef Py_GIL_DISABLED
        PyUnstable_Module_SetGIL(module, Py_MOD_GIL_NOT_USED);
    #endif
    return module;
}
