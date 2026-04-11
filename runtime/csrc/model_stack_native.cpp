#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int kAbiVersion = MODEL_STACK_ABI_VERSION;

std::map<std::string, bool> NativeOpMap() {
  return {
      {"rms_norm", false},
      {"rope", false},
      {"kv_cache_append", false},
      {"attention_decode", false},
      {"attention_prefill", false},
      {"sampling", false},
  };
}

py::dict RuntimeInfo() {
  py::dict info;
  info["abi_version"] = kAbiVersion;
#if MODEL_STACK_WITH_CUDA
  info["compiled_with_cuda"] = true;
#else
  info["compiled_with_cuda"] = false;
#endif
  info["native_ops"] = std::vector<std::string>{};
  info["planned_ops"] = std::vector<std::string>{
      "rms_norm", "rope", "kv_cache_append", "attention_decode",
      "attention_prefill", "sampling"};
  return info;
}

bool HasOp(const std::string& name) {
  const auto ops = NativeOpMap();
  const auto it = ops.find(name);
  return it != ops.end() && it->second;
}

[[noreturn]] void RaiseNotImplemented(const char* message) {
  PyErr_SetString(PyExc_NotImplementedError, message);
  throw py::error_already_set();
}

py::object RmsNormForward(py::object /*x*/, py::object /*weight*/, double /*eps*/) {
  RaiseNotImplemented(
      "Native RMSNorm is not compiled yet. Build the CUDA/C++ kernel and enable "
      "the op before routing tensors here.");
}

}  // namespace

PYBIND11_MODULE(_model_stack_native, m) {
  m.doc() = "Model-stack native C++/CUDA extension boundary.";
  m.def("runtime_info", &RuntimeInfo);
  m.def("has_op", &HasOp, py::arg("name"));
  m.def("rms_norm_forward", &RmsNormForward, py::arg("x"), py::arg("weight") = py::none(),
        py::arg("eps") = 1e-6);
}
