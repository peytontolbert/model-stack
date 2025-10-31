## activations

Activation functions and variants commonly used in transformer MLPs and gates.

### Key APIs
- `gelu`, `silu`, `bias_gelu`, `bias_silu`
- Variants: `fast_gelu`, `quick_gelu`, `mish`, `tanh_gelu`, `swiglu`, `geglu`, `reglu`

### Notes
- Bias-fused variants can reduce memory bandwidth by combining bias add with activation.


