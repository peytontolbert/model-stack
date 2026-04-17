import sys

import runtime.block_registry as _runtime_block_registry

sys.modules[__name__] = _runtime_block_registry
