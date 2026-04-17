import sys

import runtime.kv_cache as _runtime_kv_cache

sys.modules[__name__] = _runtime_kv_cache
