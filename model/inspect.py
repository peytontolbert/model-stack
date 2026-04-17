import sys

import runtime.inspect as _runtime_inspect

sys.modules[__name__] = _runtime_inspect
