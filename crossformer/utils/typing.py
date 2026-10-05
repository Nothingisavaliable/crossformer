from typing import Any, Mapping, Sequence, Union

import jax

# ``KeyArray`` was removed from the public API in newer JAX releases. Typed
# and legacy PRNG keys are both JAX arrays for the purposes of these aliases.
PRNGKey = jax.Array
PyTree = Union[jax.typing.ArrayLike, Mapping[str, "PyTree"]]
Config = Union[Any, Mapping[str, "Config"]]
Params = Mapping[str, PyTree]
Data = Mapping[str, PyTree]
Shape = Sequence[int]
Dtype = jax.typing.DTypeLike
