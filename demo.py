from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from nkpack.exceptions import InvalidParameterError

def dec2bin(x: int, sz: int) -> NDArray[np.int8]:
    output = []
    while x > 0:
        output.insert(0, x%2)
        x = int(x/2)
    if len(output)<sz:
        output = [0]*(sz-len(output)) + output
    return np.array(output, dtype=np.int8)

def func(decimal: int, len_: int) -> NDArray[np.int8]:
    if decimal >= 2 ** len_:
        raise InvalidParameterError('The binary representation of this number will not fit into the given length')

    binary = (decimal // 2**np.arange(len_)[::-1]) % 2
    return binary.astype(dtype=np.int8)

a = 370
print(dec2bin(a, 16))
print(func(a,16))
