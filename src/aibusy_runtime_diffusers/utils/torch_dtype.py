from aibusy.runtime.dtype import DType, FLOAT16, BFLOAT16, FLOAT32
from typing import Union

import torch


TORCH_DTYPES = {
    FLOAT16: torch.float16,
    BFLOAT16: torch.bfloat16,
    FLOAT32: torch.float32,
}

def to_torch_dtype(
    dtype: DType,
) -> Union[torch.dtype, None]:
    return TORCH_DTYPES.get(dtype, None)