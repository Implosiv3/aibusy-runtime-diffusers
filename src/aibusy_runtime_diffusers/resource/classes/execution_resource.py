from aibusy_runtime_diffusers.utils.torch_dtype import to_torch_dtype
from aibusy.runtime.device import Device
from aibusy.runtime.dtype import DType
from aibusy.runtime.resource.abstract import Resource
from typing import TypeVar
from abc import ABC


T = TypeVar('T')

class _ExecutionResource(
    Resource[T],
    ABC,
):
    """
    *For internal use only*

    The resource that includes `device` and
    `dtype` to control how it is executed.
    """

    @property
    def torch_device(
        self,
    ):
        return str(self.device)

    @property
    def torch_dtype(
        self,
    ):
        return to_torch_dtype(
            self.dtype
        )
    
    def __init__(
        self,
        *,
        device: Device,
        dtype: DType,
    ):
        self.device = device
        self.dtype = dtype

    def move_to_device(
        self,
        model,
    ):
        """
        Move the `model` to the `torch_device` by using
        a `model.to(self.torch_device)`.
        """
        model.to(self.torch_device)

        return model
    