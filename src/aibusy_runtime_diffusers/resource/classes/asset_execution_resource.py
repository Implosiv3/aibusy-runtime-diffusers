from aibusy_runtime_diffusers.utils.torch_dtype import to_torch_dtype
from aibusy.engine.execution.asset.installed import InstalledAsset
from aibusy.runtime.device import Device
from aibusy.runtime.dtype import DType
from aibusy.runtime.resource.abstract import Resource
from pathlib import Path
from typing import TypeVar
from abc import ABC


T = TypeVar('T')

class _AssetExecutionResource(
    Resource[T],
    ABC,
):
    """
    *For internal use only*

    The resource that includes an `installed_asset`
    but also `device` and `dtype`.
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
    
    @property
    def model_directory(
        self,
    ) -> Path:
        """
        Root directory of the installed checkpoint.
        """
        return Path(
            self.installed_asset.location.path
        )

    def subdirectory(
        self,
        name: str,
    ) -> Path:
        """
        Returns a subdirectory inside the checkpoint.
        """
        return self.model_directory / name
    
    def __init__(
        self,
        *,
        installed_asset: InstalledAsset,
        device: Device,
        dtype: DType,
    ):
        self.installed_asset = installed_asset
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
    