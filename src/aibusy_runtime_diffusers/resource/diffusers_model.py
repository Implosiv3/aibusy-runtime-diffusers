from aibusy_runtime_diffusers.utils.torch_dtype import to_torch_dtype
from aibusy.runtime.resource.abstract import Resource
from aibusy.engine.execution.asset.installed import InstalledAsset
from aibusy.runtime.device import Device
from aibusy.runtime.dtype import DType
from abc import ABC
from pathlib import Path


class _DiffusersModelResource(
    Resource,
    ABC,
):
    """
    *For internal use only*

    Class to inherit from it and avoid repeating
    code. Any diffuser model (vae, unet, etc.) must
    inherit from it.
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
        installed_asset: InstalledAsset,
        device: Device,
        dtype: DType,
    ):
        self.installed_asset = installed_asset
        self.device = device
        self.dtype = dtype

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