from aibusy_runtime_diffusers.resource.diffusers import _DiffusersResource
from aibusy_runtime_diffusers.utils.torch_dtype import to_torch_dtype
from aibusy.engine.execution.asset.installed import InstalledAsset
from aibusy.runtime.device import Device
from aibusy.runtime.dtype import DType
from abc import ABC


class _DiffusersModelResource(
    _DiffusersResource,
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