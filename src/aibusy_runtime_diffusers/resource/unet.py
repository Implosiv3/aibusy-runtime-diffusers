from aibusy.runtime.resource.abstract import Resource
from aibusy.engine.execution.asset.installed import InstalledAsset
from aibusy_runtime_diffusers.runtime.torch_dtype import to_torch_dtype
from aibusy.runtime.device import Device
from aibusy.runtime.dtype import DType
from diffusers import UNet2DConditionModel

import torch


class UNetResource(
    Resource[UNet2DConditionModel]
):

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

    async def load(
        self,
    ) -> UNet2DConditionModel:
        torch_dtype = to_torch_dtype(self.dtype)

        return UNet2DConditionModel.from_pretrained(
            self.installed_asset.location.path,
            subfolder = 'unet',
            torch_dtype = torch_dtype,
        ).to(str(self.device))

    async def unload(
        self,
        instance: UNet2DConditionModel,
    ) -> None:
        del instance

        torch.cuda.empty_cache()