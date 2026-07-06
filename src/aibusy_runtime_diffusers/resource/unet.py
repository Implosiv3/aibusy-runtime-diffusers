from aibusy_runtime_diffusers.utils.torch_dtype import to_torch_dtype
from aibusy_runtime_diffusers.resource.diffusers_model import _DiffusersModelResource
from diffusers import UNet2DConditionModel

import torch


class UNetResource(
    _DiffusersModelResource
):

    async def load(
        self,
    ) -> UNet2DConditionModel:
        torch_dtype = to_torch_dtype(self.dtype)

        model = UNet2DConditionModel.from_pretrained(
            self.installed_asset.location.path,
            subfolder = 'unet',
            torch_dtype = torch_dtype,
        )

        return self.move_to_device(model)

    async def unload(
        self,
        instance: UNet2DConditionModel,
    ) -> None:
        del instance

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()