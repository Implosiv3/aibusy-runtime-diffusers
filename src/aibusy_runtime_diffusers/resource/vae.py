from aibusy_runtime_diffusers.utils.torch_dtype import to_torch_dtype
from aibusy_runtime_diffusers.resource.diffusers_model import _DiffusersModelResource
from diffusers import AutoencoderKL

import torch


class VAEResource(
    _DiffusersModelResource
):

    async def load(
        self,
    ) -> AutoencoderKL:
        torch_dtype = to_torch_dtype(self.dtype)

        model = AutoencoderKL.from_pretrained(
            self.installed_asset.location.path,
            subfolder = 'vae',
            torch_dtype = torch_dtype,
        )

        return self.move_to_device(model)

    async def unload(
        self,
        instance: _DiffusersModelResource,
    ) -> None:
        del instance

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()