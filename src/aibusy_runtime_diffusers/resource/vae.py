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

        return AutoencoderKL.from_pretrained(
            self.installed_asset.location.path,
            subfolder = 'vae',
            torch_dtype = torch_dtype,
        ).to(self.torch_device)

    async def unload(
        self,
        instance: _DiffusersModelResource,
    ) -> None:
        del instance

        torch.cuda.empty_cache()