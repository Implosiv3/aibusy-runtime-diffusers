from aibusy_runtime_diffusers.resource.spec.vae_resource_spec import VAEResourceSpec
from aibusy_runtime_diffusers.vae import DiffusersVAE
from aibusy.runtime.resource.abstract import Resource
from aibusy.runtime.interface.vae import VAE
from diffusers import AutoencoderKL


class DiffusersVAEResource(
    Resource[VAE]
):

    def __init__(
        self,
        spec: VAEResourceSpec,
    ):
        self._spec = spec

    async def load(
        self,
    ) -> VAE:
        module = AutoencoderKL.from_pretrained(
            self._spec.asset.location.path,
            subfolder = 'vae',
            torch_dtype = self._spec.dtype,
        )

        module.to(self._spec.device)

        module.eval()

        return DiffusersVAE(module)