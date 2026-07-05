from aibusy.runtime.interface.vae import VAE
from diffusers import AutoencoderKL


class DiffusersVAE(
    VAE
):

    def __init__(
        self,
        module: AutoencoderKL,
    ):
        self._module = module

    async def encode(
        self,
        image,
    ):
        return (
            self._module
            .encode(image)
            .latent_dist
            .sample()
        )

    async def decode(
        self,
        latents,
    ):
        return (
            self._module
            .decode(latents)
            .sample
        )