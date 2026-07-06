"""
TODO: What do we do with this (?)
"""
from aibusy.runtime.interface.unet import UNet
from diffusers import UNet2DConditionModel


class DiffusersUNet(
    UNet
):

    def __init__(
        self,
        module: UNet2DConditionModel
    ):
        self._module = module

    async def infer(
        self,
        *,
        latents,
        timestep,
        encoder_hidden_states,
    ):
        return self._module(
            latents,
            timestep,
            encoder_hidden_states = encoder_hidden_states
        ).sample