from diffusers.models import UNet2DConditionModel


class DiffusersUNet:

    def __init__(
        self,
        unet: UNet2DConditionModel,
    ):
        self._unet = unet

    @property
    def unet(
        self,
    ) -> UNet2DConditionModel:
        return self._unet