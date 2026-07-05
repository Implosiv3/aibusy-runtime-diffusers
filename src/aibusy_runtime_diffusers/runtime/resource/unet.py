from aibusy_runtime_diffusers.resource.spec.unet_resource_spec import UNetResourceSpec
from aibusy_runtime_diffusers.unet import DiffusersUNet
from aibusy.runtime.resource.abstract import Resource
from aibusy.runtime.interface.unet import UNet
from diffusers import UNet2DConditionModel


class DiffusersUNetResource(
    Resource[UNet]
):

    def __init__(
        self,
        spec: UNetResourceSpec,
    ):
        self._spec = spec

    async def load(
        self,
    ) -> UNet:
        module = UNet2DConditionModel.from_pretrained(
            self._spec.asset.location.path,
            subfolder = 'unet',
            torch_dtype = self._spec.dtype,
        )

        module.to(self._spec.device)

        module.eval()

        return DiffusersUNet(module)