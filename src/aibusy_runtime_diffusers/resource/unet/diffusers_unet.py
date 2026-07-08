from aibusy_runtime_diffusers.resource.classes.asset_execution_resource import _AssetExecutionResource
from aibusy_runtime_diffusers.resource.implementation.diffusers_unet import DiffusersUNet
from diffusers.models import UNet2DConditionModel


class DiffusersUNetResource(
    _AssetExecutionResource[
        DiffusersUNet,
    ],
):

    async def load(
        self,
    ) -> DiffusersUNet:
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path = self.installed_asset.location.path,
            subfolder = 'unet',
            torch_dtype = self.torch_dtype,
        )

        # unet.to(str(self.device.value))

        unet = self.move_to_device(unet)

        return DiffusersUNet(
            unet = unet,
        )