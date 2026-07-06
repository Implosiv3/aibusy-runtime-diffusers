from aibusy.runtime.resource.builder.abstract import ResourceBuilder
from aibusy_runtime_diffusers.resource.unet import UNetResource
from aibusy_runtime_diffusers.resource.spec.unet import UNetResourceSpec
from aibusy.engine.execution.context import ExecutionContext


class UNetResourceBuilder(
    ResourceBuilder
):

    @property
    def spec_type(
        self,
    ):
        return UNetResourceSpec

    async def build(
        self,
        spec: UNetResourceSpec,
        context: ExecutionContext,
    ) -> UNetResource:
        destination_path = spec.asset.get_install_path(
            context.settings.models_directory
        )

        installed_asset = await context.assets.install(
            spec = spec.asset,
            destination_path = destination_path
        )

        return UNetResource(
            installed_asset = installed_asset,
            device = spec.device,
            dtype = spec.dtype,
        )