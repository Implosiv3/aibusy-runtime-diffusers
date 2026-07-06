from aibusy.runtime.resource.builder.abstract import ResourceBuilder
from aibusy_runtime_diffusers.resource.vae import VAEResource
from aibusy_runtime_diffusers.resource.spec.vae import VAEResourceSpec
from aibusy.engine.execution.context import ExecutionContext


class VAEResourceBuilder(
    ResourceBuilder
):

    @property
    def spec_type(
        self,
    ):
        return VAEResourceSpec

    async def build(
        self,
        spec: VAEResourceSpec,
        context: ExecutionContext,
    ) -> VAEResource:
        destination_path = spec.asset.get_install_path(
            context.settings.models_directory
        )

        installed_asset = await context.assets.install(
            spec = spec.asset,
            destination_path = destination_path
        )

        return VAEResource(
            installed_asset = installed_asset,
            device = spec.device,
            dtype = spec.dtype,
        )