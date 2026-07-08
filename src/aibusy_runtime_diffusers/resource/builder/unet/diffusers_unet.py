from aibusy_runtime_diffusers.resource.unet.diffusers_unet import DiffusersUNetResource
from aibusy_runtime_diffusers.resource.spec.unet.diffusers_unet import DiffusersUNetResourceSpec
from aibusy.runtime.resource.builder.abstract import ResourceBuilder
from aibusy.engine.execution.context import ExecutionContext


class DiffusersUNetResourceBuilder(
    ResourceBuilder
    # ResourceBuilder[
    #     DiffusersUNetResourceSpec,
    #     DiffusersUNetResource,
    # ],
):

    @property
    def spec_type(
        self,
    ) -> type[DiffusersUNetResourceSpec]:
        return DiffusersUNetResourceSpec
    
    async def build(
        self,
        resource_spec: DiffusersUNetResourceSpec,
        context: ExecutionContext,
    ) -> DiffusersUNetResource:
        destination_path = resource_spec.asset.get_install_path(
            context.settings.models_directory
        )

        installed_asset = await context.assets.install(
            spec = resource_spec.asset,
            destination_path = destination_path
        )

        return DiffusersUNetResource(
            installed_asset = installed_asset,
            device = resource_spec.device,
            dtype = resource_spec.dtype
        )