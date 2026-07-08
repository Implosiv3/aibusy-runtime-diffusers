from aibusy_runtime_diffusers.resource.scheduler.euler import EulerSchedulerResource
from aibusy_runtime_diffusers.resource.spec.scheduler.euler import EulerSchedulerResourceSpec
from aibusy.runtime.resource.builder.abstract import ResourceBuilder
from aibusy.engine.execution.context import ExecutionContext


class EulerSchedulerResourceBuilder(
    ResourceBuilder
):

    @property
    def spec_type(
        self,
    ) -> type[EulerSchedulerResourceSpec]:
        return EulerSchedulerResourceSpec

    async def build(
        self,
        resource_spec: EulerSchedulerResourceSpec,
        context: ExecutionContext,
    ) -> EulerSchedulerResource:
        destination_path = resource_spec.asset.get_install_path(
            context.settings.models_directory
        )

        installed_asset = await context.assets.install(
            spec = resource_spec.asset,
            destination_path = destination_path
        )

        return EulerSchedulerResource(
            installed_asset = installed_asset,
            device = resource_spec.device,
            dtype = resource_spec.dtype
        )