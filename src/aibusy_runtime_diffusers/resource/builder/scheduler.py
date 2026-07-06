from aibusy_runtime_diffusers.resource.scheduler import SchedulerResource
from aibusy_runtime_diffusers.resource.spec.scheduler import SchedulerResourceSpec
from aibusy.runtime.resource.builder.abstract import ResourceBuilder
from aibusy.engine.execution.context import ExecutionContext


class SchedulerResourceBuilder(
    ResourceBuilder,
):

    @property
    def spec_type(
        self,
    ) -> type[SchedulerResourceSpec]:
        return SchedulerResourceSpec

    async def build(
        self,
        spec: SchedulerResourceSpec,
        context: ExecutionContext,
    ) -> SchedulerResource:
        destination_path = spec.asset.get_install_path(
            context.settings.models_directory
        )

        installed_asset = await context.assets.install(
            spec = spec.asset,
            destination_path = destination_path
        )

        return SchedulerResource(
            installed_asset = installed_asset,
            scheduler = spec.scheduler
        )