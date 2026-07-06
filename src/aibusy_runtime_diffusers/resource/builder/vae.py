from aibusy_runtime_diffusers.runtime.resource.vae import DiffusersVAEResource
from aibusy_runtime_diffusers.resource.spec.vae import VAEResourceSpec
from aibusy.runtime.resource.builder.abstract import ResourceBuilder
from aibusy.engine.execution.context import ExecutionContext


class DiffusersVAEResourceBuilder(
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
    ) -> DiffusersVAEResource:
        ...