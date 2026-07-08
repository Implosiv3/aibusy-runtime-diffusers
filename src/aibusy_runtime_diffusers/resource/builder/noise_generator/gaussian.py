from aibusy_runtime_diffusers.resource.noise_generator.gaussian_noise_generator import GausianNoiseGeneratorResource
from aibusy_runtime_diffusers.resource.spec.noise_generator.gaussian import GaussianNoiseGeneratorResourceSpec
from aibusy.runtime.resource.builder.abstract import ResourceBuilder
from aibusy.engine.execution.context import ExecutionContext


class GaussianNoiseGeneratorResourceBuilder(
    ResourceBuilder,
):

    @property
    def spec_type(
        self,
    ) -> type[GaussianNoiseGeneratorResourceSpec]:
        return GaussianNoiseGeneratorResourceSpec

    async def build(
        self,
        resource_spec: GaussianNoiseGeneratorResourceSpec,
        context: ExecutionContext,
    ) -> GausianNoiseGeneratorResource:
        return GausianNoiseGeneratorResource(
            device = resource_spec.device,
            dtype = resource_spec.dtype,
        )