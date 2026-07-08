from aibusy_runtime_diffusers.resource.implementation.gaussian_noise_generator import GaussianNoiseGenerator
from aibusy_runtime_diffusers.resource.classes.execution_resource import _ExecutionResource


class GausianNoiseGeneratorResource(
    _ExecutionResource[
        GaussianNoiseGenerator
    ]
):
    
    async def load(
        self,
    ) -> GaussianNoiseGenerator:
        """
        The 'device' and 'dtype' is also in the
        '_DiffusersModelResource', maybe we could
        create a mixin for 'device' and 'dtype'...
        """
        return GaussianNoiseGenerator(
            # TODO: This could be in a mixin or something
            device = self.device,
            dtype = self.dtype
        )

    async def unload(
        self,
        instance: GaussianNoiseGenerator,
    ):
        pass