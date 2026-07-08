from aibusy.graph.operation.abstract.atomic_operation import AtomicOperation
from aibusy.graph.classes.input import Input
from aibusy.graph.classes.output import Output
from aibusy.utils.classes.noise import Noise
from aibusy.engine.execution.context import ExecutionContext
from aibusy.graph.data_type.types import INT, NOISE, GAUSSIAN_NOISE_GENERATOR_RESOURCE_REFERENCE


class GenerateGaussianNoise(
    AtomicOperation
):
    """
    *Atomic Operation*

    Generate gaussian noise.

    Inputs:
    - `gaussian_noise_generator` (`GAUSSIAN_NOISE_GENERATOR_RESOURCE_REFERENCE`)
    - `seed` (`INT`)
    - `width` (`INT`)
    - `height` (`INT`)
    - `batch_size` (`INT`)
    - `channels` (`INT`)

    Outputs:
    - `noise` (`NOISE`)
    """

    gaussian_noise_generator = Input(GAUSSIAN_NOISE_GENERATOR_RESOURCE_REFERENCE)
    seed = Input(INT)
    width = Input(INT)
    height = Input(INT)
    batch_size = Input(INT)
    channels = Input(INT)

    noise = Output(NOISE)

    async def execute(
        self,
        context: ExecutionContext,
    ):
        resource = await context.resources.resolve(self.gaussian_noise_generator.resource_spec)

        noise = resource.generate(
            seed = self.seed,
            batch_size = self.batch_size,
            channels = self.channels,
            height = self.height,
            width = self.width
        )

        noise = Noise(noise)

        return {
            'noise': noise
        }