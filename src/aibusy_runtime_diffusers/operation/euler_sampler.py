from aibusy.graph.operation.abstract.atomic_operation import AtomicOperation
from aibusy.graph.classes.input import Input
from aibusy.graph.classes.output import Output
from aibusy.engine.execution.context import ExecutionContext
from aibusy.graph.data_type.types import INT, LATENTS, PROMPT_EMBEDDINGS, UNET_RESOURCE_REFERENCE, SCHEDULER_RESOURCE_REFERENCE, FLOAT, NOISE
from aibusy.utils.classes.latents import Latents
import torch


class EulerSampler(
    AtomicOperation
):
    """
    *Atomic Operation*

    Predict noise using an UNet.

    Inputs:
    - `unet` (`UNET_RESOURCE_REFERENCE`)
    - `scheduler` (`SCHEDULER_RESOURCE_REFERENCE`)
    - `noise` (`NOISE`)
    - `positive_prompt_embeddings` (`PROMPT_EMBEDDINGS`)
    - `negative_prompt_embeddings` (`PROMPT_EMBEDDINGS`)
    - `steps` (`INT`)
    - `cfg_scale` (`FLOAT`)

    Outputs:
    - `latents` (`LATENTS`)
    """

    unet = Input(UNET_RESOURCE_REFERENCE)
    scheduler = Input(SCHEDULER_RESOURCE_REFERENCE)
    noise = Input(NOISE)
    positive_prompt_embeddings = Input(PROMPT_EMBEDDINGS)
    negative_prompt_embeddings = Input(PROMPT_EMBEDDINGS)
    steps = Input(INT)
    cfg_scale = Input(FLOAT)

    latents = Output(LATENTS)

    async def execute(
        self,
        context: ExecutionContext,
    ):
        with torch.inference_mode():
            unet = await context.resources.resolve(self.unet.resource_spec)
            scheduler = await context.resources.resolve(self.scheduler.resource_spec)
            latents = self.noise.value

            unet = unet.unet
            scheduler = scheduler.scheduler

            # Initialize scheduler
            scheduler.set_timesteps(
                num_inference_steps = self.steps,
            )

            # Prepare embeddings
            prompt_embeddings = torch.cat(
                (
                    self.negative_prompt_embeddings.value,
                    self.positive_prompt_embeddings.value,
                ),
                dim = 0,
            )

            for timestep in scheduler.timesteps:
                latent_model_input = scheduler.scale_model_input(
                    sample = latents,
                    timestep = timestep,
                )

                latent_model_input = torch.cat(
                    (
                        latent_model_input,
                        latent_model_input,
                    ),
                    dim = 0,
                )

                predicted_noise = unet(
                    sample = latent_model_input,
                    timestep = timestep,
                    encoder_hidden_states = prompt_embeddings,
                ).sample

                # Separate the 2 predictions
                predicted_noise_negative, predicted_noise_positive = (
                    predicted_noise.chunk(2)
                )

                # Apply the CFG
                predicted_noise = (
                    predicted_noise_negative
                    + self.cfg_scale
                    * (
                        predicted_noise_positive
                        - predicted_noise_negative
                    )
                )

                result = scheduler.step(
                    model_output = predicted_noise,
                    timestep = timestep,
                    sample = latents,
                )

                latents = result.prev_sample

            latents = Latents(latents)

        return {
            'latents': latents
        }