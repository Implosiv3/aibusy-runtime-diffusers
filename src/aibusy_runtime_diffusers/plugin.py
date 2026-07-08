from aibusy_runtime_diffusers.asset.installer.diffusers_checkpoint import DiffusersCheckpointAssetInstaller
from aibusy_runtime_diffusers.resource.builder.unet.diffusers_unet import DiffusersUNetResourceBuilder
from aibusy_runtime_diffusers.resource.builder.prompt_encoder.clip import CLIPPromptEncoderResourceBuilder
from aibusy_runtime_diffusers.resource.builder.noise_generator.gaussian import GaussianNoiseGeneratorResourceBuilder
# from aibusy_runtime_diffusers.resource.builder.scheduler import SchedulerResourceBuilder
from aibusy_runtime_diffusers.resource.builder.scheduler.euler import EulerSchedulerResourceBuilder
from aibusy_runtime_huggingface.plugin import HuggingfaceRuntimePlugin
from aibusy.service.huggingface.abstract import HuggingfaceClient
from aibusy.engine.plugin.abstract import Plugin
from aibusy.engine.builder import EngineBuilder


class DiffusersRuntimePlugin(
    Plugin
):

    @property
    def dependencies(
        self
    ):
        return (
            HuggingfaceRuntimePlugin,
        )
    
    def register(
        self,
        builder: EngineBuilder
    ):
        builder.assets_installers.register(
            DiffusersCheckpointAssetInstaller(
                builder.services.get(
                    HuggingfaceClient
                )
            )
        )

        builder.resources_builders.register(
            CLIPPromptEncoderResourceBuilder()
        )
        
        builder.resources_builders.register(
            EulerSchedulerResourceBuilder()
        )

        # builder.resources_builders.register(
        #     SchedulerResourceBuilder()
        # )

        builder.resources_builders.register(
            GaussianNoiseGeneratorResourceBuilder()
        )

        builder.resources_builders.register(
            DiffusersUNetResourceBuilder()
        )
