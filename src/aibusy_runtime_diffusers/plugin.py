from aibusy_runtime_diffusers.asset.installer.diffusers_checkpoint import DiffusersCheckpointAssetInstaller
from aibusy_runtime_diffusers.resource.builder.unet import UNetResourceBuilder
from aibusy_runtime_diffusers.resource.builder.vae import VAEResourceBuilder
from aibusy_runtime_diffusers.resource.builder.text_encoder import TextEncoderResourceBuilder
from aibusy_runtime_diffusers.resource.builder.tokenizer import TokenizerResourceBuilder
from aibusy_runtime_diffusers.resource.builder.scheduler import SchedulerResourceBuilder
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
            UNetResourceBuilder()
        )

        builder.resources_builders.register(
            VAEResourceBuilder()
        )

        builder.resources_builders.register(
            TextEncoderResourceBuilder()
        )

        builder.resources_builders.register(
            TokenizerResourceBuilder()
        )

        builder.resources_builders.register(
            SchedulerResourceBuilder()
        )
