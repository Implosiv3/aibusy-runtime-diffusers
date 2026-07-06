from aibusy_runtime_diffusers.asset.installer.diffusers_checkpoint import DiffusersCheckpointAssetInstaller
from aibusy_runtime_diffusers.resource.builder.unet import UNetResourceBuilder
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
