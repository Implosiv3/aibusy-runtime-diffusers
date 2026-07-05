from aibusy_runtime_diffusers.asset.installer.diffusers_model_asset_installer import DiffusersModelAssetInstaller
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
        builder: EngineBuilder
    ):
        builder._assets_installers.register(
            DiffusersModelAssetInstaller(
                builder.services.get(
                    HuggingfaceClient
                )
            )
        )