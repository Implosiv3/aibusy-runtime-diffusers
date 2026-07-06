from aibusy_runtime_diffusers.asset.spec.checkpoint import DiffusersCheckpointAssetSpec
from aibusy.engine.execution.asset.installer.abstract import AssetInstaller
from aibusy.engine.execution.asset.installed import InstalledAsset
from aibusy.engine.execution.asset.location import AssetLocation
from aibusy.service.huggingface.abstract import HuggingfaceClient
from pathlib import Path


class DiffusersCheckpointAssetInstaller(
    AssetInstaller
):

    spec_type = DiffusersCheckpointAssetSpec

    def __init__(
        self,
        huggingface: HuggingfaceClient,
    ):
        self._huggingface = huggingface

    async def install(
        self,
        spec: DiffusersCheckpointAssetSpec,
    ) -> InstalledAsset:
        # TODO: I need to receive the 'MODELS_DIRECTORY'
        # del EngineSettings...
        path = spec.get_install_path(Path('C:/Users/dania/.aibusy/models'))
        
        directory = await self._huggingface.download_snapshot(
            repository = spec.repository,
            revision = spec.revision,
            local_dir = path
        )

        return InstalledAsset(
            spec = spec,
            location = AssetLocation(
                uri = f'{directory.resolve().as_uri()}'
            )
        )