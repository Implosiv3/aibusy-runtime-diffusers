from aibusy_runtime_diffusers.asset.spec.diffusers_checkpoint import DiffusersCheckpointAssetSpec
from aibusy.engine.execution.asset.installer.abstract import AssetInstaller
from aibusy.engine.execution.asset.installed import InstalledAsset
from aibusy.engine.execution.asset.location import AssetLocation
from aibusy.service.huggingface.abstract import HuggingfaceClient


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
        install_path: str
    ) -> InstalledAsset:
        directory = await self._huggingface.download_snapshot(
            repository = spec.repository,
            revision = spec.revision,
            local_dir = install_path,
            allow_patterns = spec.allow_patterns,
            ignore_patterns = spec.ignore_patterns
        )

        return InstalledAsset(
            spec = spec,
            location = AssetLocation(
                uri = f'{directory.resolve().as_uri()}'
            )
        )