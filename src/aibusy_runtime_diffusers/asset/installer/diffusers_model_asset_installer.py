from aibusy.engine.execution.asset.installer.abstract import AssetInstaller
from aibusy.engine.execution.asset.spec.diffusers_model_asset_spec import DiffusersModelAssetSpec
from aibusy.engine.execution.asset.installed import InstalledAsset
from aibusy.engine.execution.asset.location import AssetLocation
from aibusy.service.huggingface.abstract import HuggingfaceClient


class DiffusersModelAssetInstaller(
    AssetInstaller
):

    spec_type = DiffusersModelAssetSpec

    def __init__(
        self,
        huggingface: HuggingfaceClient,
    ):
        self._huggingface = huggingface

    async def install(
        self,
        spec: DiffusersModelAssetSpec,
    ) -> InstalledAsset:
        directory = await self._huggingface.download_snapshot(
            repository = spec.repository,
            revision = spec.revision,
        )

        return InstalledAsset(
            spec = spec,
            location = AssetLocation(
                uri = f'file://{directory}'
            )
        )