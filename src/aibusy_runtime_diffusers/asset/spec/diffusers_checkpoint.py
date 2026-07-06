from aibusy.engine.execution.asset.spec.abstract import AssetSpec
from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen = True)
class DiffusersCheckpointAssetSpec(
    AssetSpec
):

    repository: str
    revision: Union[str, None] = None
    variant: Union[str, None] = None

    @property
    def allow_patterns(
        self,
    ):
        return [
            'model_index.json',

            'scheduler/**',

            'tokenizer/**',
            'tokenizer_2/**',

            'text_encoder/**',
            'text_encoder_2/**',

            'unet/**',

            'vae/**',

            '*.json',
            '*.txt',
            '*.model',
            '*.safetensors',
        ]
    
    @property
    def ignore_patterns(
        self,
    ):
        return [
            '*.bin',
            '*non_ema*',
        ]

    @property
    def organization(
        self
    ):
        return self.repository.split('/', 1)[0]
    
    @property
    def model_name(
        self
    ):
        return self.repository.split('/', 1)[1]
    
    def get_install_path(
        self,
        root_path: Path,
    ) -> Path:
        """
        Get the path where the diffusers checkpoint files
        must be downloaded.
        """
        revision = (
            self.revision or
            'main'
        )

        return (
            Path(root_path)
            / self.organization
            / self.model_name
            / revision
        )