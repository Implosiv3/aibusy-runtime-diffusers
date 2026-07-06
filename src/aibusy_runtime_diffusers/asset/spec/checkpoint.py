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
        models_root: Path,
    ) -> Path:
        """
        Get the path where the diffusers checkpoint files
        must be downloaded.
        """
        organization, model_name = self.repository.split('/', 1)
        revision = (
            self.revision or
            'main'
        )

        return (
            models_root
            / organization
            / model_name
            / revision
        )