from aibusy.engine.execution.asset.installed import InstalledAsset
from aibusy.runtime.resource.abstract import Resource
from pathlib import Path
from typing import TypeVar
from abc import ABC


T = TypeVar('T')

class _AssetResource(
    Resource[T],
    ABC,
):
    """
    *For internal use only*

    The resource that includes an `installed_asset`.
    """

    @property
    def model_directory(
        self,
    ) -> Path:
        """
        Root directory of the installed checkpoint.
        """
        return Path(
            self.installed_asset.location.path
        )

    def subdirectory(
        self,
        name: str,
    ) -> Path:
        """
        Returns a subdirectory inside the checkpoint.
        """
        return self.model_directory / name
    
    def __init__(
        self,
        *,
        installed_asset: InstalledAsset
    ):
        self.installed_asset = installed_asset