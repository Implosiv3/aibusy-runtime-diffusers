from aibusy.runtime.resource.abstract import Resource
from aibusy.engine.execution.asset.installed import InstalledAsset
from abc import ABC
from pathlib import Path


class _DiffusersResource(
    Resource,
    ABC,
):
    """
    *For internal use only*

    Class to include the basicp art of a diffusers
    resource, like the asset and some utils.
    """
    
    def __init__(
        self,
        *,
        installed_asset: InstalledAsset,
    ):
        self.installed_asset = installed_asset

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