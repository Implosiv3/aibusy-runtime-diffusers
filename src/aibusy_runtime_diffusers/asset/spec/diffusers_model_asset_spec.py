from aibusy.engine.execution.asset.spec.abstract import AssetSpec
from dataclasses import dataclass
from typing import Union


@dataclass(frozen = True)
class DiffusersModelAssetSpec(
    AssetSpec
):

    repository: str
    revision: Union[str, None] = None
    variant: Union[str, None] = None