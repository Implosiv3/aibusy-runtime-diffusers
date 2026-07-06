from aibusy_runtime_diffusers.asset.spec.diffusers_checkpoint import DiffusersCheckpointAssetSpec
from aibusy.runtime.resource.spec.abstract import ResourceSpec
from dataclasses import dataclass



@dataclass(frozen = True)
class TokenizerResourceSpec(
    ResourceSpec,
):
    asset: DiffusersCheckpointAssetSpec