from aibusy_runtime_diffusers.asset.spec.diffusers_checkpoint import DiffusersCheckpointAssetSpec
from aibusy.runtime.resource.spec.abstract import ResourceSpec
from aibusy.runtime.device import Device
from aibusy.runtime.dtype import DType
from dataclasses import dataclass


@dataclass(frozen = True)
class VAEResourceSpec(
    ResourceSpec
):

    asset: DiffusersCheckpointAssetSpec
    device: Device
    dtype: DType