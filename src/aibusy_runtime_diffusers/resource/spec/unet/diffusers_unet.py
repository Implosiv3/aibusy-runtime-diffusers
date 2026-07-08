from aibusy_runtime_diffusers.resource.spec.unet.abstract import UNetResourceSpec
from aibusy_runtime_diffusers.asset.spec.diffusers_checkpoint import DiffusersCheckpointAssetSpec
from aibusy.runtime.device import Device
from aibusy.runtime.dtype import DType
from dataclasses import dataclass


@dataclass(frozen = True)
class DiffusersUNetResourceSpec(
    UNetResourceSpec,
):

    asset: DiffusersCheckpointAssetSpec
    device: Device
    dtype: DType