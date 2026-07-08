from aibusy_runtime_diffusers.resource.spec.scheduler.abstract import SchedulerResourceSpec
from aibusy_runtime_diffusers.asset.spec.diffusers_checkpoint import DiffusersCheckpointAssetSpec
from aibusy.runtime.device import Device
from aibusy.runtime.dtype import DType
from dataclasses import dataclass


@dataclass(frozen = True)
class EulerSchedulerResourceSpec(
    SchedulerResourceSpec,
):
    
    asset: DiffusersCheckpointAssetSpec
    device: Device
    dtype: DType