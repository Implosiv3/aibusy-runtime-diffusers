from aibusy.runtime.resource.spec.abstract import ResourceSpec
from aibusy.runtime.device import Device
from aibusy.runtime.dtype import DType
from dataclasses import dataclass


@dataclass(frozen = True)
class GaussianNoiseGeneratorResourceSpec(
    ResourceSpec
):
    
    device: Device
    dtype: DType