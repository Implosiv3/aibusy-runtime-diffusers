from aibusy.runtime.resource.spec.abstract import ResourceSpec
from aibusy.engine.execution.asset.spec.abstract import AssetSpec
from aibusy.runtime.device import Device
from aibusy.runtime.torch_dtype import TorchDType
from dataclasses import dataclass


@dataclass(frozen = True)
class VAEResourceSpec(
    ResourceSpec
):

    asset: AssetSpec
    device: Device
    dtype: TorchDType