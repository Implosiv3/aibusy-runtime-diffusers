from aibusy.runtime.resource.spec.abstract import ResourceSpec
from aibusy.engine.execution.asset.installed import InstalledAsset
from aibusy.runtime.device import Device
from aibusy.runtime.torch_dtype import TorchDType
from dataclasses import dataclass


@dataclass(frozen = True)
class VAEResourceSpec(
    ResourceSpec
):

    asset: InstalledAsset
    device: Device
    dtype: TorchDType