from dataclasses import dataclass

from aibusy.runtime.resource.spec.abstract import ResourceSpec


@dataclass(frozen = True)
class UNetResourceSpec(
    ResourceSpec,
):
    pass