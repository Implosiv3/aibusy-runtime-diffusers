from aibusy.runtime.resource.spec.abstract import ResourceSpec
from dataclasses import dataclass


@dataclass(frozen = True)
class SchedulerResourceSpec(
    ResourceSpec,
):
    pass