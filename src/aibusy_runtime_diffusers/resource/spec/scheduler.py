from aibusy_runtime_diffusers.asset.spec.diffusers_checkpoint import DiffusersCheckpointAssetSpec
from aibusy.runtime.resource.spec.abstract import ResourceSpec
from aibusy.runtime.scheduler.abstract import Scheduler
from dataclasses import dataclass


@dataclass(frozen = True)
class SchedulerResourceSpec(
    ResourceSpec,
):

    asset: DiffusersCheckpointAssetSpec
    scheduler: type[Scheduler]
    """
    Our custom class that includes the `diffusers`
    class that must be used to instantiate the real
    scheduler. The Resource will use the internal
    `scheduler_class` that this custom class is
    including.
    """