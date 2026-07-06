from aibusy_runtime_diffusers.resource.diffusers import _DiffusersResource


class SchedulerResource(
    _DiffusersResource,
):

    def __init__(
        self,
        *,
        installed_asset,
        scheduler,
    ):
        super().__init__(
            installed_asset = installed_asset,
        )

        self.scheduler = scheduler

    async def load(
        self,
    ):
        scheduler_class = self.scheduler.scheduler_class

        scheduler = scheduler_class.from_pretrained(
            self.subdirectory('scheduler'),
        )

        return scheduler

    async def unload(
        self,
        instance,
    ):
        del instance