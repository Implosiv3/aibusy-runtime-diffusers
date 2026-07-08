from diffusers import EulerDiscreteScheduler


class EulerScheduler:

    def __init__(
        self,
        scheduler: EulerDiscreteScheduler,
    ):
        self._scheduler = scheduler

    @property
    def scheduler(
        self,
    ) -> EulerDiscreteScheduler:
        return self._scheduler