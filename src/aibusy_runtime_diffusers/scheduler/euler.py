from aibusy.runtime.scheduler.abstract import Scheduler
from diffusers import EulerDiscreteScheduler


class EulerScheduler(
    Scheduler,
):

    scheduler_class = EulerDiscreteScheduler