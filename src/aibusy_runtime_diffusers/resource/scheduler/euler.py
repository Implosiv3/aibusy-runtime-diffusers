from aibusy_runtime_diffusers.resource.implementation.euler_scheduler import EulerScheduler
from aibusy_runtime_diffusers.resource.classes.asset_execution_resource import _AssetExecutionResource
from diffusers import EulerDiscreteScheduler



class EulerSchedulerResource(
    _AssetExecutionResource[
        EulerScheduler
    ],
):

    async def load(
        self,
    ) -> EulerScheduler:
        scheduler = EulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path = self.installed_asset.location.path,
            subfolder = 'scheduler',
        )

        return EulerScheduler(
            scheduler = scheduler,
        )