from aibusy_runtime_diffusers.runtime.resource.unet import DiffusersUNetResource
from aibusy_runtime_diffusers.resource.spec.unet_resource_spec import UNetResourceSpec
from aibusy.runtime.resource.builder.abstract import ResourceBuilder


class DiffusersUNetResourceBuilder(
    ResourceBuilder
):

    resource_type = UNetResourceSpec

    def create(
        self,
        spec: UNetResourceSpec
    ) -> DiffusersUNetResource:
        return DiffusersUNetResource(spec)