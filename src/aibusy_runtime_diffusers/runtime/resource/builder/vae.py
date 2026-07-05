from aibusy_runtime_diffusers.runtime.resource.vae import DiffusersVAEResource
from aibusy_runtime_diffusers.resource.spec.vae_resource_spec import VAEResourceSpec
from aibusy.runtime.resource.builder.abstract import ResourceBuilder


class DiffusersVAEResourceBuilder(
    ResourceBuilder
):

    spec_type = VAEResourceSpec

    def create(
        self,
        spec: VAEResourceSpec,
    ) -> DiffusersVAEResource:
        return DiffusersVAEResource(spec)