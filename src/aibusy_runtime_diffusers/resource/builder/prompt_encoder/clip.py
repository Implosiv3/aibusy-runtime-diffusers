from aibusy_runtime_diffusers.resource.prompt_encoder.clip_prompt_encoder import CLIPPromptEncoderResource
from aibusy_runtime_diffusers.resource.spec.prompt_encoder.clip import CLIPPromptEncoderResourceSpec
from aibusy.runtime.resource.builder.abstract import ResourceBuilder
from aibusy.engine.execution.context import ExecutionContext


class CLIPPromptEncoderResourceBuilder(
    ResourceBuilder,
):

    @property
    def spec_type(
        self,
    ) -> type[CLIPPromptEncoderResourceSpec]:
        return CLIPPromptEncoderResourceSpec

    async def build(
        self,
        resource_spec: CLIPPromptEncoderResourceSpec,
        context: ExecutionContext,
    ) -> CLIPPromptEncoderResource:
        destination_path = resource_spec.asset.get_install_path(
            context.settings.models_directory
        )

        installed_asset = await context.assets.install(
            spec = resource_spec.asset,
            destination_path = destination_path
        )

        return CLIPPromptEncoderResource(
            installed_asset = installed_asset,
            device = resource_spec.device,
            dtype = resource_spec.dtype,
        )