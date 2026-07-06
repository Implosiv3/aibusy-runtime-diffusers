from aibusy_runtime_diffusers.resource.text_encoder import TextEncoderResource
from aibusy_runtime_diffusers.resource.spec.text_encoder import TextEncoderResourceSpec
from aibusy.runtime.resource.builder.abstract import ResourceBuilder
from aibusy.engine.execution.context import ExecutionContext


class TextEncoderResourceBuilder(
    ResourceBuilder,
):

    @property
    def spec_type(
        self,
    ) -> type[TextEncoderResourceSpec]:
        return TextEncoderResourceSpec

    async def build(
        self,
        spec: TextEncoderResourceSpec,
        context: ExecutionContext,
    ) -> TextEncoderResource:
        destination_path = spec.asset.get_install_path(
            context.settings.models_directory
        )

        installed_asset = await context.assets.install(
            spec = spec.asset,
            destination_path = destination_path
        )

        return TextEncoderResource(
            installed_asset = installed_asset,
            device = spec.device,
            dtype = spec.dtype,
        )