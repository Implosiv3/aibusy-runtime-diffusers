from aibusy_runtime_diffusers.resource.tokenizer import TokenizerResource
from aibusy_runtime_diffusers.resource.spec.tokenizer import TokenizerResourceSpec
from aibusy.runtime.resource.builder.abstract import ResourceBuilder
from aibusy.engine.execution.context import ExecutionContext


class TokenizerResourceBuilder(
    ResourceBuilder,
):

    @property
    def spec_type(
        self,
    ) -> type[TokenizerResourceSpec]:
        return TokenizerResourceSpec

    async def build(
        self,
        spec: TokenizerResourceSpec,
        context: ExecutionContext,
    ) -> TokenizerResource:
        destination_path = spec.asset.get_install_path(
            context.settings.models_directory
        )

        installed_asset = await context.assets.install(
            spec = spec.asset,
            destination_path = destination_path
        )

        return TokenizerResource(
            installed_asset = installed_asset
        )