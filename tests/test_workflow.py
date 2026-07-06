import pytest


@pytest.mark.mandatory
@pytest.mark.asyncio
async def test_workflow():
    from aibusy_runtime_diffusers.plugin import DiffusersRuntimePlugin
    from aibusy_runtime_diffusers.asset.spec.checkpoint import DiffusersCheckpointAssetSpec
    from aibusy.engine.builder import EngineBuilder

    engine = (
        EngineBuilder()
        .add_plugin(
            DiffusersRuntimePlugin()
        )
        .build()
    )

    installed = await engine.context.assets.install(
        DiffusersCheckpointAssetSpec(
            repository = 'runwayml/stable-diffusion-v1-5'
        )
    )

    assert installed.location.path == '/C:/Users/dania/.aibusy/models/runwayml/stable-diffusion-v1-5/main'