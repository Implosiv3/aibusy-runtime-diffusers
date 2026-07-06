import pytest


@pytest.mark.mandatory
@pytest.mark.asyncio
async def test_checkpoint_asset_spec():
    from aibusy_runtime_diffusers.plugin import DiffusersRuntimePlugin
    from aibusy_runtime_diffusers.asset.spec.diffusers_checkpoint import DiffusersCheckpointAssetSpec
    from aibusy.engine.builder import EngineBuilder
    from aibusy.engine.execution.context import ExecutionContext
    from aibusy.engine.cache.memory_cache import MemoryCache

    engine = (
        EngineBuilder()
        .add_plugin(
            DiffusersRuntimePlugin()
        )
        .build()
    )

    execution_context = ExecutionContext(
        engine = engine.context,
        cache = MemoryCache(),
    )

    spec = DiffusersCheckpointAssetSpec(
        repository = 'runwayml/stable-diffusion-v1-5'
    )

    destination_path = spec.get_install_path(
        execution_context.settings.models_directory
    )

    installed = await engine.context.assets.install(
        spec = spec,
        destination_path = destination_path,
    )

    assert installed.location.path == 'C:/Users/dania/.aibusy/models/runwayml/stable-diffusion-v1-5/main'


@pytest.mark.mandatory
@pytest.mark.asyncio
async def test_unet_resource_builder():
    from aibusy_runtime_diffusers.resource.spec.unet import UNetResourceSpec
    from aibusy_runtime_diffusers.resource.builder.unet import UNetResourceBuilder
    from aibusy_runtime_diffusers.asset.spec.diffusers_checkpoint import DiffusersCheckpointAssetSpec
    from aibusy_runtime_diffusers.plugin import DiffusersRuntimePlugin
    from aibusy.engine.execution.context import ExecutionContext
    from aibusy.engine.cache.memory_cache import MemoryCache
    from aibusy.engine.builder import EngineBuilder
    from aibusy.runtime.device import CUDA
    from aibusy.runtime.dtype import FLOAT32

    engine = (
        EngineBuilder()
        .add_plugin(
            DiffusersRuntimePlugin()
        )
        .build()
    )

    spec = UNetResourceSpec(
        asset = DiffusersCheckpointAssetSpec(
            repository = 'runwayml/stable-diffusion-v1-5'
        ),
        device = CUDA,
        dtype = FLOAT32,
    )

    builder = UNetResourceBuilder()

    execution_context = ExecutionContext(
        engine = engine.context,
        cache = MemoryCache(),
    )

    resource = await builder.build(
        spec,
        execution_context,
    )

    assert resource.installed_asset.location.uri == 'file:///C:/Users/dania/.aibusy/models/runwayml/stable-diffusion-v1-5/main'


@pytest.mark.mandatory
@pytest.mark.asyncio
async def test_execution_resource_manager():
    from aibusy_runtime_diffusers.resource.spec.unet import UNetResourceSpec
    from aibusy_runtime_diffusers.resource.builder.unet import UNetResourceBuilder
    from aibusy_runtime_diffusers.asset.spec.diffusers_checkpoint import DiffusersCheckpointAssetSpec
    from aibusy_runtime_diffusers.plugin import DiffusersRuntimePlugin
    from aibusy.engine.execution.context import ExecutionContext
    from aibusy.engine.cache.memory_cache import MemoryCache
    from aibusy.engine.builder import EngineBuilder
    from aibusy.runtime.device import CUDA
    from aibusy.runtime.dtype import FLOAT32
    from diffusers import UNet2DConditionModel

    engine = (
        EngineBuilder()
        .add_plugin(
            DiffusersRuntimePlugin()
        )
        .build()
    )

    spec = UNetResourceSpec(
        asset = DiffusersCheckpointAssetSpec(
            repository = 'runwayml/stable-diffusion-v1-5'
        ),
        device = CUDA,
        dtype = FLOAT32,
    )

    execution_context = ExecutionContext(
        engine = engine.context,
        cache = MemoryCache(),
    )

    resource = await execution_context.resources.resolve(spec)

    assert isinstance(resource, UNet2DConditionModel)