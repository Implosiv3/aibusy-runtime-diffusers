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
async def test_resource_builder():
    from aibusy_runtime_diffusers.resource.spec.unet import UNetResourceSpec
    from aibusy_runtime_diffusers.resource.spec.vae import VAEResourceSpec
    from aibusy_runtime_diffusers.resource.spec.text_encoder import TextEncoderResourceSpec
    from aibusy_runtime_diffusers.resource.spec.tokenizer import TokenizerResourceSpec
    from aibusy_runtime_diffusers.resource.spec.scheduler import SchedulerResourceSpec
    from aibusy_runtime_diffusers.resource.builder.unet import UNetResourceBuilder
    from aibusy_runtime_diffusers.resource.builder.vae import VAEResourceBuilder
    from aibusy_runtime_diffusers.resource.builder.text_encoder import TextEncoderResourceBuilder
    from aibusy_runtime_diffusers.resource.builder.tokenizer import TokenizerResourceBuilder
    from aibusy_runtime_diffusers.resource.builder.scheduler import SchedulerResourceBuilder
    from aibusy_runtime_diffusers.asset.spec.diffusers_checkpoint import DiffusersCheckpointAssetSpec
    from aibusy_runtime_diffusers.scheduler.euler import EulerScheduler
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

    execution_context = ExecutionContext(
        engine = engine.context,
        cache = MemoryCache(),
    )

    # Test UNet
    unet_spec = UNetResourceSpec(
        asset = DiffusersCheckpointAssetSpec(
            repository = 'runwayml/stable-diffusion-v1-5'
        ),
        device = CUDA,
        dtype = FLOAT32,
    )

    builder = UNetResourceBuilder()

    resource = await builder.build(
        unet_spec,
        execution_context,
    )

    assert resource.installed_asset.location.uri == 'file:///C:/Users/dania/.aibusy/models/runwayml/stable-diffusion-v1-5/main'

    # Test VAE
    vae_spec = VAEResourceSpec(
        asset = DiffusersCheckpointAssetSpec(
            repository = 'runwayml/stable-diffusion-v1-5'
        ),
        device = CUDA,
        dtype = FLOAT32,
    )

    builder = VAEResourceBuilder()

    resource = await builder.build(
        vae_spec,
        execution_context,
    )

    assert resource.installed_asset.location.uri == 'file:///C:/Users/dania/.aibusy/models/runwayml/stable-diffusion-v1-5/main'

    # Test TextEncoder
    text_encoder_spec = TextEncoderResourceSpec(
        asset = DiffusersCheckpointAssetSpec(
            repository = 'runwayml/stable-diffusion-v1-5'
        ),
        device = CUDA,
        dtype = FLOAT32,
    )

    builder = TextEncoderResourceBuilder()

    resource = await builder.build(
        text_encoder_spec,
        execution_context,
    )

    assert resource.installed_asset.location.uri == 'file:///C:/Users/dania/.aibusy/models/runwayml/stable-diffusion-v1-5/main'

    # Test Tokenizer
    tokenizer_spec = TokenizerResourceSpec(
        asset = DiffusersCheckpointAssetSpec(
            repository = 'runwayml/stable-diffusion-v1-5'
        )
    )

    builder = TokenizerResourceBuilder()

    resource = await builder.build(
        tokenizer_spec,
        execution_context,
    )

    assert resource.installed_asset.location.uri == 'file:///C:/Users/dania/.aibusy/models/runwayml/stable-diffusion-v1-5/main'

    # Test Scheduler
    scheduler_spec = SchedulerResourceSpec(
        asset = DiffusersCheckpointAssetSpec(
            repository = 'runwayml/stable-diffusion-v1-5'
        ),
        scheduler = EulerScheduler
    )

    builder = SchedulerResourceBuilder()

    resource = await builder.build(
        scheduler_spec,
        execution_context,
    )

    assert resource.installed_asset.location.uri == 'file:///C:/Users/dania/.aibusy/models/runwayml/stable-diffusion-v1-5/main'


@pytest.mark.mandatory
@pytest.mark.asyncio
async def test_execution_resource_manager():
    from aibusy_runtime_diffusers.resource.spec.unet import UNetResourceSpec
    from aibusy_runtime_diffusers.resource.spec.vae import VAEResourceSpec
    from aibusy_runtime_diffusers.resource.spec.text_encoder import TextEncoderResourceSpec
    from aibusy_runtime_diffusers.resource.spec.tokenizer import TokenizerResourceSpec
    from aibusy_runtime_diffusers.resource.spec.scheduler import SchedulerResourceSpec
    from aibusy_runtime_diffusers.scheduler.euler import EulerScheduler
    from aibusy_runtime_diffusers.asset.spec.diffusers_checkpoint import DiffusersCheckpointAssetSpec
    from aibusy_runtime_diffusers.plugin import DiffusersRuntimePlugin
    from aibusy.engine.execution.context import ExecutionContext
    from aibusy.engine.cache.memory_cache import MemoryCache
    from aibusy.engine.builder import EngineBuilder
    from aibusy.runtime.device import CUDA
    from aibusy.runtime.dtype import FLOAT32
    from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
    from transformers import CLIPTextModel, CLIPTokenizer

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

    # Test UNet
    unet_spec = UNetResourceSpec(
        asset = DiffusersCheckpointAssetSpec(
            repository = 'runwayml/stable-diffusion-v1-5'
        ),
        device = CUDA,
        dtype = FLOAT32,
    )

    resource = await execution_context.resources.resolve(unet_spec)

    assert isinstance(resource, UNet2DConditionModel)

    # Test VAE
    vae_spec = VAEResourceSpec(
        asset = DiffusersCheckpointAssetSpec(
            repository = 'runwayml/stable-diffusion-v1-5'
        ),
        device = CUDA,
        dtype = FLOAT32,
    )

    resource = await execution_context.resources.resolve(vae_spec)

    assert isinstance(resource, AutoencoderKL)

    # Test TextEncoder
    text_encoder_spec = TextEncoderResourceSpec(
        asset = DiffusersCheckpointAssetSpec(
            repository = 'runwayml/stable-diffusion-v1-5'
        ),
        device = CUDA,
        dtype = FLOAT32,
    )

    resource = await execution_context.resources.resolve(text_encoder_spec)

    assert isinstance(resource, CLIPTextModel)

    # Test Tokenizer
    tokenizer_spec = TokenizerResourceSpec(
        asset = DiffusersCheckpointAssetSpec(
            repository = 'runwayml/stable-diffusion-v1-5'
        )
    )

    resource = await execution_context.resources.resolve(tokenizer_spec)

    assert isinstance(resource, CLIPTokenizer)

    # Test Scheduler
    scheduler_spec = SchedulerResourceSpec(
        asset = DiffusersCheckpointAssetSpec(
            repository = 'runwayml/stable-diffusion-v1-5'
        ),
        scheduler = EulerScheduler
    )

    resource = await execution_context.resources.resolve(scheduler_spec)

    assert isinstance(resource, EulerDiscreteScheduler)