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
    from aibusy_runtime_diffusers.resource.spec.prompt_encoder.clip import CLIPPromptEncoderResourceSpec
    from aibusy_runtime_diffusers.resource.spec.unet.diffusers_unet import DiffusersUNetResourceSpec
    from aibusy_runtime_diffusers.resource.builder.prompt_encoder.clip import CLIPPromptEncoderResourceBuilder
    from aibusy_runtime_diffusers.resource.builder.unet.diffusers_unet import DiffusersUNetResourceBuilder
    # from aibusy_runtime_diffusers.resource.builder.scheduler import SchedulerResourceBuilder
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

    # Test TextEncoder
    prompt_encoder_spec = CLIPPromptEncoderResourceSpec(
        asset = DiffusersCheckpointAssetSpec(
            repository = 'runwayml/stable-diffusion-v1-5'
        ),
        device = CUDA,
        dtype = FLOAT32,
    )

    builder = CLIPPromptEncoderResourceBuilder()

    resource = await builder.build(
        prompt_encoder_spec,
        execution_context,
    )

    assert resource.installed_asset.location.uri == 'file:///C:/Users/dania/.aibusy/models/runwayml/stable-diffusion-v1-5/main'

    # Test DiffusersUNet
    diffusers_unet_spec = DiffusersUNetResourceSpec(
        asset = DiffusersCheckpointAssetSpec(
            repository = 'runwayml/stable-diffusion-v1-5'
        ),
        device = CUDA,
        dtype = FLOAT32,
    )

    builder = DiffusersUNetResourceBuilder()

    resource = await builder.build(
        diffusers_unet_spec,
        execution_context,
    )

    assert resource.installed_asset.location.uri == 'file:///C:/Users/dania/.aibusy/models/runwayml/stable-diffusion-v1-5/main'

    # # Test Scheduler
    # scheduler_spec = SchedulerResourceSpec(
    #     asset = DiffusersCheckpointAssetSpec(
    #         repository = 'runwayml/stable-diffusion-v1-5'
    #     ),
    #     scheduler = EulerScheduler
    # )

    # builder = SchedulerResourceBuilder()

    # resource = await builder.build(
    #     scheduler_spec,
    #     execution_context,
    # )

    # assert resource.installed_asset.location.uri == 'file:///C:/Users/dania/.aibusy/models/runwayml/stable-diffusion-v1-5/main'


@pytest.mark.mandatory
@pytest.mark.asyncio
async def test_execution_resource_manager():
    # TODO: Remove VAE
    from aibusy_runtime_diffusers.resource.spec.prompt_encoder.clip import CLIPPromptEncoderResourceSpec
    from aibusy_runtime_diffusers.scheduler.euler import EulerScheduler
    from aibusy_runtime_diffusers.asset.spec.diffusers_checkpoint import DiffusersCheckpointAssetSpec
    from aibusy_runtime_diffusers.resource.spec.scheduler.euler import EulerSchedulerResourceSpec
    from aibusy_runtime_diffusers.resource.spec.unet.diffusers_unet import DiffusersUNetResourceSpec
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

    # Test PromptEncoder
    prompt_encoder_spec = CLIPPromptEncoderResourceSpec(
        asset = DiffusersCheckpointAssetSpec(
            repository = 'runwayml/stable-diffusion-v1-5'
        ),
        device = CUDA,
        dtype = FLOAT32,
    )

    resource = await execution_context.resources.resolve(prompt_encoder_spec)

    assert isinstance(resource.tokenizer, CLIPTokenizer)
    assert isinstance(resource.text_encoder, CLIPTextModel)

    # # Test Scheduler
    # scheduler_spec = SchedulerResourceSpec(
    #     asset = DiffusersCheckpointAssetSpec(
    #         repository = 'runwayml/stable-diffusion-v1-5'
    #     ),
    #     scheduler = EulerScheduler
    # )

    # resource = await execution_context.resources.resolve(scheduler_spec)

    # assert isinstance(resource, EulerDiscreteScheduler)

    # Test Euler Scheduler
    euler_scheduler_spec = EulerSchedulerResourceSpec(
        asset = DiffusersCheckpointAssetSpec(
            repository = 'runwayml/stable-diffusion-v1-5'
        ),
        device = CUDA,
        dtype = FLOAT32,
    )

    resource = await execution_context.resources.resolve(euler_scheduler_spec)

    assert isinstance(resource.scheduler, EulerDiscreteScheduler)

    # Test Diffusers UNet
    diffusers_unet_resource_spec = DiffusersUNetResourceSpec(
        asset = DiffusersCheckpointAssetSpec(
            repository = 'runwayml/stable-diffusion-v1-5'
        ),
        device = CUDA,
        dtype = FLOAT32,
    )

    resource = await execution_context.resources.resolve(diffusers_unet_resource_spec)

    assert isinstance(resource.unet, UNet2DConditionModel)