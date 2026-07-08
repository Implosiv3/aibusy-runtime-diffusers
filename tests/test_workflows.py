import pytest


@pytest.mark.mandatory
@pytest.mark.asyncio
async def test_workflow():
    from aibusy_runtime_diffusers.plugin import DiffusersRuntimePlugin
    from aibusy_runtime_diffusers.asset.spec.diffusers_checkpoint import DiffusersCheckpointAssetSpec
    from aibusy.engine.builder import EngineBuilder
    from aibusy.engine.execution.context import ExecutionContext
    from aibusy.engine.cache.memory_cache import MemoryCache
    from aibusy_runtime_diffusers.operation.clip_prompt_encode import CLIPPromptEncode
    from aibusy_runtime_diffusers.operation.generate_gaussian_noise import GenerateGaussianNoise
    from aibusy_runtime_diffusers.resource.spec.prompt_encoder.clip import CLIPPromptEncoderResourceSpec
    from aibusy_runtime_diffusers.resource.spec.noise_generator.gaussian import GaussianNoiseGeneratorResourceSpec
    from aibusy.runtime.resource.reference import ResourceReference
    from aibusy.runtime.device import CUDA
    from aibusy.runtime.dtype import FLOAT32
    from tests.common import PromptEmbeddingsShape
    import torch
    

    engine = (
        EngineBuilder()
        .add_plugin(
            DiffusersRuntimePlugin()
        )
        .build()
    )

    checkpoint_asset_spec = DiffusersCheckpointAssetSpec(
        repository = 'runwayml/stable-diffusion-v1-5'
    )

    clip_prompt_encode_operation = CLIPPromptEncode(
        prompt_encoder = ResourceReference(
            resource_spec = CLIPPromptEncoderResourceSpec(
                asset = checkpoint_asset_spec,
                device = CUDA,
                dtype = FLOAT32,
            )
        ),
        prompt = 'This is an example of a prompt'
    )

    shape_operation = PromptEmbeddingsShape(
        prompt_embeddings = clip_prompt_encode_operation.prompt_embeddings
    )

    result = await engine.run(
        shape_operation.shape
    )

    assert isinstance(result, torch.Size)
    assert result == (1, 77, 768)

    gaussian_noise_operation = GenerateGaussianNoise(
        gaussian_noise_generator = ResourceReference(
            resource_spec = GaussianNoiseGeneratorResourceSpec(
                device = CUDA,
                dtype = FLOAT32,
            )
        ),
        seed = 0,
        width = 512,
        height = 512,
        batch_size = 1,
        channels = 4
    )

    result = await engine.run(
        gaussian_noise_operation.noise
    )

    assert isinstance(result.value, torch.Tensor)

    return


@pytest.mark.mandatory
@pytest.mark.asyncio
async def test_euler_sampler():
    from aibusy_runtime_diffusers.plugin import DiffusersRuntimePlugin
    from aibusy_runtime_diffusers.asset.spec.diffusers_checkpoint import DiffusersCheckpointAssetSpec
    from aibusy_runtime_diffusers.operation.clip_prompt_encode import CLIPPromptEncode
    from aibusy_runtime_diffusers.operation.generate_gaussian_noise import GenerateGaussianNoise
    from aibusy_runtime_diffusers.operation.euler_sampler import EulerSampler
    from aibusy_runtime_diffusers.resource.spec.prompt_encoder.clip import CLIPPromptEncoderResourceSpec
    from aibusy_runtime_diffusers.resource.spec.noise_generator.gaussian import GaussianNoiseGeneratorResourceSpec
    from aibusy_runtime_diffusers.resource.spec.scheduler.euler import EulerSchedulerResourceSpec
    from aibusy_runtime_diffusers.resource.spec.unet.diffusers_unet import DiffusersUNetResourceSpec
    from aibusy.engine.builder import EngineBuilder
    from aibusy.runtime.device import CUDA
    from aibusy.runtime.dtype import FLOAT32
    from aibusy.runtime.resource.reference import ResourceReference
    from aibusy.utils.classes.latents import Latents

    engine = (
        EngineBuilder()
        .add_plugin(
            DiffusersRuntimePlugin()
        )
        .build()
    )


    checkpoint_asset_spec = DiffusersCheckpointAssetSpec(
        repository = 'runwayml/stable-diffusion-v1-5'
    )

    prompt_encoder = ResourceReference(
        CLIPPromptEncoderResourceSpec(
            asset = checkpoint_asset_spec,
            device = CUDA,
            dtype = FLOAT32,
        )
    )

    scheduler = ResourceReference(
        EulerSchedulerResourceSpec(
            asset = checkpoint_asset_spec,
            device = CUDA,
            dtype = FLOAT32,
        )
    )

    unet = ResourceReference(
        DiffusersUNetResourceSpec(
            asset = checkpoint_asset_spec,
            device = CUDA,
            dtype = FLOAT32,
        )
    )

    noise_generator = ResourceReference(
        GaussianNoiseGeneratorResourceSpec(
            device = CUDA,
            dtype = FLOAT32,
        )
    )

    positive = CLIPPromptEncode(
        prompt_encoder = prompt_encoder,
        prompt = 'A beautiful landscape',
    )

    negative = CLIPPromptEncode(
        prompt_encoder = prompt_encoder,
        prompt = '',
    )

    noise = GenerateGaussianNoise(
        gaussian_noise_generator = noise_generator,
        seed = 42,
        batch_size = 1,
        channels = 4,
        width = 64,
        height = 64,
    )

    sampler = EulerSampler(
        scheduler = scheduler,
        unet = unet,
        noise = noise.noise,
        positive_prompt_embeddings = positive.prompt_embeddings,
        negative_prompt_embeddings = negative.prompt_embeddings,
        steps = 20,
        cfg_scale = 7.5,
    )

    result = await engine.run(sampler.latents)

    assert isinstance(result, Latents)
    assert result.value.shape == (1, 4, 64, 64,)

