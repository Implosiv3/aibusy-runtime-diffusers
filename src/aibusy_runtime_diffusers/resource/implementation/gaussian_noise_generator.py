from aibusy_runtime_diffusers.utils.torch_dtype import to_torch_dtype
from aibusy.runtime.device import Device
from aibusy.runtime.dtype import DType
import torch


class GaussianNoiseGenerator:
    """
    The class that knows how to perform the
    gaussian noise generation.
    """

    def __init__(
        self,
        device: Device,
        dtype: DType
    ):
        self.device = device
        self.dtype = dtype

    def generate(
        self,
        seed: int,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        generator = torch.Generator(
            device = str(self.device)
        )

        generator.manual_seed(seed)

        return torch.randn(
            (
                batch_size,
                channels,
                height,
                width,
            ),
            generator = generator,
            device = str(self.device),
            dtype = to_torch_dtype(self.dtype),
        )