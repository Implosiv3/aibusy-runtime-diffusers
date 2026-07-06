from aibusy_runtime_diffusers.resource.diffusers_model import _DiffusersModelResource
from transformers import CLIPTextModel


class TextEncoderResource(
    _DiffusersModelResource
):

    async def load(
        self,
    ):
        model = CLIPTextModel.from_pretrained(
            self.subdirectory('text_encoder'),
            torch_dtype = self.torch_dtype,
        )

        return self.move_to_device(model)

    async def unload(
        self,
        instance,
    ):
        del instance

        if self.device.type == 'cuda':
            import torch
            torch.cuda.empty_cache()