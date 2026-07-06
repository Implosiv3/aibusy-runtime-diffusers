from aibusy_runtime_diffusers.resource.diffusers import _DiffusersResource
from transformers import CLIPTokenizer


class TokenizerResource(
    _DiffusersResource
):

    async def load(
        self,
    ):
        return CLIPTokenizer.from_pretrained(
            self.subdirectory('tokenizer')
        )

    async def unload(
        self,
        instance,
    ):
        del instance