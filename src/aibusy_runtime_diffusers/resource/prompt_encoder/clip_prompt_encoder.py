from aibusy_runtime_diffusers.resource.classes.asset_execution_resource import _AssetExecutionResource
from aibusy_runtime_diffusers.resource.implementation.clip_prompt_encoder import CLIPPromptEncoder
from transformers import CLIPTextModel, CLIPTokenizer


class CLIPPromptEncoderResource(
    _AssetExecutionResource[
        CLIPPromptEncoder
    ],
):

    # def __init__(
    #     self,
    #     installed_asset,
    #     tokenizer,
    #     text_encoder,
    # ):
    #     super().__init__(
    #         installed_asset = installed_asset,
    #     )

    #     self._tokenizer = tokenizer
    #     self._text_encoder = text_encoder

    # @property
    # def tokenizer(
    #     self,
    # ):
    #     return self._tokenizer

    # @property
    # def text_encoder(
    #     self,
    # ):
    #     return self._text_encoder
    
    async def load(
        self,
    ) -> CLIPPromptEncoder:
        self._tokenizer = CLIPTokenizer.from_pretrained(
            self.subdirectory('tokenizer'),
            torch_dtype = self.torch_dtype,
        )

        self._text_encoder = CLIPTextModel.from_pretrained(
            self.subdirectory('text_encoder'),
            torch_dtype = self.torch_dtype,
        )
        self.move_to_device(self._text_encoder)

        return CLIPPromptEncoder(
            tokenizer = self._tokenizer,
            text_encoder = self._text_encoder
        )

    async def unload(
        self,
        instance: CLIPPromptEncoder,
    ):
        del instance

        if self.device.type == 'cuda':
            import torch
            torch.cuda.empty_cache()




# class CLIPPromptEncoderResource(
#     _DiffusersModelResource
# ):

    # async def load(
    #     self,
    # ) -> CLIPTextModel:
    #     model = CLIPTextModel.from_pretrained(
    #         self.subdirectory('text_encoder'),
    #         torch_dtype = self.torch_dtype,
    #     )

    #     return self.move_to_device(model)

    # async def unload(
    #     self,
    #     instance,
    # ):
    #     del instance

    #     if self.device.type == 'cuda':
    #         import torch
    #         torch.cuda.empty_cache()