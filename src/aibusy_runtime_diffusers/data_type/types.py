from aibusy.graph.data_type.abstract import DataType
# from aibusy_runtime_diffusers.resource.spec.tokenizer import TokenizerResourceSpec
from aibusy_runtime_diffusers.resource.prompt_encoder.clip_prompt_encoder import CLIPPromptEncoderResource
from aibusy_runtime_diffusers.resource.spec.prompt_encoder.clip import CLIPPromptEncoderResourceSpec


# TOKENIZER_RESOURCE_SPEC = DataType(
#     name = 'TokenizerResourceSpec',
#     runtime_type = TokenizerResourceSpec,
#     parent = None
# )

CLIP_TEXT_ENCODER_RESOURCE_SPEC = DataType(
    name = 'CLIPPromptEncoderResourceSpec',
    runtime_type = CLIPPromptEncoderResourceSpec,
    parent = None

)
CLIP_TEXT_ENCODER_RESOURCE = DataType(
    name = 'CLIPPromptEncoderResource',
    runtime_type = CLIPPromptEncoderResource,
    parent = None
)