from aibusy.graph.operation.abstract.atomic_operation import AtomicOperation
from aibusy.graph.classes.input import Input
from aibusy.graph.classes.output import Output
from aibusy.engine.execution.context import ExecutionContext
from aibusy.utils.classes.prompt_embeddings import PromptEmbeddings
from aibusy.graph.data_type.types import CLIP_PROMPT_ENCODER_RESOURCE_REFERENCE, PROMPT, PROMPT_EMBEDDINGS


class CLIPPromptEncode(
    AtomicOperation
):
    """
    *Atomic Operation*

    Encode the `prompt` given using a CLIP
    `encoder` and `tokenizer`.

    Inputs:
    - `prompt_encoder` (`CLIP_PROMPT_ENCODER_RESOURCE_REFERENCE`)
    - `prompt` (`STRING`)

    Outputs:
    - `prompt_embeddings` (`PROMPT_EMBEDDINGS`)
    """

    prompt_encoder = Input(CLIP_PROMPT_ENCODER_RESOURCE_REFERENCE)
    prompt = Input(PROMPT)

    prompt_embeddings = Output(PROMPT_EMBEDDINGS)

    async def execute(
        self,
        context: ExecutionContext,
    ):
        resource = await context.resources.resolve(self.prompt_encoder.resource_spec)

        tokenized = resource.tokenizer(
            self.prompt,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt',
        )

        outputs = resource.text_encoder(
            tokenized.input_ids.to(
                resource.text_encoder.device
            )
        )

        prompt_embeddings = PromptEmbeddings(
            value = outputs.last_hidden_state,
        )

        return {
            'prompt_embeddings': prompt_embeddings,
        }