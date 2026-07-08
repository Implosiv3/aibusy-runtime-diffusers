from aibusy.graph.operation.abstract.atomic_operation import AtomicOperation
from aibusy.graph.classes.input import Input
from aibusy.graph.classes.output import Output
from aibusy.engine.execution.context import ExecutionContext
from aibusy.utils.classes.prompt_embeddings import PromptEmbeddings
from aibusy.graph.data_type.types import CLIP_PROMPT_ENCODER_RESOURCE_REFERENCE, PROMPT, PROMPT_EMBEDDINGS


from aibusy.graph.data_type.types import DataType

SHAPE = DataType(
    'SHAPE',
    tuple,
)


class PromptEmbeddingsShape(
    AtomicOperation,
):
    
    prompt_embeddings = Input(PROMPT_EMBEDDINGS)

    shape = Output(SHAPE)

    async def execute(
        self,
        context: ExecutionContext,
    ):
        return {
            'shape': self.prompt_embeddings.value.shape,
        }