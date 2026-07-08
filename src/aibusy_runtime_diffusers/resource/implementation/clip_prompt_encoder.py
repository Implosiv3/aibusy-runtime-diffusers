from transformers import CLIPTokenizer
from transformers import CLIPTextModel
from dataclasses import dataclass



@dataclass(frozen = True)
class CLIPPromptEncoder:

    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel