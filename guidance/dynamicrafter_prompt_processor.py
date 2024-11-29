from dataclasses import dataclass

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessor
from threestudio.utils.typing import *


@threestudio.register("dynamicrafter-prompt-processor")
class DynamiCrafterPromptProcessor(PromptProcessor):
    # Empty prompt processor class

    @dataclass
    class Config(PromptProcessor.Config):
        pass

    cfg: Config

    ### these functions are unused, kept for debugging ###
    def configure_text_encoder(self) -> None:
        pass

    def destroy_text_encoder(self) -> None:
        pass

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ):
        return None, None

    ###

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir):
        pass
