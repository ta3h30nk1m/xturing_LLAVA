from xturing.preprocessors.instruction_collator import InstructionDataCollator
from xturing.preprocessors.text_collator import TextDataCollator
from xturing.preprocessors.llava_instruct_collator import Llava_InstructionDataCollator
from xturing.registry import BaseParent

class BasePreprocessor(BaseParent):
    registry = {}


BasePreprocessor.add_to_registry(
    InstructionDataCollator.config_name, InstructionDataCollator
)
BasePreprocessor.add_to_registry(TextDataCollator.config_name, TextDataCollator)

BasePreprocessor.add_to_registry(Llava_InstructionDataCollator.config_name, Llava_InstructionDataCollator)