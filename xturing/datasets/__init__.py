from .base import BaseDataset
#from .visual_instruction_dataset import InstructionDataset, InstructionDatasetMeta
from .text2image_dataset import Text2ImageDataset
from .text_dataset import TextDataset, TextDatasetMeta

# from .visual_instruction_dataset import 

BaseDataset.add_to_registry(TextDataset.config_name, TextDataset)
#BaseDataset.add_to_registry(InstructionDataset.config_name, InstructionDataset)
BaseDataset.add_to_registry(Text2ImageDataset.config_name, Text2ImageDataset)
