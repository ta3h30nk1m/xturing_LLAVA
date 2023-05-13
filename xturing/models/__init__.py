from .base import BaseModel
from .llama import Llama, LlamaInt8, LlamaLora, LlamaLoraInt8, LlamaLoraInt4

BaseModel.add_to_registry(Llama.config_name, Llama)
BaseModel.add_to_registry(LlamaLora.config_name, LlamaLora)
BaseModel.add_to_registry(LlamaInt8.config_name, LlamaInt8)
BaseModel.add_to_registry(LlamaLoraInt8.config_name, LlamaLoraInt8)
BaseModel.add_to_registry(LlamaLoraInt4.config_name, LlamaLoraInt4)