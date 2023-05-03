from multimodal_xturing.datasets import #InstructionDataset
from multimodal_xturing.models import BaseModel

# Load the dataset
instruction_dataset = #InstructionDataset("./alpaca_data")

# Initialize the model
model = BaseModel.create("llama_lora")

# Finetune the model
model.finetune(dataset=instruction_dataset)

# Perform inference
output = model.generate(texts=["Why LLM models are becoming so important?"])

print("Generated output by the model: {}".format(output))