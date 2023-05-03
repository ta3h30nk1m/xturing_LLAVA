from xturing.datasets import TextDataset
from xturing.models import BaseModel

# Load the dataset
instruction_dataset = TextDataset("./mmc4")

# Initialize the model
model = BaseModel.create("llama_lora_int4")

# Finetune the model
model.finetune(dataset=instruction_dataset)

# Perform inference
output = model.generate(texts=["Why LLM models are becoming so important?"])

print("Generated output by the model: {}".format(output))