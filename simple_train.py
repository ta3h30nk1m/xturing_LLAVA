import argparse

from xturing.datasets import TextDataset
from xturing.models import BaseModel

def main(args):
    # Load the dataset
    dataset = args.dataset
    instruction_dataset = TextDataset(dataset)
    print("datanum: " len(instruction_dataset))

    # Initialize the model
    model = BaseModel.create("llama_lora_int4")

    # Finetune the model
    model.finetune(dataset=instruction_dataset)

    # Perform inference
    output = model.generate(texts=["Why LLM models are becoming so important?"])

    print("Generated output by the model: {}".format(output))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="./mmc4/")
    args = parser.parse_args()
    main(args)
