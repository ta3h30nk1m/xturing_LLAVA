import argparse

import os
import zipfile

from xturing.datasets import InstructionDataset
from xturing.models import BaseModel

def main(args):
    # Load the dataset
    dataset = args.dataset
   
    ###############################################
    # Todo : if /images folder not exist, unzip images.zip
    ###############################################
    output_folder = "/app/output"
    output_img_folder = "/app/output/images"
    
    if not os.path.isdir(f"{output_folder}/images") :
        print("No images folder. make it in app/output/images")
        
        # Path to the ZIP file
        zip_path = os.path.join(args.dataset, "images.zip")
        extract_path = os.path.join(args.output + "images")  

        # Open the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract all the files to the specified folder
            zip_ref.extractall(extract_path)

        print(f"unzip Images.zip into {extract_path} completed.")    
        output_img_folder = extract_path
    else :
        print("already unziped Images.zip")
    
    instruction_dataset = InstructionDataset(dataset, output_img_folder)
    print("datanum: ", len(instruction_dataset))

    # Initialize the model
    model = BaseModel.create("llama_lora_int4")

    # Finetune the model
    model.finetune(dataset=instruction_dataset, output_dir = args.output)

    # Perform inference
    output = model.generate(texts=["Why LLM models are becoming so important?"])

    print("Generated output by the model: {}".format(output))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="./mmc4/")
    parser.add_argument('--output', default="/app/output/")
    args = parser.parse_args()
    main(args)
