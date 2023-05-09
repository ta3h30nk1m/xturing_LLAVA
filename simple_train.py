import argparse

import time
from math import floor

import os
import zipfile

from xturing.datasets import InstructionDataset
from xturing.models import BaseModel

def main(args):
    # Load the dataset
    dataset = args.dataset
   

    """
    ###############################################
    # Todo : if /images folder not exist, unzip images.zip
    ############################################### this is only done in kiml
    output_folder = "/app/output"
    output_img_folder = "/app/output/images"


    if not os.path.isdir(f"{output_folder}/images") :
        print("No images folder. make it in app/output/images")
        st_time = time.time()
        
        # Path to the ZIP file
        zip_path = os.path.join(args.dataset, "images.zip")
        extract_path = os.path.join(args.output + "images")  

        # Open the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract all the files to the specified folder
            zip_ref.extractall(extract_path)

        print(f"unzip Images.zip into {extract_path} completed.")    
        output_img_folder = extract_path
        print(f"{floor((time.time() - st_time)/60)} min {round((time.time() - st_time) % 60,3)} secs")
    else :
        print("Already unziped Images.zip")
        
    img_folder = output_img_folder                          # in kiml
    """
    img_folder = os.path.join(args.dataset, "images")       # in colab or local
    
    instruction_dataset = InstructionDataset(dataset, img_folder)
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
