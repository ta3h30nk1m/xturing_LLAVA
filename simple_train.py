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

    # Initialize the model
    print("init model")
    if args.weights_path == "":
        weights_path = None
        mm_projector_path = None
    else:
        weights_path = args.weights_path
        mm_projector_path = os.path.join(weights_path, "mm_projector.bin")
    
    lr = None
    bs = None
    epochs = None
    if args.bs != -1:
        bs = args.bs
    if args.lr != -1:
        lr = args.lr
    if args.epochs != -1:
        epochs = args.epochs
    model = BaseModel.create("llama_lora_int4", weights_path=weights_path, pretrain_mm_mlp_adapter=mm_projector_path, first_stage=args.first_stage,
                             epochs=epochs, batch_size=bs, learning_rate=lr)

    print("init dataset")
    img_folder = os.path.join(args.dataset, "images")       # in colab or local
    
    instruction_dataset = InstructionDataset(dataset, img_folder)
    print("datanum: ", len(instruction_dataset))
    # Finetune the model
    if args.weights_path != "":
        # load hyperparameters back from checkpoint
        pass
    model.finetune(dataset=instruction_dataset, output_dir = args.output)
    print("done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="")
    parser.add_argument('--weights_path', default="")
    parser.add_argument('--first_stage', type=bool, default=True)
    parser.add_argument('--output', default="/app/output/")
    parser.add_argument('--bs', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=-1)
    args = parser.parse_args()
    main(args)
