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

    print(":",model.engine.model.model.model.mm_projector.weight.dtype)
    model.finetune(dataset=instruction_dataset, output_dir = args.output)
    print("done")

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="")
    parser.add_argument('--weights_path', default="")
    parser.add_argument('--first_stage', type=str2bool, default=True)
    parser.add_argument('--output', default="/app/output/")
    parser.add_argument('--bs', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=-1)
    args = parser.parse_args()
    main(args)
"""
For readers :
    1. model = BaseModel.create("llama_lora_int4", weights_path=weights_path, pretrain_mm_mlp_adapter=mm_projector_path, first_stage=args.first_stage,
                             epochs=epochs, batch_size=bs, learning_rate=lr)
                class BaseModel(BaseParent):
                        BaseModel.add_to_registry(LlamaLoraInt4.config_name, LlamaLoraInt4)
                class LlamaLoraInt4(CausalLoraInt8Model):
                    super().__init__(LlamaLoraInt4Engine.config_name, weights_path, first_stage=first_stage, pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter,
                                         epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
                class CausalLoraInt8Model(CausalLoraModel):
                        assert_not_cpu_int8()
                        super().__init__(engine, weights_path, first_stage=first_stage, pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter,
                                         epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
                class CausalLoraModel(CausalModel):
                        super().__init__(engine, weights_path, first_stage=first_stage, pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter,
                                         epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
                class CausalModel(BaseModel):
                        self.engine = BaseEngine.create(engine, weights_path, first_stage, pretrain_mm_mlp_adapter)
                class BaseEngine(BaseParent):
                        registry = {}
                        quant_utiles/__init__.py
                        BaseEngine.add_to_registry(LlamaLoraInt4Engine.config_name, LlamaLoraInt4Engine)
                class LlamaLoraInt4Engine(CausalLoraEngine):
                        config_name: str = "llama_lora_int4_engine"
                        def __init__(self, weights_path: Optional[Union[str, Path]] = None, first_stage: bool = True, pretrain_mm_mlp_adapter:str = None):
                                     model_name = "decapoda-research/llama-7b-hf"
                Start read from here (LlamaLoraInt4Engine)
    2. model.finetune(dataset=instruction_dataset, output_dir = args.output)
"""
###
