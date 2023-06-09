import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from datasets import Dataset
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk

from xturing.datasets.base import BaseDataset
from xturing.model_apis import TextGenerationAPI
from xturing.self_instruct import (
    bootstrap_instructions,
    generate_instances,
    identify_if_classification,
    prepare_for_finetuning,
    prepare_seed_tasks,
)
from xturing.utils.utils import create_temp_directory, extract_text_from_directory

class ListPromptTemplate:
    def __init__(
        self,
        template: str,
        input_variables: List[str],
    ):
        self.template = template
        self.input_variables = input_variables

    def build(self, **kwargs) -> str:
        for i in self.input_variables:
            if i not in kwargs:
                raise ValueError(f"Missing input variable {i}")

        return self.template.format(**kwargs)


@dataclass
class InstructionDatasetMeta:
    infix_instruction: bool = False
    list_prompt_template: Optional[ListPromptTemplate] = None

class InstructionDataset(BaseDataset):
    config_name: str = "instruction_dataset"

    def __init__(
        self,
        path: Union[str, Path, HFDataset, dict],   # kakao i cloud #path will be "app/input/dataset/llava-cc3m-595k"   /app/input/dataset/{데이터세트 이름}
        output_img_folder : Union[str,Path],        # unziped CC3M
        infix_instruction: bool = False,
        promt_template: str = None,
    ):
        
        ################################ not use
        if isinstance(path, HFDataset) or isinstance(path, DatasetDict):
            self.data = path
        elif isinstance(path, dict): 
            self.data = {"train": HFDataset.from_dict(path)}
        ################################
        
        else:    # <- since path is str, excuted from here
            path = Path(path)
            assert Path(path).exists(), "path does not exist"
            
                
            #make data (load data to memory)
            self.data = []
            NumsOfImgsNotExist = 0

            with open(os.path.join(path, "chat.json"), "r") as f:
                chat_data = json.load(f)
                """
                [
                  {
                    "id": "GCC_train_002582585",
                    "image": "GCC_train_002582585.jpg",
                    "conversations": [
                      {
                        "from": "human",
                        "value": "Provide a brief description of the given image.\n<image>"
                      },
                      {
                        "from": "gpt",
                        "value": "olive oil is a healthy ingredient used liberally ."
                      }
                    ]
                  },
                    
                  ...
                ]
                """
            try:
                #count = 0
                for line in chat_data:
                    #text_ = 'You are GPT0, a large language and vision assistant.\nYou are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\nFollow the instructions carefully and explain your answers in detail.\n'
                    #image_ = os.path.join(path, "images" , line['image'])  
                    image_ = os.path.join(output_img_folder, line['image'])
                    instruction_ = line['conversations'][0]['value'] #'USER: ' + line['conversations'][0]['value'] + '\ASSISTANT: '
                    target_ = line['conversations'][1]['value']
                    
                    if os.path.isfile(image_) :
                        self.data.append({ 'image': image_, 'instruction': instruction_, 'target': target_})
                    else :
                        #print(f"{image_} is in chat.json but not in images folder.")
                        NumsOfImgsNotExist += 1
                    #count+=1
                    #if(count>100): break
            except KeyError:
                raise ValueError(
                    "The json file should have keys text, instruction and target"
                )
            print(f"imgs detected : {len(self.data)},  imgs not exists : {NumsOfImgsNotExist}")
 
        #self._validate()

        list_prompt_template = None

        if promt_template is not None:
            list_prompt_template = ListPromptTemplate(
                promt_template, input_variables=["text", "instruction"]
            )

        self._meta = InstructionDatasetMeta(
            infix_instruction=infix_instruction,
            list_prompt_template=list_prompt_template,
        )

    # def _validate(self):
    #     # check is hf dataset has train split and if it has column text, and if there are any other - it should be target
    #     assert "train" in self.data, "The dataset should have a train split"
    #     assert (
    #         "text" in self.data["train"].column_names
    #     ), "The dataset should have a column named text"
    #     assert (
    #         "target" in self.data["train"].column_names
    #     ), "The dataset should have a column named target"
    #     assert (
    #         "instruction" in self.data["train"].column_names
    #     ), "The dataset should have a column named instruction"
    #     assert (
    #         "image" in self.data["train"].column_names
    #     ), "The dataset should have a column named image"
    #     assert (
    #         len(self.data["train"].column_names) == 4
    #     ), "The dataset should have only three columns, instruction, text and target and image"

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx):
        return self.data[idx] 
    
    #def save(self, path):
    #    return self.data.save_to_disk(path)
