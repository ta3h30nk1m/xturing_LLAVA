import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import transformers
from torch import nn
from transformers import AutoTokenizer

from xturing.engines.causal import CausalEngine, CausalLoraEngine
from xturing.engines.llama_utils import LlamaConfig, LlamaTokenizer, LlamaForCausalLM
from xturing.engines.lora_engine import prepare_model_for_int8_training
from xturing.engines.quant_utils import autotune_warmup, make_quant
from xturing.utils.hub import ModelHub


class LLamaEngine(CausalEngine):
    config_name: str = "llama_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "aleksickx/llama-7b-hf"
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = LlamaTokenizer.from_pretrained(model_name, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(weights_path=weights_path, model=model, tokenizer=tokenizer)

    def save(self, saving_path: Union[str, Path]):
        self.model.save_pretrained(saving_path)
        self.tokenizer.save_pretrained(saving_path)


class LlamaLoraEngine(CausalLoraEngine):
    config_name: str = "llama_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "aleksickx/llama-7b-hf"
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        tokenizer = LlamaTokenizer.from_pretrained(model_name, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(
            weights_path=weights_path,
            model=model,
            tokenizer=tokenizer,
            target_modules=["q_proj", "v_proj"],
        )


class LLamaInt8Engine(CausalEngine):
    config_name: str = "llama_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "aleksickx/llama-7b-hf"
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map=device_map,
        )
        model = prepare_model_for_int8_training(model)
        tokenizer = LlamaTokenizer.from_pretrained(model_name, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(
            weights_path=weights_path, model=model, tokenizer=tokenizer, load_8bit=True
        )

    def save(self, saving_path: Union[str, Path]):
        self.model.save_pretrained(saving_path)
        self.tokenizer.save_pretrained(saving_path)


class LlamaLoraInt8Engine(CausalLoraEngine):
    config_name: str = "llama_lora_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "aleksickx/llama-7b-hf"
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map=device_map,
        )
        model = prepare_model_for_int8_training(model)

        tokenizer = LlamaTokenizer.from_pretrained(model_name, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(
            weights_path=weights_path,
            model=model,
            tokenizer=tokenizer,
            load_8bit=True,
            target_modules=["q_proj", "v_proj"],
        )


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res

#################
# here to use Vicuna
#################
class LlamaLoraInt4Engine(CausalLoraEngine):
    config_name: str = "llama_lora_int4_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "decapoda-research/llama-7b-hf"

        if weights_path is None:
            weights_path = ModelHub().load("x/llama_lora_int4")
        
        print("LLamaConfig_from_pretrained start")
        config = LlamaConfig.from_pretrained(model_name)
        print("LLamaConfig_from_pretrained end")
        
        saved_kaiming_uniform_ = torch.nn.init.kaiming_uniform_
        saved_uniform_ = torch.nn.init.uniform_
        saved_normal_ = torch.nn.init.normal_

        def noop(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = noop
        torch.nn.init.uniform_ = noop
        torch.nn.init.normal_ = noop
           
        ########
        ## change to load from huggingface directly
        #######
        torch.set_default_dtype(torch.half)
        transformers.modeling_utils._init_weights = False
        #torch.set_default_dtype(torch.half)
        model = LlamaForCausalLM(config)
        #model = Llava(config)#LlamaForCausalLM(config)#.from_pretrained("Aitrepreneur/vicuna-7B-1.1-GPTQ-4bit-128g") ### changed 
        torch.set_default_dtype(torch.float)
        model = model.eval()
        layers = find_layers(model)
        #print(layers)
        key_to_del = []
        for name in ["lm_head", "visual_model", "mm_projector"]:
            for key in layers.keys():
                if name in key:
                    key_to_del.append(key)
        for key in key_to_del:
            del layers[key]
            # if name in layers:
            #     del layers[name]

        wbits = 4
        groupsize = 128
        warmup_autotune = True

        make_quant(model, layers, wbits, groupsize)

        state_dict = torch.load(
            weights_path / Path("pytorch_model.bin"), map_location="cpu"
        )
        # import requests
        # from io import BytesIO
        # url = "https://huggingface.co/Aitrepreneur/vicuna-7B-1.1-GPTQ-4bit-128g/resolve/main/vicuna-7B-1.1-GPTQ-4bit-128g.no-act-order.pt"
        # # output_path = "./vicuna-7B-1.1-GPTQ-4bit-128g.no-act-order.pt"
        # response = requests.get(url)
        # in_mem_file = BytesIO(response.content)
        # in_mem_file.seek(0)
        
        # print("torch.load start")
        # state_dict = torch.load(in_mem_file, map_location='cpu')
        # print("torch.load end")

        new_state_dict = {}
        for key, value in state_dict.items():
            new_state_dict[key[6:]] = value
        model.load_state_dict(new_state_dict, strict=False)

        if warmup_autotune:
            autotune_warmup(model)

        model.seqlen = 2048

        model.gptq = True

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        # only training mm_projector
        model.requires_grad_(False)
        for p in model.mm_projector.parameters():
            p.requires_grad = True

        tokenizer = LlamaTokenizer.from_pretrained(model_name, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        model.initialize_vision_tokenizer(True, tokenizer, device='cpu')

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
        )

        torch.nn.init.kaiming_uniform_ = saved_kaiming_uniform_
        torch.nn.init.uniform_ = saved_uniform_
        torch.nn.init.normal_ = saved_normal_
