# Modified from https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa.py

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import json
from tqdm import tqdm

from xturing.models import BaseModel
from xturing.preprocessors.conversation import conv_templates
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria

from PIL import Image
import random
import math

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


def test(args):
    # Model
    disable_torch_init()
    if args.weights_path == "":
        weights_path = None
        mm_projector_path = None
    else:
        weights_path = args.weights_path
        mm_projector_path = os.path.join(weights_path, "mm_projector.bin")
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
    model = BaseModel.create("llama_lora_int4", weights_path=weights_path, pretrain_mm_mlp_adapter = mm_projector_path)
    mm_use_im_start_end = getattr(model.engine.model.config, "mm_use_im_start_end", False)
    vision_config = model.engine.model.visual_model.config
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # answers_file = os.path.expanduser(args.answers_file)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")
    # for i, line in enumerate(tqdm(questions)):
        #idx = line["question_id"]
    image_file = args.image_file#line["image"]
    qs = args.text#line["text"]
    cur_prompt = qs
    if mm_use_im_start_end:
        qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
    else:
        qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

    if args.conv_mode == 'simple_legacy':
        qs += '\n\n### Response:'
    # conv = default_conversation.copy()
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    prompt = conv.get_prompt()
    inputs = model.engine.tokenizer([prompt])
    if image_file == "":
        image_tensor = None
    else:
        image = Image.open(image_file)
    # image.save(os.path.join(save_image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    # new stopping implementation
    class KeywordsStoppingCriteria(StoppingCriteria):
        def __init__(self, keywords, tokenizer, input_ids):
            self.keywords = keywords
            self.tokenizer = tokenizer
            self.start_len = None
            self.input_ids = input_ids

        def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            if self.start_len is None:
                self.start_len = self.input_ids.shape[1]
            else:
                outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
                for keyword in self.keywords:
                    if keyword in outputs:
                        return True
            return False

    keywords = ['###']
    stopping_criteria = KeywordsStoppingCriteria(keywords, model.engine.tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.engine.model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] Sample: {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = model.engine.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

    if args.conv_mode == 'simple_legacy' or args.conv_mode == 'simple':
        while True:
            cur_len = len(outputs)
            outputs = outputs.strip()
            for pattern in ['###', 'Assistant:', 'Response:']:
                if outputs.startswith(pattern):
                    outputs = outputs[len(pattern):].strip()
            if len(outputs) == cur_len:
                break

    try:
        index = outputs.index(conv.sep)
    except ValueError:
        outputs += conv.sep
        index = outputs.index(conv.sep)

    outputs = outputs[:index].strip()

    print("prompt: ", cur_prompt)
    print("text: ", outputs)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default="")
    parser.add_argument("--image_file", type=str, default="")
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="simple")
    args = parser.parse_args()

    test(args)