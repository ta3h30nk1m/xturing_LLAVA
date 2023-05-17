from typing import Optional

import torch
import torch.nn.functional as F
import transformers
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from xturing.datasets import Llava_InstructionDatasetMeta
from PIL import Image
from torchvision import transforms

from typing import Dict, Optional, Sequence

import copy
import xturing.preprocessors.conversation as conv_lib

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

image_token_len = 256  ## todo : Hard coded, fix later

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conv_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conv_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

class Llava_InstructionDataCollator:
    config_name = "llava_instruction_dataset"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: Optional[int] = None,
        meta: Llava_InstructionDatasetMeta = Llava_InstructionDatasetMeta(),
        transformer: transforms = None,
    ):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.meta = meta
        
        print("Llava_InstructionDataCollator __init__ :")
        print(f"\t max_length : {self.max_length}")
        print(f"\t tokenizer : {self.tokenizer}")
        print(f"\t transformer : {self.transformer}")
        print(f"\t meta : {self.meta}")
        print("InstructionDataCollator __init__  finished")

    def _process_instruction(self, instruction, tags=None):
        # check if the instruction is valid
        # split the instruction into parts
        # check how many {text}/{target} parts are in the instruction
        if tags is None:
            tags = ["{text}", "{target}"]

        for tag in tags:
            assert (
                instruction.count(tag) == 1
            ), f"There should be exactly one {tag} in the instruction."

        parts = []

        for tag in tags:
            left, right = instruction.split(tag)
            parts.append(left)
            instruction = right

        parts.append(instruction)

        return parts

    def __call__(self, batches):
        texts = []
        images = []
        header = f"{conv_lib.default_conversation.system}\n\n"

        for sample in batches:
            input_img = self.transformer(Image.open(sample["image"]).convert('RGB'))   
            images.append(input_img.to(torch.float16))
            source = sample["conversations"]
            conversation = _add_speaker_and_signal(header, source)
            texts.append(conversation)
        # tokenize conversations
        conversations_tokenized = _tokenize_fn(texts, self.tokenizer)
        input_ids = conversations_tokenized["input_ids"]
        targets = copy.deepcopy(input_ids)
        for target, sample in zip(targets, batches):
            source = sample["conversations"]
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                        self.tokenizer)["input_ids_lens"]
            speakers = [sentence["from"] for sentence in source]
            _mask_targets(target, tokenized_lens, speakers)
            
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(
                targets,
                batch_first=True,
                padding_value=IGNORE_INDEX)
            
            batch = dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                images=torch.stack(images)
            )

        return batch
    """ # from LLaVa dataset __get_item__
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.multimodal_cfg['image_folder']
            processor = self.multimodal_cfg['image_processor']
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.multimodal_cfg['image_aspect_ratio'] == 'keep':
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 448, 224
                shortest_edge = int(min(max_len / aspect_ratio, min_len))
                image = processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge})['pixel_values'][0]
            elif self.multimodal_cfg['image_aspect_ratio'] == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            cur_token_len = (image.shape[1]//14) * (image.shape[2]//14)   # FIXME: 14 is hardcoded patch size
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.multimodal_cfg, cur_token_len)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.multimodal_cfg['is_multimodal']:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.multimodal_cfg['image_processor'].crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict
"""
# from LLaVa dataset __get_item__
