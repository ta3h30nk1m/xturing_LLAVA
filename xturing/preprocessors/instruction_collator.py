from typing import Optional

import torch
import torch.nn.functional as F
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from xturing.datasets import InstructionDatasetMeta
from PIL import Image
from torchvision import transforms

from typing import Dict, Optional, Sequence

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


image_token_len = 256  ## todo : Hard coded, fix later

# def preprocess_multimodal(
#     sources: Sequence[str],
#     multimodal_cfg: dict,
#     cur_token_len: int,
# ) -> Dict:
#     is_multimodal = multimodal_cfg['is_multimodal']
#     # image_token_len = multimodal_cfg['image_token_len']
#     image_token_len = cur_token_len
#     if not is_multimodal:
#         return sources

#     for source in sources:
#         if multimodal_cfg['sep_image_conv_front']:
#             assert DEFAULT_IMAGE_TOKEN in source[0]['value']
#             source[0]['value'] = source[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
#             source[0]['value'] = DEFAULT_IMAGE_TOKEN + conversation_lib.default_conversation.sep + conversation_lib.default_conversation.roles[0] + ": " + source[0]['value']
#         for sentence in source:
#             replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
#             if multimodal_cfg['use_im_start_end']:
#                 replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
#             sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

#     return sources

# def preprocess(
#     sources: Sequence[str],
#     tokenizer: transformers.PreTrainedTokenizer,
# ) -> Dict:
#     """
#     Given a list of sources, each is a conversation list. This transform:
#     1. Add signal '### ' at the beginning each sentence, with end signal '\n';
#     2. Concatenate conversations together;
#     3. Tokenize the concatenated conversation;
#     4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
#     """
#     if conversation_lib.default_conversation.version == "v1":
#         return preprocess_v1(sources, tokenizer)
#     if conversation_lib.default_conversation.version == "mpt":
#         return preprocess_mpt(sources, tokenizer)
#     # add end signal and concatenate together
#     conversations = []
#     for source in sources:
#         header = f"{conversation_lib.default_conversation.system}\n\n"
#         conversation = _add_speaker_and_signal(header, source)
#         conversations.append(conversation)
#     # tokenize conversations
#     conversations_tokenized = _tokenize_fn(conversations, tokenizer)
#     input_ids = conversations_tokenized["input_ids"]
#     targets = copy.deepcopy(input_ids)
#     for target, source in zip(targets, sources):
#         tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
#                                       tokenizer)["input_ids_lens"]
#         speakers = [sentence["from"] for sentence in source]
#         _mask_targets(target, tokenized_lens, speakers)

#     return dict(input_ids=input_ids, labels=targets)

class InstructionDataCollator:
    config_name = "instruction_dataset"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: Optional[int] = None,
        meta: InstructionDatasetMeta = InstructionDatasetMeta(),
        transformer: transforms = None,
    ):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.meta = meta

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
        flatten_samples = []
        label_masks = []

        for sample in batches:
            system_msg = sample["text"]
            system_msg = system_msg + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
            input_text = self.tokenizer(system_msg)                         # system msg
            input_img = self.transformer(Image.open(sample["image"]).convert('RGB'))   
            input_target = self.tokenizer(sample["target"])                     
            
            
            #####  Now meta.list_prompt_template is None ###  Not Enter
            if self.meta.list_prompt_template is not None:
                print("entered if 1")
                combine = self.meta.list_prompt_template.build(
                    instruction=sample["instruction"], text=sample["text"]
                )
                print(combine)
                input_combine = self.tokenizer(combine)
                input_ids = input_combine["input_ids"] + input_target["input_ids"]
                label_mask = [False] * len(input_combine["input_ids"]) + [True] * len(
                    input_target["input_ids"]
                )
            ##########################################################
            
            ##### Now meta.infix_instruction is None , Enter Here #####
            elif not self.meta.infix_instruction:
                #print("entered elif 2")
                input_instruction = self.tokenizer(sample["instruction"])
#                 input_ids = (
#                     input_instruction["input_ids"]
#                     + input_text["input_ids"]
#                     + input_target["input_ids"]
#                 )
                # fix here. Instruction dataset Variable Name is different between LLaVA and xturing
                input_ids = (
                    input_text["input_ids"]
                    + input_instruction["input_ids"]
                    + input_target["input_ids"]
                )
                label_mask = (
                    [False] * len(input_instruction["input_ids"])
                    + [False] * len(input_text["input_ids"])
                    + [True] * len(input_target["input_ids"])
                )
            ############################################################
                
            else:  #####  Not Enter
                print("entered else 3")
                parts = self._process_instruction(sample["instruction"])

                input_instructions = [self.tokenizer(part) for part in parts]

                assert (
                    len(input_instructions) == 3
                ), "There should be exactly three parts in the instruction."

                input_ids = (
                    input_instructions[0]["input_ids"]
                    + input_text["input_ids"]
                    + input_instructions[1]["input_ids"]
                    + input_target["input_ids"]
                    + input_instructions[2]["input_ids"]
                )

                label_mask = (
                    [False] * len(input_instructions[0]["input_ids"])
                    + [False] * len(input_text["input_ids"])
                    + [False] * len(input_instructions[1]["input_ids"])
                    + [True] * len(input_target["input_ids"])
                    + [False] * len(input_instructions[2]["input_ids"])
                )
            ################# end

            input_ids = input_ids[: self.max_length - 1]
            input_ids.append(self.tokenizer.eos_token_id)
            attention_mask = [1] * len(input_ids)

            label_mask = label_mask[: self.max_length - 1]
            label_mask = label_mask + [True]

            flatten_samples.append(
                {
                    "images": input_img,
                    "input_ids": torch.tensor(input_ids).long(),
                    "attention_mask": torch.tensor(attention_mask).long(),
                }
            )
            label_masks.append(label_mask)

        batch = self.tokenizer.pad(
            flatten_samples,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        dim = batch["input_ids"].shape[-1]

        batch["label_masks"] = torch.stack(
            [
                F.pad(torch.tensor(x), (0, dim - len(x)), value=False)
                for x in label_masks
            ]
        )
        batch["targets"] = torch.roll(batch["input_ids"], -1, -1)

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
