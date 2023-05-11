# xturing_LLAVA
## Visual Language Model (VLM) with Quantized LLM
##### This project was carried out in Yonsei AI([YAI](https://github.com/yonsei-YAI)) as YAICON

## Background
  LLM(Large Language Model) has been popular recently and it has showen that it can generate text very well and help us in real life a lot. And now there are several researches that tries to combine visual encoder and LLM. There are models like [Flamingo](https://github.com/lucidrains/flamingo-pytorch), [CLIP](https://github.com/openai/CLIP), [Kosmos](https://github.com/microsoft/unilm#llm--mllm-multimodal-llm), [BLIP](https://github.com/salesforce/BLIP), and [LLaVA](https://github.com/haotian-liu/LLaVA). However, these models are too large that hard to train or inference on small hardware. Thus, we aim to minimize gpu usage for VLM model. We especially try to implement [LLaVA](https://github.com/haotian-liu/LLaVA) because of its simple model architecture and high performance. 
  We also found there are various research for quantizing LLM. So we applied those techniques to reduce LLM and connect it with visual encoder so that the total model is runnable in single gpu. We tested that this is runnable even in colab.

## Reference
  Here is the list of researches/codes that we used to implement this code:
  - [LLaVA](https://github.com/haotian-liu/LLaVA)
  - [transformers:huggingface codes](https://github.com/huggingface/transformers)
  - [GPTQ: Large Language Model Quantization](https://github.com/IST-DASLab/gptq)
  - [LoRA: efficient Finetuning method](https://github.com/microsoft/LoRA)
  - [xturing: library of efficient LLM](https://github.com/stochasticai/xturing) <-- we clone this and modify a bit.

## Pretrained weight
TBD

## How to start
```
git clone https://github.com/ta3h30nk1m/xturing_LLAVA.git
pip install -r ./xturing_LLAVA/requirements.txt
```

## Datasets
We used the same datasets as [LLaVA](https://github.com/haotian-liu/LLaVA) which are CC3M and LLAVA_instruct datasets.
- [CC-3M](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K)
- [LLaVA-Instruct](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)

## Training
```
python simple_train.py --dataset ./dataset_folder --weights_path ./checkpoint --first_stage True --output --./output_path
```
- --dataset: dataset folder path
- --weights_path: (optional) if pretrained weights exist. Otherwise, default LLaMA checkpoint is called
- --first_stage: True for only train projector layer that connect visual encoder and LLM. False for train both projector and LLM
- --output: specify checkpoint output path 

  in order to change hyperparameters, go to ./xturing_LLAVA/config/finetuning_config.yaml and change config of 'llama_lora_int4'

## Generation
```
python simple_generate.py --weights_path ./checkpoint --image_file ./image.png --text "input text to the model"
```
- --weights_path: model checkpoint path
- --image_file: input image file
- --text: input text

## Chat??
???
