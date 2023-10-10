
import torch 



import random 


import random
from typing import Callable

import torch



from transformers import IdeficsImageProcessor
from transformers import IdeficsForVisionText2Text, AutoProcessor

from PIL import Image
import re 

class hf_image_processor_wrapper(IdeficsImageProcessor):
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, x):
        if isinstance(x, Image.Image):
            return self.processor(x).squeeze()
        else:
            return self.processor(x)
        


def create_model_and_transforms(
    lang_encoder_path: str,
    tokenizer_path: str,
    use_local_files: bool = True,
    low_cpu: bool = False,
):

    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k")
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        use_local_files (bool, optional): whether to use local files. Defaults to False.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    




    if low_cpu:
        print("load in bfloat16")
        try:
            lang_encoder = IdeficsForVisionText2Text.from_pretrained(
                lang_encoder_path, local_files_only=use_local_files, revision="bfloat16", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
            )
        except:
            lang_encoder = IdeficsForVisionText2Text.from_pretrained(
                lang_encoder_path, local_files_only=use_local_files, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
            )
    else:
        lang_encoder = IdeficsForVisionText2Text.from_pretrained(
            lang_encoder_path, local_files_only=use_local_files, torch_dtype=torch.bfloat16
        )


    processor = AutoProcessor.from_pretrained(tokenizer_path, local_files_only=use_local_files)



    text_tokenizer = processor.tokenizer


    model = lang_encoder

    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0


    print(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    return model, processor, text_tokenizer


split_words = ["Short answer:", "answer:", "Good Answer:", "Good:", "Bad Answer:", "Bad:", "Better Answer:"]
split_words += ["Caption:", "Good Caption:", "Bad Caption:", ":"]
split_words += ["Explanation:", "because", "Bad Explanation:", "Good Explanation:"]
# Combine the words into a regular expression pattern using '|'
split_pattern = "(" + "|".join(re.escape(word) for word in split_words) + ")" # "() to include the splitting word"

def split_user_assistant(txt, split_pattern=split_pattern, instruct_model=False, train=True):
    # instruct_model=False 
    txt = txt.replace("<image>", "").replace("<|endofchunk|>", "\n")
    if  instruct_model:
        txt = txt.replace("\n", "") 
        c_split = re.split(split_pattern, txt, flags=re.IGNORECASE, maxsplit=1)
        u, a = c_split[0].strip().replace(":", ''), "".join(c_split[1:]).strip().replace(":", '')
        user = "User: "+u+"<end_of_utterance>\n"
        if not train:
            assistant = "Assistant: " + a
        else:
            assistant = "Assistant: "+ a +"<end_of_utterance>\n"
        return [user, assistant] 
    else:
        return [txt]

def get_context(
    get_prompt: Callable[[dict], str],
    in_context_samples,
    effective_num_shots,
    num_shots,
    neg_samples=None,
    mode=None,
    instruction='',
    instruct_model=False
) -> str:

    if num_shots == 1:
        effective_num_shots == 0
        
    if mode is None:
        context_text = (
            [get_prompt(s) for s in in_context_samples]
            if effective_num_shots > 0
            else ""
        )
    elif 'contrastive' in mode:
        text = []
        for s in in_context_samples:
            if effective_num_shots > 0:
                ns = random.sample(neg_samples, 1)[0]
                text.append(get_prompt(s, ns))
            else:
                text.append("")

        context_text = text
    else:
        context_text = (
            [get_prompt(s) for s in in_context_samples]
            if effective_num_shots > 0
            else ""
        )
       
    if num_shots == 0 or (mode is not None and "blind" in mode):
        context_text = [c.replace("<image>", "") for c in context_text]

    context_images = [
            s["image"] for s in in_context_samples
        ]
    
    context = []

    instruct_model = False # force disable for now 
    if instruct_model:
        context_text = [c.replace("<image>", "").replace("<|endofchunk|>", "\n") for c in context_text]

        if num_shots > 0:
            for txt, img in zip(context_text, context_images):
                user, assistant = split_user_assistant(txt, split_pattern, instruct_model=instruct_model)
                context += [img, user, assistant]
        else:
            for txt in context_text:
                user, assistant = split_user_assistant(txt, split_pattern, instruct_model=instruct_model)
                context += [user, assistant]  
    else:
        context_text = [c.replace("<image>", "").replace("<|endofchunk|>", "\n") for c in context_text]

        if num_shots > 0:
            
            for img, txt in zip(context_images, context_text):
                context+=[img, txt]
        else:
            context = context_text
    
    
    if num_shots == 1:
        context = []
        
    if instruction:
        context.insert(0, instruction+"\n")



    return context

def get_context_images(image_processor, in_context_samples, num_shots):
    if num_shots > 0:
        context_images = [
            image_processor(s["image"]).unsqueeze(0) for s in in_context_samples
        ]
        context_images = torch.cat(context_images, dim=0)
        context_images = context_images.unsqueeze(1).unsqueeze(0)
    else:
        context_images = None
    return context_images


def get_outputs(
    model,
    inputs,
    device,
    max_generation_length,
    num_beams,
    length_penalty,
    do_sample=False,
):
    
    
    input_ids = inputs['input_ids']
    max_generation_length += input_ids.shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            do_sample=do_sample,
        )

    outputs = outputs[:, len(input_ids[0]) :]
    return outputs