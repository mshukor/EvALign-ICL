
import torch 



import random 


import random
from typing import Callable

import torch



from transformers import IdeficsImageProcessor
from transformers import IdeficsForVisionText2Text, AutoProcessor

from PIL import Image
import re 


DEVICE_MAP ={
    "80B_5GPUs": {
                'model.embed_tokens': 0, 
                'model.vision_model': 0, 
                'model.perceiver_resampler': 0, 
                'model.layers.0': 0, 
                'model.layers.1': 0, 
                'model.layers.2': 0, 
                'model.layers.3': 0, 
                'model.layers.4': 0, 
                'model.layers.5': 0, 
                'model.layers.6': 0, 
                'model.layers.7': 0, 
                'model.layers.8': 0, 
                'model.layers.9': 0, 
                'model.layers.10': 0, 
                'model.layers.11': 0, 
                'model.layers.12': 0, 
                'model.layers.13': 0, 
                'model.layers.14': 0, 
                'model.layers.15': 0, 
                'model.layers.16': 0, 
                'model.layers.17': 0, 
                'model.layers.18': 0, 
                'model.layers.19': 0, 
                'model.layers.20.self_attn.q_proj': 0, 
                'model.layers.20.self_attn.k_proj': 1, 
                'model.layers.20.self_attn.v_proj': 1, 
                'model.layers.20.self_attn.o_proj': 1, 
                'model.layers.20.self_attn.rotary_emb': 1, 
                'model.layers.20.mlp': 1, 
                'model.layers.20.input_layernorm': 1, 
                'model.layers.20.post_attention_layernorm': 1, 
                'model.layers.21': 1, 
                'model.layers.22': 1, 
                'model.layers.23': 1, 
                'model.layers.24': 1, 
                'model.layers.25': 1, 
                'model.layers.26': 1, 
                'model.layers.27': 1, 
                'model.layers.28': 1, 
                'model.layers.29': 1, 
                'model.layers.30': 1, 
                'model.layers.31': 1, 
                'model.layers.32': 1, 
                'model.layers.33': 1, 
                'model.layers.34': 1, 
                'model.layers.35': 1, 
                'model.layers.36': 1, 
                'model.layers.37': 1, 
                'model.layers.38': 1, 
                'model.layers.39': 1, 
                'model.layers.40': 1, 
                'model.layers.41.self_attn': 1, 
                'model.layers.41.mlp.gate_proj': 1, 
                'model.layers.41.mlp.down_proj': 2, 
                'model.layers.41.mlp.up_proj': 2, 
                'model.layers.41.mlp.act_fn': 2, 
                'model.layers.41.input_layernorm': 2, 
                'model.layers.41.post_attention_layernorm': 2, 
                'model.layers.42': 2, 
                'model.layers.43': 2, 
                'model.layers.44': 2, 
                'model.layers.45': 2, 
                'model.layers.46': 2, 
                'model.layers.47': 2, 
                'model.layers.48': 2, 
                'model.layers.49': 2, 'model.layers.50': 2, 
                'model.layers.51': 2, 'model.layers.52': 2, 'model.layers.53': 2, 
                'model.layers.54': 2, 'model.layers.55': 2, 'model.layers.56': 2, 
                'model.layers.57': 2, 'model.layers.58': 2, 'model.layers.59': 2, 
                'model.layers.60': 2, 'model.layers.61': 2, 'model.layers.62': 2, 
                'model.layers.63.self_attn.q_proj': 2, 'model.layers.63.self_attn.k_proj': 2, 
                'model.layers.63.self_attn.v_proj': 3, 'model.layers.63.self_attn.o_proj': 3, 
                'model.layers.63.self_attn.rotary_emb': 3, 'model.layers.63.mlp': 3, 
                'model.layers.63.input_layernorm': 3, 'model.layers.63.post_attention_layernorm': 3, 
                'model.layers.64': 3, 'model.layers.65': 3, 'model.layers.66': 3, 
                'model.layers.67': 3, 'model.layers.68': 3, 'model.layers.69': 3, 
                'model.layers.70': 3, 'model.layers.71': 3, 'model.layers.72': 3, 'model.layers.73': 3, 
                'model.layers.74': 3, 'model.layers.75': 3, 'model.layers.76': 3, 'model.layers.77': 3, 
                'model.layers.78': 3, 'model.layers.79': 3, 'model.gated_cross_attn_layers.0': 3, 
                'model.gated_cross_attn_layers.1': 3, 'model.gated_cross_attn_layers.2': 3, 
                'model.gated_cross_attn_layers.3': 3, 'model.gated_cross_attn_layers.4': 3, 
                'model.gated_cross_attn_layers.5.alpha_cross_attn': 3, 'model.gated_cross_attn_layers.5.alpha_dense': 3, 
                'model.gated_cross_attn_layers.5.cross_attn': 3, 'model.gated_cross_attn_layers.5.mlp.gate_proj': 3, 
                'model.gated_cross_attn_layers.5.mlp.down_proj': 4, 'model.gated_cross_attn_layers.5.mlp.up_proj': 4, 
                'model.gated_cross_attn_layers.5.mlp.act_fn': 4, 'model.gated_cross_attn_layers.5.input_layernorm': 4, 
                'model.gated_cross_attn_layers.5.post_attention_layernorm': 4, 
                'model.gated_cross_attn_layers.5.act_cross_attn': 4, 'model.gated_cross_attn_layers.5.act_dense': 4, 
                'model.gated_cross_attn_layers.6': 4, 'model.gated_cross_attn_layers.7': 4, 
                'model.gated_cross_attn_layers.8': 4, 'model.gated_cross_attn_layers.9': 4, 
                'model.gated_cross_attn_layers.10': 4, 'model.gated_cross_attn_layers.11': 4, 
                'model.gated_cross_attn_layers.12': 4, 'model.gated_cross_attn_layers.13': 4, 
                'model.gated_cross_attn_layers.14': 4, 'model.gated_cross_attn_layers.15': 4, 
                'model.gated_cross_attn_layers.16': 4, 'model.gated_cross_attn_layers.17': 4, 
                'model.gated_cross_attn_layers.18': 4, 'model.gated_cross_attn_layers.19': 4, 
                'model.norm': 4, 'lm_head': 4},
    "80B_8GPUs": {
                'model.embed_tokens': 0, 
                'model.vision_model': 0, 
                'model.perceiver_resampler': 0, 
                'model.layers.0': 0, 
                'model.layers.1': 0, 
                'model.layers.2': 0, 
                'model.layers.3': 0, 
                'model.layers.4': 0, 
                'model.layers.5': 0, 
                'model.layers.6': 0, 
                'model.layers.7': 0, 
                'model.layers.8': 0, 
                'model.layers.9': 0, 
                'model.layers.10': 0, 
                'model.layers.11': 0, 
                'model.layers.12': 0, 
                'model.layers.13': 1, 
                'model.layers.14': 1, 
                'model.layers.15': 1, 
                'model.layers.16': 1, 
                'model.layers.17': 1, 
                'model.layers.18': 1, 
                'model.layers.19': 1, 
                'model.layers.20.self_attn.q_proj': 1, 
                'model.layers.20.self_attn.k_proj': 1, 
                'model.layers.20.self_attn.v_proj': 1, 
                'model.layers.20.self_attn.o_proj': 1, 
                'model.layers.20.self_attn.rotary_emb': 1, 
                'model.layers.20.mlp': 1, 
                'model.layers.20.input_layernorm': 1, 
                'model.layers.20.post_attention_layernorm': 1, 
                'model.layers.21': 2, 
                'model.layers.22': 2, 
                'model.layers.23': 2, 
                'model.layers.24': 2, 
                'model.layers.25': 2, 
                'model.layers.26': 2, 
                'model.layers.27': 2, 
                'model.layers.28': 2, 
                'model.layers.29': 2, 
                'model.layers.30': 2, 
                'model.layers.31': 2, 
                'model.layers.32': 2, 
                'model.layers.33': 2, 
                'model.layers.34': 2, 
                'model.layers.35': 2, 
                'model.layers.36': 3, 
                'model.layers.37': 3, 
                'model.layers.38': 3, 
                'model.layers.39': 3, 
                'model.layers.40': 3, 
                'model.layers.41.self_attn': 3, 
                'model.layers.41.mlp.gate_proj': 3, 
                'model.layers.41.mlp.down_proj': 3, 
                'model.layers.41.mlp.up_proj': 3, 
                'model.layers.41.mlp.act_fn': 3, 
                'model.layers.41.input_layernorm': 3, 
                'model.layers.41.post_attention_layernorm': 3, 
                'model.layers.42': 3, 
                'model.layers.43': 3, 
                'model.layers.44': 3, 
                'model.layers.45': 3, 
                'model.layers.46': 4, 
                'model.layers.47': 4, 
                'model.layers.48': 4, 
                'model.layers.49': 4, 'model.layers.50': 4, 
                'model.layers.51': 4, 'model.layers.52': 4, 'model.layers.53': 4, 
                'model.layers.54': 4, 
                'model.layers.55': 4, 
                'model.layers.56': 4, 
                'model.layers.57': 4, 
                'model.layers.58': 4, 
                'model.layers.59': 4, 
                'model.layers.60': 5, 
                'model.layers.61': 5, 
                'model.layers.62': 5, 
                'model.layers.63.self_attn.q_proj': 5, 
                'model.layers.63.self_attn.k_proj': 5, 
                'model.layers.63.self_attn.v_proj': 5, 
                'model.layers.63.self_attn.o_proj': 5, 
                'model.layers.63.self_attn.rotary_emb': 5, 
                'model.layers.63.mlp': 5, 
                'model.layers.63.input_layernorm': 5, 
                'model.layers.63.post_attention_layernorm': 5, 
                'model.layers.64': 5, 
                'model.layers.65': 5, 
                'model.layers.66': 5, 
                'model.layers.67': 5, 
                'model.layers.68': 5, 
                'model.layers.69': 5, 
                'model.layers.70': 5, 
                'model.layers.71': 5, 
                'model.layers.72': 5, 
                'model.layers.73': 6, 
                'model.layers.74': 6, 
                'model.layers.75': 6, 
                'model.layers.76': 6, 
                'model.layers.77': 6, 
                'model.layers.78': 6, 
                'model.layers.79': 6, 
                'model.gated_cross_attn_layers.0': 6, 
                'model.gated_cross_attn_layers.1': 6, 
                'model.gated_cross_attn_layers.2': 6, 
                'model.gated_cross_attn_layers.3': 6, 
                'model.gated_cross_attn_layers.4': 6, 
                'model.gated_cross_attn_layers.5.alpha_cross_attn': 6, 
                'model.gated_cross_attn_layers.5.alpha_dense': 6, 
                'model.gated_cross_attn_layers.5.cross_attn': 6, 
                'model.gated_cross_attn_layers.5.mlp.gate_proj': 6, 
                'model.gated_cross_attn_layers.5.mlp.down_proj': 6, 
                'model.gated_cross_attn_layers.5.mlp.up_proj': 6, 
                'model.gated_cross_attn_layers.5.mlp.act_fn': 6, 
                'model.gated_cross_attn_layers.5.input_layernorm': 7, 
                'model.gated_cross_attn_layers.5.post_attention_layernorm': 7, 
                'model.gated_cross_attn_layers.5.act_cross_attn': 7, 
                'model.gated_cross_attn_layers.5.act_dense': 7, 
                'model.gated_cross_attn_layers.6': 7, 
                'model.gated_cross_attn_layers.7': 7, 
                'model.gated_cross_attn_layers.8': 7, 
                'model.gated_cross_attn_layers.9': 7, 
                'model.gated_cross_attn_layers.10': 7, 
                'model.gated_cross_attn_layers.11': 7, 
                'model.gated_cross_attn_layers.12': 7, 
                'model.gated_cross_attn_layers.13': 7, 
                'model.gated_cross_attn_layers.14': 7, 
                'model.gated_cross_attn_layers.15': 7, 
                'model.gated_cross_attn_layers.16': 7, 
                'model.gated_cross_attn_layers.17': 7, 
                'model.gated_cross_attn_layers.18': 7, 
                'model.gated_cross_attn_layers.19': 7, 
                'model.norm': 7, 
                'lm_head': 7}
}

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
    gpu_margin=8,
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
        if "80" in lang_encoder_path:
            
            max_memory_map = get_max_memory()

            for key in max_memory_map.keys():
                max_memory_map[key] = max_memory_map[key] // (1024 * 1024 * 1024)
                max_memory_map[key] = f"{max_memory_map[key] - gpu_margin} GiB"
            print(max_memory_map)
            offload_folder = "/gpfsscratch/rech/dyf/ugz83ue/tmp/offload"

            device_map = "auto" #DEVICE_MAP['80B_8GPUs'] # auto
            lang_encoder = IdeficsForVisionText2Text.from_pretrained(
                lang_encoder_path,
                device_map=device_map,
                offload_folder=offload_folder,
                torch_dtype=torch.bfloat16,
                max_memory=max_memory_map,
                low_cpu_mem_usage=True,
            )
            print("Current device map:", lang_encoder.hf_device_map)

        else:
            lang_encoder = IdeficsForVisionText2Text.from_pretrained(
                lang_encoder_path, local_files_only=use_local_files, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
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
    

split_words_qa = ["Short answer:", "answer:", "Answer:"]

split_words_caption = ["Caption:", "Good Caption:", "Bad Caption:", ":"]
split_words_caption_exp = ["Explanation:", "Bad Explanation:", "Good Explanation:"]

def split_user_assistant(txt, split_pattern=None, instruct_model=False, train=True, merge=False):
    txt = txt.replace("<image>", "").replace("<|endofchunk|>", "\n")
    if  instruct_model:
        txt = txt.replace("\n", "") 
        split_words = split_words_qa
        
        if "Explanation" in txt:
            split_words = split_words_caption_exp
        if "Caption" in txt or 'Describe the' in txt:
            split_words+=split_words_caption
        split_pattern = "(" + "|".join(re.escape(word) for word in split_words) + ")" # "() to include the splitting word"

        c_split = re.split(split_pattern, txt, flags=re.IGNORECASE, maxsplit=1)
        u, a = c_split[0].strip(), "".join(c_split[1:]).strip()

        if "Explanation" not in txt:
            u, a = u.replace(":", ''), a.replace(":", '')

        u = u.replace("Question", "")
        for k in split_words_qa:
            a = a.replace(k.replace(":", ''), "")
        user = "User: "+u+"<end_of_utterance>\n"
        if not train:
            assistant = "Assistant: " + a
        else:
            assistant = "Assistant: "+ a +"<end_of_utterance>\n"

        if merge:
            return [user+assistant]
        else:
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

    if instruct_model:
        context_text = [c.replace("<image>", "").replace("<|endofchunk|>", "\n") for c in context_text]
        if num_shots > 0:
            for txt, img in zip(context_text, context_images):

                u_a = split_user_assistant(txt, instruct_model=instruct_model, merge="merge" in mode)
                if len(u_a) == 2:
                    user, assistant = u_a
                    context += [img, user, assistant]
                else:
                    context += [img]+u_a
        else:
            for txt in context_text:
                u_a = split_user_assistant(txt, instruct_model=instruct_model, merge="merge" in mode)
                
                if len(u_a) == 2:
                    user, assistant = u_a
                    context += [user, assistant]  
                else:
                    context += u_a

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