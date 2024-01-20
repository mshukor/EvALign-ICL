
import random
import re 




################## get prompt
def get_prompt(sample, train=True, label=None):
    if label is None:
        if random.random() > 0.5:
            return f"<image>Which sentence describes the image? a) {sample['pos_caption'].strip()} b) {sample['neg_caption'].strip()} Answer:{'a)' if train else ''}{'<|endofchunk|>' if train else ''}"
        else:
            return f"<image>Which sentence describes the image? a) {sample['neg_caption'].strip()} b) {sample['pos_caption'].strip()} Answer:{'b)' if train else ''}{'<|endofchunk|>' if train else ''}"
    else:
        if label == 'a':
            return f"<image>Which sentence describes the image? a) {sample['pos_caption'].strip()} b) {sample['neg_caption'].strip()} Answer:{'a)' if train else ''}{'<|endofchunk|>' if train else ''}"
        else:
            return f"<image>Which sentence describes the image? a) {sample['neg_caption'].strip()} b) {sample['pos_caption'].strip()} Answer:{'b)' if train else ''}{'<|endofchunk|>' if train else ''}"
       
def postprocess_generation(predictions):
    return predictions.split("Which", 1)[0]



 

def get_prompt_itm(sample, train=True, label=None):
    suffix =  ""
    if label is None:
        if random.random() > 0.5:
            suffix =  ""
            return f"<image>Does the following sentence describe the image?: {sample['pos_caption'].strip()} Answer:{'yes' + suffix if train else ''}{'<|endofchunk|>' if train else ''}"
        else:
            return f"<image>Does the following sentence describe the image?: {sample['neg_caption'].strip()} Answer:{'no' + suffix if train else ''}{'<|endofchunk|>' if train else ''}"
    else:
        if label in ['a', 'yes']:
            return f"<image>Does the following sentence describe the image?: {sample['pos_caption'].strip()} Answer:{'yes' + suffix if train else ''}{'<|endofchunk|>' if train else ''}"
        else:
            return f"<image>Does the following sentence describe the image?: {sample['neg_caption'].strip()} Answer:{'no' + suffix if train else ''}{'<|endofchunk|>' if train else ''}"
       

def postprocess_generation_itm(predictions):
    answer = re.split("Does|User", predictions, 1)[0]
    return answer
        



def get_prompt_itm_objects(sample, train=True, label=None, prompt=' because ', use_objects=False, max_n_objs=10):
    
    objs = ''
    if use_objects:
        objs = []
        obs = sample['objects']

        obs= random.choices(obs, k=len(obs))
        for o in obs:
            item = o['object']
            objs.append(item)
        objs = list(set(objs))
        objs = ", ".join(objs[:max_n_objs])

    suffix =  f" {prompt} {objs}"

    if label is None:
        if random.random() > 0.5:
            return f"<image>Does the following sentence describe the image?: {sample['pos_caption'].strip()} Answer:{'yes' + suffix if train else ''}{'<|endofchunk|>' if train else ''}"
        else:
            return f"<image>Does the following sentence describe the image?: {sample['neg_caption'].strip()} Answer:{'no' + suffix if train else ''}{'<|endofchunk|>' if train else ''}"
    else:
        if label in ['a', 'yes']:
            return f"<image>Does the following sentence describe the image?: {sample['pos_caption'].strip()} Answer:{'yes' + suffix if train else ''}{'<|endofchunk|>' if train else ''}"
        else:
            return f"<image>Does the following sentence describe the image?: {sample['neg_caption'].strip()} Answer:{'no' + suffix if train else ''}{'<|endofchunk|>' if train else ''}"
       

def postprocess_generation_itm_objects(predictions, prompt=None):
    if prompt is not None:
        return predictions.split(prompt.strip(), 1)[0]
    else:
        return predictions.split("Does", 1)[0]