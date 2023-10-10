import re 




################## get prompt

def get_prompt(sample, prompt='Output:'):
    prompt_ = prompt if prompt  else 'Output'
    return f"<image>{prompt_}{sample['caption'].strip()}<|endofchunk|>"


    

def postprocess_generation(predictions, prompt='Output'):
    prompt_ = prompt if prompt else 'Output'
    return predictions.split(f"{prompt_}", 1)[0]




###############

def get_prompt_caption_objects(sample1, objects_first=True, objects_prompt="Objects:", train=True):
    if objects_first:
        return f"<image>{objects_prompt}{' '.join(sample1['objects']).strip()}. Caption:{sample1['caption'].strip()}<|endofchunk|>"
    else:
        return f"<image>Caption:{sample1['caption'].strip()} {objects_prompt}{' '.join(sample1['objects']).strip()}<|endofchunk|>"

def postprocess_generation_caption_objects(predictions, objects_prompt="Caption:", objects_first=True):
    if objects_first:
        try:
            return predictions.split("Caption", 1)[1].split(objects_prompt)[0]
        except:
            return " "
    else:
        return predictions.split(objects_prompt, 1)[0]






