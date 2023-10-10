import random

import re 




######### get_prompt


def get_prompt(sample, train=True, long_ans=False):
    ans = f"<image>Question:{sample['question'].strip()} Short answer:{sample['answers'][0].strip() if train else ''}{'<|endofchunk|>' if train else ''}"

    if long_ans:
        ans = ans.replace("Short answer", "Answer")
        
    return ans
def get_prompt_explain(sample, sample2=None, train=True, answer_first=True, onlyexplain=False, exp_prompt=' because ', contrastive=False, prevqout=False, 
                           bad_prompt="Bad", good_prompt="Good"):

    
    if contrastive:
        good_answer = random.choice(sample['explanation']).strip().replace(".", "")
        if train:
            bad_answer = sample['prev_q_out'].strip() if prevqout  else random.choice(sample2['explanation']).strip()
            bad_answer = bad_answer.replace(".", "")
        else: 
            bad_answer = ''


        response = f"{good_prompt}{good_answer}. {bad_prompt}{bad_answer}."

        exp_prompt = good_prompt

    else:
        response = exp_prompt+random.choice(sample['explanation']).strip()

    if onlyexplain:

        return f"<image>Question:{sample['question'].strip()} {'Short answer:'+sample['answers'][0].strip()} {response if train else exp_prompt}{'<|endofchunk|>' if train else ''}"

    else:
        if answer_first:
            # if num_shots == 0 and instruction is not None:
            #     return f"<image>Question:{sample['question'].strip()} Short answer:{sample['answers'][0].strip() if train else ''}{' '+instruction+response if train else ''}{'<|endofchunk|>' if train else ''}"
            # else:
            return f"<image>Question:{sample['question'].strip()} Short answer:{sample['answers'][0].strip() if train else ''}{' '+response if train else ''}{'<|endofchunk|>' if train else ''}"
        else:
            return f"<image>Question:{sample['question'].strip()} {response if train else exp_prompt} {'the Short answer:'+sample['answers'][0].strip() if train else ''}{'<|endofchunk|>' if train else ''}"



##############


def get_prompt_vqa_abstention(sample, train=True, answer_first=True, prompt='The question is:', onlyunans=False, prompt_first=False):

    unans = 'no' if sample['answer_type'] in ['unanswerable', "absurd"] else 'yes'
    
    if onlyunans:
        if prompt_first:

            return f"<image>{prompt} '{sample['question'].strip()}' Answer:{unans if train else ''}{'<|endofchunk|>' if train else ''}"
        else:
            return f"<image>Question:{sample['question'].strip()} {prompt+unans if train else prompt}{'<|endofchunk|>' if train else ''}"
    else:
        if answer_first:
            return f"<image>Question:{sample['question'].strip()} Short answer:{sample['answers'][0].strip() if train else ''}{'. '+prompt+unans if train else ''}{'<|endofchunk|>' if train else ''}"
        else:
            return f"<image>Question:{sample['question'].strip()} {prompt+unans if train else prompt}{'. Short answer:'+sample['answers'][0].strip() if train else ''}{'<|endofchunk|>' if train else ''}"


def postprocess_generation_vqa_abstention(predictions, prompt='The question is:', long_ans=False):
    long_splitter = ",|\." if not long_ans else ",,,,,," # dummy splitter to avoid any split
    try:
        return re.split(long_splitter, re.split(f"{prompt}|Question|Answer|answer", predictions, 1)[0], 1)[0]
    except IndexError:
        print("error", predictions)
        return ' '
    
def postprocess_unansgeneration_vqa_abstention(predictions, prompt='The question is:', onlyunans=False):
    try:
        if onlyunans:
            ans = re.split(f"Question|{prompt}|Answer|answer", predictions, 1)[0].strip().replace(".", "")
            ans = "absurd" if ans == 'no' else 'valid'
            return ans
        else:
            return re.findall(f"yes|no", re.split(f"Question|{prompt}|Answer|answer", predictions, 1)[1])[0].strip().replace(".", "")
            # return re.split(f"Question|{prompt}|Answer|answer", predictions, 1)[1].strip().replace(".", "")
    except IndexError:
        print("error", predictions, prompt, re.split(f"Question|{prompt}|Answer|answer", predictions, 1), re.split(prompt, predictions, 1))
        return ' '
    


    

    
############### postprocess

def postprocess_generation_vqa(predictions, long_ans=False):
    long_splitter = ",|\." if not long_ans else ",,,,,," # dummy splitter to avoid any split

    answer = re.split("Question|Answer|answer|Short", predictions, 1)[0]
    answer = re.split(long_splitter, answer, 1)[0]
    return answer







def postprocess_generation_vqa_exponly(predictions, exp_prompt=' because ', long_ans=False):


    long_splitter = ",|\." if not long_ans else ",,,,,," # dummy splitter to avoid any split

        
    return re.split(long_splitter, re.split("Question|Answer|answer", predictions, 1)[0].split(exp_prompt, 1)[0], 1)[0]

def postprocess_expgeneration_vqa_exponly(predictions, bad_prompt="Bad", good_prompt="Good"):
    try:
        return re.split(f"Question|Answer|answer|{bad_prompt}", predictions, 1)[0] #.split("because", 1)[1]
    except IndexError:
        print("error", predictions, re.split("Question|Answer|answer", predictions, 1))
        return ' '


def postprocess_generation_vqa_explain(predictions, exp_prompt='because', long_ans=False):
    long_splitter = ",|\." if not long_ans else ",,,,,," # dummy splitter to avoid any split
    return re.split(long_splitter, re.split("Question|Answer|answer|Explain", predictions, 1)[0].split(exp_prompt, 1)[0], 1)[0]

def postprocess_expgeneration_vqa_explain(predictions, exp_prompt=' because ', contrastive=False, 
                                                       good_prompt="Good:", bad_prompt="Bad:", contrastive_with_ans=False):
    try:
        if contrastive:
            if contrastive_with_ans:
                return re.split(bad_prompt, predictions, 1)[0].split(exp_prompt, 1)[1]
            else:
                return re.split(bad_prompt, predictions, 1)[0].split(good_prompt, 1)[1]
        else:
            return re.split("Question|Answer|answer", predictions, 1)[0].split(exp_prompt, 1)[1]
    except IndexError:
        print("error", predictions)
        return ' '
    
