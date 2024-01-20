

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist

import argparse
import json 
import utils
from itertools import chain
from functools import partial
import re 


INSTRUCTIONS = {
    'truthful': "You are a language assistant that helps find which one of the two answers is better according to these criteria: accurate, honest and truthful in answering the question from the image. The information about the image is contained in the Image Context. Image Context contains; image descriptions and objects with location as bounding box in the form of x(top left corner), y(top left corner), w(width), h(height).  Decide which one of the answers is better and explain your decision.",
    'truthful_cot': "You are a language assistant that helps find which one of the two answers is better according to these criteria: accurate, honest and truthful in answering the question from the image. The information about the image is contained in the Image Context. Image Context contains; image descriptions and objects with location as bounding box in the form of x(top left corner), y(top left corner), w(width), h(height).  Decide which one of the answers is better and explain your decision. Before answering, think step by step:",
    'truthful_phrases': "You are a language assistant that helps find which one of the two answers is better according to these criteria: accurate, honest and truthful in answering the question from the image. The information about the image is contained in the Image Context. Image Context contains; image descriptions and objects with location as bounding box in the form of x(top left corner), y(top left corner), w(width), h(height).  Decide which one of the answers is better and explain your decision. First compare each phrase in both answers and then provide a general recommendation:",
}


def get_prompt_general(sample, instruction_type='truthful'):

    instruction = INSTRUCTIONS[instruction_type]
    examples = f"Image Context contains the following descriptions and objects:  Objects; {','.join(sample['bbox'])}. Descriptions;{'.'.join(sample['caps'])}"
    query = f"Question: {sample['question']} Answer 1: {sample['output_1']} \n Answer 2: {sample['output_2']}"

    messages = [
        {"role": "user", "content": f"{instruction}\n {examples}\n {query}\n"},
    ]

    return messages

def get_response_general(response, keyword='Preference'):

    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    try:
        res = pattern.split(response)[1]
    except:
        pass
    pref = re.findall(r'\d+', res)
    pref = 1 if not pref else int(pref[0])

    return pref

def compute_accuracy(p, t):
    res = [1 if p[i] == t[i] else 0 for i in range(len(p))]
    acc = sum(res)/len(res)
    return acc*100.

class PreferenceDataset(Dataset):
    def __init__(self, args=None):
        super().__init__()

        self.path = args.path
        self.args = args 

        data = json.load(open(self.path))
        # ann_path = '/data/mshukor/data/coco/annotations/captions_train2017.json'
        self.ann_caption = json.load(open(args.ann_caption_path)) if args.ann_caption_path else None
        # ann_inst_path = '/data/mshukor/data/coco/annotations/instances_train2017.json'
        self.ann_inst = json.load(open(args.ann_inst_path)) if args.ann_inst_path else None


        if self.ann_inst is not None:
            catid_2_cat = {k['id']: k['name'] for k in self.ann_inst['categories']}
            self.imid_2_bbox = {}
            self.imid_2_cls = {}
            for d in self.ann_inst['annotations']:
                image_id = d['image_id']
                x, y, w, h = d['bbox'] # top left x position, top left y position, width, height
                category_id = d['category_id']
                cat = catid_2_cat[category_id]
                item = f"{cat} at x:{x}, y:{y}, w:{w}, h:{h}"
                if image_id not in self.imid_2_bbox:
                    self.imid_2_bbox[image_id] = [item]  
                    self.imid_2_cls[image_id] = [cat]
                else:
                    self.imid_2_bbox[image_id] += [item]
                    self.imid_2_cls[image_id] += [cat]

        if self.ann_caption is not None:
            self.imid_2_caps = {}
            for d in self.ann_caption['annotations']:
                image_id = d['image_id']
                if image_id not in self.imid_2_caps:
                    self.imid_2_caps[image_id] = [d['caption']]  
                else:
                    self.imid_2_caps[image_id] += [d['caption']]

        self.data = []
        for item in data:

            image_id = int(item['image'].split('.')[0])


            bbox = self.imid_2_bbox[image_id]
            caps = self.imid_2_caps[image_id]

            question = item['conversations'][-2]['value'].replace('<image>', '')
            output_1 = item['output_1']['value']
            output_2 = item['output_2']['value']
            label = item['preference']

            new_datum = {
                'question': question,
                'output_1': output_1,
                'output_2': output_2,
                'label': label,
                'caps': caps,
                'bbox': bbox,
            }

            self.data.append(new_datum)




    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        datum = self.data[idx]

        return datum

    # def collate_fn(self, batch):
    #     batch_entry = {}

    #     B = len(batch)



    #     question = []
    #     output_1 = []
    #     output_2 = []
    #     input_text = []
    #     label = []
    #     sents = []

    #     for i, entry in enumerate(batch):

    #         images.append(entry['image'])
    #         img_ids.append(entry['img_id'])

    #         if 'target_ids' in entry:
    #             target_ids[i, :entry['target_length']] = entry['target_ids']



    #         if 'targets' in entry:
    #             targets.append(entry['targets'])
    #         if 'sent' in entry:
    #             sents.append(entry['sent'])


    #     batch_entry['images'] = torch.stack(images)
    #     batch_entry['img_id'] = img_ids
    #     batch_entry['img_paths'] = img_paths
    #     if 'sent' in entry:
    #         batch_entry['sent'] = sents



    #     batch_entry['targets'] = targets

    #     batch_entry['task'] = 'caption'

    #     return batch_entry
    



parser = argparse.ArgumentParser()
parser.add_argument("--lm_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
parser.add_argument("--path", type=str, default='/data/mshukor/data/llava/llava_7b_v1_preference.json')
parser.add_argument("--ann_caption_path", type=str, default='/data/mshukor/data/llava/llava_7b_v1_preference.json')
parser.add_argument("--ann_inst_path", type=str, default='/data/mshukor/data/llava/llava_7b_v1_preference.json')
parser.add_argument("--mode", default="", type=str)
parser.add_argument("--save_path", default="/data/mshukor/logs/rlaif/preferences.json", type=str)


parser.add_argument("--batch_size", default=8, type=int)



@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, accelerator=None, max_length=1000, only_main=False, mode='general'):
    # test
    # model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate Caption test result:'
    print_freq = 50
    
        
    predictions = []
    raw_predictions = []
    targets = []

    pad_token = tokenizer.pad_token
    eos_token = tokenizer.eos_token

    num_beams = 3
    do_sample = False
    accelerator.print("num_beams", num_beams, "do_sample", do_sample, "max_length", max_length)


    if mode == 'general':
        instruction_type = 'truthful'
        get_prompt = partial(get_prompt_general, instruction_type=instruction_type)
        get_response = get_response_general
    else:
        get_prompt = get_prompt_general
        get_response = get_response_general

    for n, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        

        messages = [get_prompt(s) for s in batch]
        text_input = tokenizer.apply_chat_template(messages, return_tensors="pt", padding='longest').to(device) 

        out = model(text_input, return_dict=True, max_length=max_length, 
                    do_sample=do_sample, num_beams=num_beams)
        
        generated_ids = out['sequences']
        generated_ids = generated_ids[:, text_input['input_ids'].shape[1]:]
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


        out_decode = []
        raw_response = []
        for i, o in enumerate(generated_text):
            response = get_response(o) 
            raw_response.append({"output": o, "input": batch[i]})
            out_decode.append(response)


        predictions.extend(out_decode)
        raw_predictions.extend(raw_response)
        targets.extend(batch['label'])



    if dist.get_world_size() > 1 and not only_main:
        gather_predictions = [None for _ in range(dist.get_world_size())]
        gather_targets = [None for _ in range(dist.get_world_size())]

        dist.all_gather_object(gather_predictions, predictions)
        dist.all_gather_object(gather_targets, targets)

        predictions = list(chain(*gather_predictions)) 
        targets = list(chain(*gather_targets)) 

    acc = compute_accuracy(predictions, targets)

    return acc, raw_predictions


def main():


    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    accelerator.print("Create model")

    accelerator.print("Loading model: ", args.lm_path)
    model = AutoModelForCausalLM.from_pretrained(args.lm_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.lm_path)

    dataset = PreferenceDataset(args=args)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
        sampler=None,
        drop_last=False)
    
    model, loader = accelerator.prepare(model, loader)


    # model.eval()

    acc, raw_predictions = evaluation(model, loader, tokenizer, device, accelerator=accelerator, max_length=1000, only_main=True, mode=args.mode)



    accelerator.print("Accuracy: ", acc)
    accelerator.print("save results to:", args.save_path)
    with open(args.save_path, "w") as f:
        json.dump(raw_predictions, f)

