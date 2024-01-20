import json
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from open_flamingo.eval.imagenet_utils import IMAGENET_1K_CLASS_ID_TO_LABEL
import copy
import random 

from tqdm import tqdm 

def word_count(string):
    return len(string.split())

class COCOFlickrDataset(Dataset):
    def __init__(
        self,
        image_dir_path="/mmfs1/gscratch/efml/anasa2/data/coco/train2017/",
        annotations_path="/mmfs1/gscratch/efml/anasa2/data/coco/annotations/captions_train2017.json",
        is_flickr=False,
        previous_query_predictions=None,
        is_llava=False,
        seed=0,
        long_captions=None,
    ):
        self.image_dir_path = image_dir_path
        self.is_flickr = is_flickr
        self.is_llava = is_llava
        max_ans_len = 400 # ~ 90 words 
        self.long_captions = long_captions
        print("max_ans_len: ", max_ans_len)
        print("long_captions:", long_captions)
        print("annotations from: ", annotations_path, image_dir_path)
        if self.is_llava:
            annotations = json.load(open(annotations_path))
            for ann in annotations:
                if "dialog" in ann:
                    dialogue = []
                    for example in ann['dialog']:
                        answer =  example['agent'][:max_ans_len//2]
                        dialogue.append(f"Question: {example['human']} Answer: {answer}")
                    ann['caption'] = " ".join(dialogue)
                else:
                    ann['caption'] = ann['answer'][:max_ans_len]
                
            self.annotations = annotations
        else:
            self.annotations = json.load(open(annotations_path))["annotations"]


        self.previous_query_predictions = previous_query_predictions
        if self.previous_query_predictions is not None:
            if 'seed' in previous_query_predictions:
                previous_query_predictions_ = self.previous_query_predictions.replace("seed", str(seed))
            else:
                previous_query_predictions_ = self.previous_query_predictions
            print("load previous predictions from:", previous_query_predictions_)
            predictions = json.load(open(previous_query_predictions_, "r"))
            if "annotations" in predictions:
                predictions = predictions['annotations']
                
            self.imgid_to_preds = {}

            for p in tqdm(predictions):
                self.imgid_to_preds[p['image_id']] = p['caption'] 

            self.count = 0

    def __len__(self):
        return len(self.annotations)

    def get_img_path(self, idx):
        if self.is_flickr:
            return f"{self.image_dir_path}/{self.annotations[idx]['image_id']}.jpg"
        
        elif self.is_llava:
            if "train" in self.image_dir_path:
                return f"{self.image_dir_path}/COCO_train2014_{self.annotations[idx]['image_id']:012d}.jpg"
            else:
                return f"{self.image_dir_path}/COCO_val2014_{self.annotations[idx]['image_id']:012d}.jpg"
        else:
            # return f"{self.image_dir_path}/{self.annotations[idx]['file_name']}"
            if 'train' in self.image_dir_path:
                return f"{self.image_dir_path}/COCO_train2014_{self.annotations[idx]['image_id']:012d}.jpg"
            else:
                return f"{self.image_dir_path}/COCO_val2014_{self.annotations[idx]['image_id']:012d}.jpg"

    def __getitem__(self, idx):
        try:
            ann = self.annotations[idx]
            image_id = ann['image_id']
            img_path = self.get_img_path(idx)
            image = copy.deepcopy(Image.open(img_path))
        except Exception as e:
            idx = random.randint(0, len(self) - 1)
            print(e, "sample new idx", idx)
            return self.__getitem__(idx)

        previous_query_predictions = ""
        if self.previous_query_predictions is not None:
            previous_query_predictions = self.imgid_to_preds.get(ann["image_id"], "")
            if ann["image_id"] not in self.imgid_to_preds:
                self.count+=1
                # print("no prev out for this query:", self.annotations[idx], "sample new query")
                # new_idx = random.randint(0, len(self) - 1)
                # return self.__getitem__(new_idx)


        caption =ann["caption"]
        if isinstance(caption, list):
            if self.long_captions is not None:
                caps = sorted(caption, key=word_count)
                l_cap = caps[-1]
                s_cap = caps[0]
                if self.long_captions:
                    caption = l_cap
                    previous_query_predictions = s_cap
                else:
                    caption = s_cap
                    previous_query_predictions = l_cap
            else:
                caption = random.choice(caption)

        caption = caption + '.' if caption[-1] != '.' else caption
        
        question_id = ann.get("question_id", "image_id")
        question = ann.get("question", "")
        
        item = {
            "image": image,
            "caption": caption,
            "image_id": self.annotations[idx]["image_id"],
            "img_path": img_path,
            "image_id": image_id,
            "prev_q_out": previous_query_predictions,
            "question_id": question_id,
            "question": question,
            "ann": ann["caption"],
        }
        if 'objects' in self.annotations[idx]:
            item['objects'] = self.annotations[idx]['objects']

        return item 



class ITMDataset(Dataset):
    def __init__(
        self,
        image_dir_path="/mmfs1/gscratch/efml/anasa2/data/coco/train2017/",
        annotations_path="/mmfs1/gscratch/efml/anasa2/data/coco/annotations/captions_train2017.json",
        compos='sys',
        neg_type='atom',
        n_prod=4,
    ):
        self.image_dir_path = image_dir_path
        self.compos = compos # 'sys' 'prod'
        self.neg_type = neg_type # sys: 'atom' 'comp' 'combined', prod: 'hard_negs'
        self.n_prod = n_prod
        annotations = json.load(open(annotations_path))
        pos_key = 'caption'

        if self.compos == 'sys':
            neg_key = f'valid_hard_negs_{self.neg_type}'
        elif self.compos == 'prod':
            neg_key = self.neg_type
        elif self.compos == 'sugar':
            neg_key = "negative_caption"
        else:
            self.negative_captions = {a['image_id']: a['caption'] for a in annotations}
            neg_key='caption'
            
        self.annotations = []
        for ann in annotations:
            if self.compos == 'prod' and (int(ann['n']) != self.n_prod and self.n_prod >= 4):
                continue
            self.annotations.append(ann)

        for ann in self.annotations:
            ann['pos_caption'] = ann[pos_key]
            if self.compos in ['normal']:
                ann['neg_caption'] = random.choices(self.annotations, k=1)[0][neg_key]
            else:
                ann['neg_caption'] = ann[neg_key]

            

        print("data size", len(self.annotations))

    def __len__(self):
        return len(self.annotations)

    def get_img_path(self, idx):

        return os.path.join(self.image_dir_path, self.annotations[idx]['image_path']) 

    def __getitem__(self, idx):
        try:
            item = self.annotations[idx]
            image_id = item['image_id']
            img_path = self.get_img_path(idx)
            image = copy.deepcopy(Image.open(img_path))
        except Exception as e:
            idx = random.randint(0, len(self) - 1)
            print(e, "sample new idx", idx)
            return self.__getitem__(idx)

        pos_caption = item["pos_caption"]
        pos_caption = pos_caption + '.' if pos_caption[-1] != '.' else pos_caption
        
        neg_caption = item["neg_caption"]
        if isinstance(neg_caption, list):
            neg_caption = random.choices(neg_caption, k=1)[0]
                                         
        neg_caption = neg_caption + '.' if neg_caption[-1] != '.' else neg_caption
        
        relations = ''
        if "relations" in item:
            relations = item['relations']

        objects = ''
        if "objects" in item:
            objects = item['objects']
            
        image_w, image_h = item.get('image_w', 0), item.get('image_h', 0)
        x, y = int(float(item.get('x', 0))), int(float(item.get('y', 0)))
        width, height = int(float(item.get('width', 0))), int(float(item.get('height', 0)))

        item = {
            "image": image,
            "pos_caption": pos_caption,
            "neg_caption": neg_caption,
            "image_id": image_id,
            "img_path": img_path,
            "relations": relations,
            "objects": objects,
            "image_w": image_w,
            "image_h": image_h,
            'x': x,
            'y': y,
            'width': width,
            'height': height,
        }


        return item 
    

class VQADataset(Dataset):
    def __init__(
        self,
        image_dir_path="/mmfs1/gscratch/efml/anasa2/data/vqav2/train2014/",
        question_path=None,
        annotations_path="/mmfs1/gscratch/efml/anasa2/data/vqav2/v2_mscoco_train2014_annotations.json",
        vqa_dataset="vqa",
        previous_predictions=None,
        previous_query_predictions=None,
        is_llava=False,
        seed=0,
        args=None,
    ):
        print(question_path, annotations_path, image_dir_path, previous_predictions, previous_query_predictions)
        self.questions = None
        if question_path is not None:
            self.questions = json.load(open(question_path, "r"))["questions"]

        self.is_llava = is_llava
        max_ans_len = 400 # ~ characters
        max_rounds = 2
        print("max_ans_len: ", max_ans_len, "max_rounds: ", max_rounds)

        if self.is_llava:
            self.answers = json.load(open(annotations_path, "r"))
            for ann in self.answers:
                question_id = ann.get("question_id", ann['anns_id'])
                if "dialog" in ann:
                    if "dialog" in args.mode:
                        max_ans_len = 200 # ~ characters
                        q_as = []
                        diag = ann['dialog'][-max_rounds:]
                        len_dialog = len(diag)
                        for i in range(len_dialog):
                            example = diag[i]

                            question =  "Question:"+example['human'][:max_ans_len] if i > 0 else example['human'][:max_ans_len]
                            answer = "Answer:"+example['agent'] if i < (len_dialog - 1) else ""

                            q_a = f"{question} {answer}"

                            q_as.append(q_a)


                        q_as = " ".join(q_as)

                        ann['question'] = {'question': q_as, 'image_id': ann['image_id'], 'question_id': question_id}
                        ann['answers'] = [{'answer': diag[-1]["agent"]}]
                    else:
                        example = random.choice(ann['dialog'])

                        answer =  example['agent'][:max_ans_len]
                        question = example['human']

                        ann['question'] = {'question': question, 'image_id': ann['image_id'], 'question_id': question_id}
                        ann['answers'] = [{'answer': answer}]
                else:
                    ann['question'] = {'question': ann['question'], 'image_id': ann['image_id'], 'question_id': question_id}
                    ann['answers'] = [{'answer': ann['answer'][:max_ans_len]}]

        else:
            self.answers = json.load(open(annotations_path, "r"))["annotations"]
        self.image_dir_path = image_dir_path
        self.vqa_dataset = vqa_dataset

        self.previous_predictions = previous_predictions
        if self.previous_predictions is not None:
            if 'seed' in previous_predictions:
                previous_predictions = self.previous_predictions.replace("seed", str(seed))
            else:
                previous_predictions = self.previous_predictions
            print("load previous predictions from:", previous_predictions)
            predictions = json.load(open(previous_predictions, "r"))
            self.quesid_to_preds = {}

            for p in tqdm(predictions):
                self.quesid_to_preds[p['question_id']] = p['prediction'] 

            self.count = 0


        self.previous_query_predictions = previous_query_predictions
        if self.previous_query_predictions is not None:
            if 'seed' in previous_query_predictions:
                previous_query_predictions_ = self.previous_query_predictions.replace("seed", str(seed))
            else:
                previous_query_predictions_ = self.previous_query_predictions
            print("load previous query predictions from:", previous_query_predictions_)
            predictions = json.load(open(previous_query_predictions_, "r"))
            self.query_quesid_to_preds = {}

            for p in tqdm(predictions):
                self.query_quesid_to_preds[p['question_id']] = p['answer'] 

            self.query_count = 0


    def __len__(self):
        return len(self.answers)

    def get_img_path(self, question):
        if self.vqa_dataset in ["vqa", 'vqax', 'tdiuc', 'llava']:
            if 'train' in self.image_dir_path:
                return os.path.join(
                    self.image_dir_path, f"COCO_train2014_{question['image_id']:012d}.jpg"
                )
            else:
                return os.path.join(
                    self.image_dir_path, f"COCO_val2014_{question['image_id']:012d}.jpg"
                )
        elif self.vqa_dataset == "vizwiz":
            return os.path.join(
                self.image_dir_path, question['image']
            )
        elif self.vqa_dataset == "ok_vqa":
            return os.path.join(
                self.image_dir_path, f"COCO_val2014_{question['image_id']:012d}.jpg"
            )
        else:
            raise Exception(f"Unknown VQA dataset {self.vqa_dataset}")

    def __getitem__(self, idx):

        answers = self.answers[idx]

        if 'question'  not in answers:
            question = self.questions[idx]
        else:
            question = answers['question']

        img_path = self.get_img_path(question)
    
        try:
            image = copy.deepcopy(Image.open(img_path))
        except Exception as e:
            idx = random.randint(0, len(self) - 1)
            print(e, "sample new idx", idx)
            return self.__getitem__(idx)



        explanation = ['']
        if 'explanation' in answers:
            explanation = answers['explanation']

        captions = ['']
        if 'captions' in answers:
            captions = answers['captions']

        objects = ['']
        if 'objects' in answers:
            objects = ' '.join(answers['objects'])

        answer_type = ''
        if 'answer_type' in answers:
            answer_type = answers['answer_type']
        elif 'question_type' in answers:
            answer_type = answers['question_type']


        
        scores_dic = {}
        for a_s in answers["answers"]:
            if a_s['answer'] in scores_dic:
                scores_dic[a_s['answer']]+= 1
            else:
                scores_dic[a_s['answer']]= 1
        
        answers = [k for k, v in sorted(scores_dic.items(), key=lambda pair: pair[1], reverse=True)]

        previous_unans_prediction = ""
        if self.previous_predictions is not None:
            previous_unans_prediction = self.quesid_to_preds.get(question["question_id"], "")
            if question["question_id"] not in self.quesid_to_preds:
                self.count+=1
                # print(self.count)

        previous_query_predictions = ""
        if self.previous_query_predictions is not None:
            previous_query_predictions = self.query_quesid_to_preds.get(question["question_id"], "")
            if question["question_id"] not in self.query_quesid_to_preds:
                self.query_count+=1


        return {
            "image": image,
            "question": question["question"],
            "answers": answers, # [a["answer"] for a in answers["answers"]]
            "question_id": question["question_id"],
            "explanation": explanation,
            "objects": objects,
            "captions": captions,
            "answer_type": answer_type,
            "img_path": img_path,
            "image_id": question['image_id'],
            "prev_unans_pred": previous_unans_prediction,
            "prev_q_out": previous_query_predictions,
        }


class ImageNetDataset(ImageFolder):
    """Class to represent the ImageNet1k dataset."""

    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        target_label = IMAGENET_1K_CLASS_ID_TO_LABEL[target]
        return {
            "image": sample,
            "class_id": target,  # numeric ID of the ImageNet class
            "class_name": target_label,  # human-readable name of ImageNet class
        }
