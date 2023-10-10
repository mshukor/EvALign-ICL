
import os
import nltk 
import random 
from pattern.en import singularize

from tqdm import tqdm


import os
import random
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm




def aggregate_results(scores, results, shot):

    if isinstance(scores[0], dict):
        all_scores = {}
        for s in scores:
            for k, v in s.items():
                all_scores[k] = all_scores[k]+[v] if k in all_scores else [v]
        mean_scores = {k: round(np.mean(s), 4) for k, s in all_scores.items()}
        stddev_scores = {k: round(np.nanstd(s), 4) for k, s in all_scores.items()} 
        print(mean_scores, stddev_scores)
        print(f"Shots {shot} Mean  score: {mean_scores.values()}")
        print(f"Shots {shot} STD  score: {stddev_scores.values()}")
        results["coco"].append(
            {"shots": shot, "trials": scores, "mean": list(mean_scores.values()), "stddev": list(stddev_scores.values()), "scores": mean_scores}
        )
    else:
        print(f"Shots {shot} Mean score: {np.mean(scores)}")
        print(f"Shots {shot} STD  score: {np.nanstd(scores)}")
        results["coco"].append(
            {"shots": shot, "trials": scores, "mean": np.mean(scores), "std": np.nanstd(scores)}
        )
        
    return results



def get_random_indices(num_samples, query_set_size, full_dataset, seed, query_dataset=None):
    if query_dataset is None and num_samples + query_set_size > len(full_dataset):
        raise ValueError(
            f"num_samples {num_samples} + num_shots {query_set_size}  must be less than {len(full_dataset)}"
        )

    # get a random subset of the dataset
    np.random.seed(seed)
    if query_dataset is not None:
        random_indices = np.random.choice(
            len(full_dataset), num_samples, replace=False
        )
        query_indices = np.random.choice(
            len(query_dataset), query_set_size, replace=False
        )

        return (random_indices, query_indices)
    else:
        random_indices = np.random.choice(
            len(full_dataset), num_samples + query_set_size, replace=False
        )
        return random_indices


def prepare_eval_samples_and_dataset(full_dataset, random_indices, query_set_size, query_indices=None, query_dataset=None, mode=None):
    # get in context samples
    if query_dataset is not None:
        in_context_samples = [query_dataset[i] for i in query_indices[:query_set_size]]
        eval_dataset = torch.utils.data.Subset(
            full_dataset, random_indices
        )
    else:
        in_context_samples = [full_dataset[i] for i in random_indices[:query_set_size]]

        eval_dataset = torch.utils.data.Subset(
            full_dataset, random_indices[query_set_size:]
        )
    return in_context_samples, eval_dataset


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


def get_context_text(
    get_prompt: Callable[[dict], str],
    in_context_samples,
    effective_num_shots,
    num_shots,
    neg_samples=None,
    mode=None,
    text_prompt='',
    instruction='',
) -> str:

    if mode is None:
        context_text = (
            "".join([get_prompt(s) for s in in_context_samples])
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

        context_text = ("".join(text))
    else:
        context_text = (
            "".join([get_prompt(s) for s in in_context_samples])
            if effective_num_shots > 0
            else ""
        )
       
    context_text = text_prompt+context_text

    if num_shots == 0 or ( mode is not None and "blind" in mode):
        context_text = context_text.replace("<image>", "")

    context_text = f"{instruction} {context_text}"
    return context_text


def prepare_batch_images(batch, image_processor, context_images, num_shots):
    batch_images = None
    for b, sample_imgs in zip(batch, context_images):
        b_image = image_processor(b["image"]).unsqueeze(0).unsqueeze(1).unsqueeze(0)
        b_image = torch.cat([sample_imgs, b_image], dim=1) if num_shots > 0 else b_image

        if batch_images is None:
            batch_images = b_image
        else:
            batch_images = torch.cat([batch_images, b_image], dim=0)
    return batch_images


def sample_batch_demos_from_query_set(query_set, num_samples, batch_size, mode=None):

    icl_examples = [random.sample(query_set, num_samples) for _ in range(batch_size)]
    
    return icl_examples


def get_outputs(
    model,
    batch_images,
    device,
    attention_mask,
    max_generation_length,
    num_beams,
    length_penalty,
    input_ids,
):
    with torch.inference_mode():
        outputs = model.generate(
            batch_images.to(device if device >= 0 else "cpu"),
            input_ids.to(device if device >= 0 else "cpu"),
            attention_mask=attention_mask.to(device if device >= 0 else "cpu"),
            max_new_tokens=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

    outputs = outputs[:, len(input_ids[0]) :]
    return outputs



class COCONegativeObjects(object):

    def __init__(self, synonym_path):



        #read in synonyms
        
        synonyms = open(os.path.join(synonym_path)).readlines()
        synonyms = [s.strip().split(', ') for s in synonyms]
        self.synonyms = {}
        self.mscoco_objects = [] #mscoco objects and *all* synonyms
        self.inverse_synonym_dict = {}
        for synonym in tqdm(synonyms):
            self.mscoco_objects.extend(synonym)
            self.synonyms[synonym[0]] = synonym
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]
                

        #Some hard coded rules for implementing CHAIR metrics on MSCOCO
        
        #common 'double words' in MSCOCO that should be treated as a single word
        coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track']
        
        #Hard code some rules for special cases in MSCOCO
        #qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
        #qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
        vehicle_words = ['jet', 'train']
        
        #double_word_dict will map double words to the word they should be treated as in our analysis
        
        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict['baby %s' %animal_word] = animal_word
            self.double_word_dict['adult %s' %animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' %vehicle_word] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'
        
    def caption_to_words(self, caption):

            '''
            Input: caption
            Output: MSCOCO words in the caption
            '''

            #standard preprocessing
            words = nltk.word_tokenize(caption.lower())
            words = [singularize(w) for w in words]
            
            tokenized_words = words.copy()
            #replace double words
            i = 0
            double_words = []
            idxs = []
            while i < len(words):
                idxs.append(i) 
                double_word = ' '.join(words[i:i+2])
                if double_word in self.double_word_dict: 
                    double_words.append(self.double_word_dict[double_word])
                    i += 2
                else:
                    double_words.append(words[i])
                    i += 1
            words = double_words

            #toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
            if ('toilet' in words) & ('seat' in words): words = [word for word in words if word != 'seat']

            #get synonyms for all words in the caption
            idxs = [idxs[idx] for idx, word in enumerate(words) \
                    if word in set(self.mscoco_objects)]
            words = [word for word in words if word in set(self.mscoco_objects)]
            node_words = []
            for word in words:
                node_words.append(self.inverse_synonym_dict[word])
            #return all the MSCOCO objects in the caption
            return tokenized_words, words, node_words, idxs, double_words
        
    def remove_item(self, key, l):
        
        return [n for n in l if n != key]
    
    def get_neg_caption(self, caption, num_replacement=1):
        
        tokenized_words, words, node_words, idxs, double_words = self.caption_to_words(caption)
        
        replaced = 0
        for i  in range(len(node_words)):
            
            if replaced < num_replacement:
                new_object = random.choice(self.remove_item(node_words[i], list(self.synonyms.keys())))
                new_word = random.choice(self.synonyms[new_object])

                if new_word != tokenized_words[idxs[i]]:
                    tokenized_words[idxs[i]] = new_word
                    
                    replaced+=1
                        
        new_caption = ' '.join(tokenized_words)

        new_sample = {}
        new_sample['caption'] = new_caption

        return new_sample, replaced