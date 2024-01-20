import argparse
import json
from math import ceil
import os
import random
from collections import defaultdict

import more_itertools
import numpy as np
import torch
from coco_metric import compute_cider
from eval_datasets import COCOFlickrDataset, VQADataset, ImageNetDataset, ITMDataset
from tqdm import tqdm

from open_flamingo.eval.ok_vqa_utils import postprocess_ok_vqa_generation
from vqa_metric import compute_vqa_accuracy, f1_score
from open_flamingo.eval.classification import (
    compute_per_sample_probs,
    compute_per_sample_loss,
)
from open_flamingo.eval.imagenet_utils import (
    openai_imagenet_classnames,
    IMAGENET_1K_CLASS_ID_TO_LABEL,
)


from open_flamingo.eval import vqa_utils
from accelerate import Accelerator
from functools import partial



from open_flamingo.eval.utils import (
    get_random_indices, 
    prepare_eval_samples_and_dataset,
    sample_batch_demos_from_query_set,
    aggregate_results,
)


from open_flamingo.eval.idefics_utils import (
    create_model_and_transforms,
    get_outputs,
    get_context,
    split_user_assistant,
)


import shortuuid


parser = argparse.ArgumentParser()
parser.add_argument("--lm_path", type=str, default="facebook/opt-1.3b")
parser.add_argument("--lm_tokenizer_path", type=str, default="facebook/opt-30b")
parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument(
    "--cross_attn_every_n_layers",
    type=int,
    default=1,
    help="how often to add a cross-attention layer after each transformer layer",
)
parser.add_argument(
    "--results_file", type=str, default=None, help="JSON file to save results"
)

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0, 4, 8, 16, 32], type=int)
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    default=[0],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples", type=int, default=5000, help="Number of samples to evaluate on"
)

parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--device", type=int, default=0)

# Per-dataset evaluation flags
parser.add_argument(
    "--eval_coco",
    action="store_true",
    default=False,
    help="Whether to evaluate on COCO.",
)
parser.add_argument(
    "--eval_vqav2",
    action="store_true",
    default=False,
    help="Whether to evaluate on VQAV2.",
)
parser.add_argument(
    "--eval_ok_vqa",
    action="store_true",
    default=False,
    help="Whether to evaluate on OK-VQA.",
)
parser.add_argument(
    "--eval_imagenet",
    action="store_true",
    default=False,
    help="Whether to evaluate on ImageNet.",
)

parser.add_argument(
    "--eval_flickr30",
    action="store_true",
    default=False,
    help="Whether to evaluate on Flickr30.",
)

# Dataset arguments

## Flickr30 Dataset
parser.add_argument(
    "--flickr_image_dir_path",
    type=str,
    help="Path to the flickr30/flickr30k_images directory.",
    default=None,
)
parser.add_argument(
    "--flickr_annotations_json_path",
    type=str,
    help="Path to the dataset_flickr30k_coco_style.json file.",
    default=None,
)

## COCO Dataset
parser.add_argument(
    "--coco_image_dir_path",
    type=str,
    help="Path to the flickr30/flickr30k_images directory.",
    default=None,
)
parser.add_argument(
    "--coco_annotations_json_path",
    type=str,
    default=None,
)

## VQAV2 Dataset
parser.add_argument(
    "--vqav2_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_annotations_json_path",
    type=str,
    default=None,
)

## OK-VQA Dataset
parser.add_argument(
    "--ok_vqa_image_dir_path",
    type=str,
    help="Path to the vqav2/train2014 directory.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_train2014_questions.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_annotations_json_path",
    type=str,
    help="Path to the v2_mscoco_train2014_annotations.json file.",
    default=None,
)

## Imagenet dataset
parser.add_argument("--imagenet_root", type=str, default="/tmp")


parser.add_argument(
    "--low_cpu",
    action="store_true",
    default=False,
    help="low cpu",
)

parser.add_argument("--precision", type=str, default="float32")

parser.add_argument("--mode", type=str, default=None)


parser.add_argument("--num_replacement", type=int, default=1)
parser.add_argument("--synonym_path", type=str, default="synonym.txt")


parser.add_argument(
    "--query_set_size", type=int, default=2048, help="Number of samples to evaluate on"
)



parser.add_argument(
    "--eval_vqax",
    action="store_true",
    default=False,
    help="Whether to evaluate on VQA-X.",
)

parser.add_argument(
    "--eval_vizwiz",
    action="store_true",
    default=False,
    help="Whether to evaluate on VQA-X.",
)

parser.add_argument("--query_image_dir_path", type=str, default=None)
parser.add_argument("--query_questions_json_path", type=str, default=None)
parser.add_argument("--query_annotations_json_path", type=str, default=None)

parser.add_argument("--objects_prompt", type=str, default="Objects:")

parser.add_argument("--text_prompt", type=str, default="")


parser.add_argument(
    "--eval_tdiuc",
    action="store_true",
    default=False,
    help="Whether to evaluate on TDIUC.",
)

parser.add_argument("--output_log", type=str, default="output_log path")

parser.add_argument("--previous_predictions", type=str, default=None)
parser.add_argument("--previous_query_predictions", type=str, default=None)


parser.add_argument(
    "--eval_itm",
    action="store_true",
    default=False,
    help="Whether to evaluate on ITM.",
)
parser.add_argument("--compos", type=str, default="sys")
parser.add_argument("--neg_type", type=str, default="atom")

parser.add_argument("--image_dir_path", type=str, default=None)
parser.add_argument("--annotations_json_path", type=str, default=None)
parser.add_argument("--n_prod", type=int, default=4)


parser.add_argument("--cache_dir", type=str, default=None)

parser.add_argument("--max_n_objs", type=int, default=10)

parser.add_argument(
    "--skip_use_local_files",
    action="store_true",
    default=False,
    help="skip_use_local_files.",
)


parser.add_argument("--instruction", type=str, default='')


parser.add_argument("--num_beams", type=int, default=3)
parser.add_argument(
    "--do_sample",
    action="store_true",
    default=False,
    help="sampling.",
)
parser.add_argument("--gpu_margin", type=int, default=8)

### Instruction following 


parser.add_argument(
    "--testset",
    action="store_true",
    default=False,
    help="Whether to evaluate on a testset.",
)
parser.add_argument("--test_image_dir_path", type=str, default='')
parser.add_argument("--test_annotations_json_path", type=str, default='')

parser.add_argument("--prompt", type=str, default=None)


def main():
    args = parser.parse_args()

    accelerator = Accelerator()
    accelerator.print("Create model")
    # load model
    flamingo, processor, tokenizer = create_model_and_transforms(
        args.lm_path,
        args.lm_tokenizer_path,
        low_cpu=args.low_cpu, 
        use_local_files=not args.skip_use_local_files,
        gpu_margin=args.gpu_margin,
    )
    accelerator.print("Finish creating model")



    flamingo.requires_grad_(False)
    if args.precision == 'float16':
        print("cast to half")
        flamingo.half()

    if "80" not in args.lm_path:
        flamingo.to(args.device if args.device >= 0 else "cpu")

    results = defaultdict(list)

 


    

    if args.eval_flickr30:
        print("Evaluating on Flickr30...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(range(args.num_trials), range(args.num_trials)):
                
                cider_score = evaluate_coco_flickr(
                    model=flamingo,
                    tokenizer=tokenizer,
                    processor=processor,
                    batch_size=args.batch_size,
                    image_dir_path=args.flickr_image_dir_path,
                    annotations_json_path=args.flickr_annotations_json_path,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    device=args.device,
                    seed=seed,
                    is_flickr=True,
                    mode=args.mode,
                    args=args,
                    output_log=args.output_log,
                )
                print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
                scores.append(cider_score)
            print(f"Shots {shot} Mean CIDEr score: {np.mean(scores)}")
            results["flickr30"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)}
            )

    if args.eval_coco:
        print("Evaluating on COCO...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(range(args.num_trials), range(args.num_trials)):
                cider_score = evaluate_coco_flickr(
                    model=flamingo,
                    tokenizer=tokenizer,
                    processor=processor,
                    batch_size=args.batch_size,
                    image_dir_path=args.coco_image_dir_path,
                    annotations_json_path=args.coco_annotations_json_path,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    device=args.device,
                    seed=seed,
                    mode=args.mode,
                    eval_chair=True,
                    synonym_path=args.synonym_path,
                    args=args,
                    output_log=args.output_log,
                    query_set_size=args.query_set_size,
                )
                print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
                scores.append(cider_score)

            results = aggregate_results(scores, results, shot)




    if args.eval_itm:
        print("Evaluating on ITM...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(range(args.num_trials), range(args.num_trials)):
                print(args.num_trials, "trials")
                cider_score = evaluate_itm(
                    model=flamingo,
                    tokenizer=tokenizer,
                    processor=processor,
                    batch_size=args.batch_size,
                    image_dir_path=args.image_dir_path,
                    annotations_json_path=args.annotations_json_path,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    device=args.device,
                    seed=seed,
                    mode=args.mode,
                    args=args,
                    query_set_size=args.query_set_size,
                    output_log=args.output_log,
                    compos=args.compos,
                    neg_type=args.neg_type,
                    n_prod=args.n_prod,
                    text_prompt=args.text_prompt,
                    max_n_objs=args.max_n_objs,

                )
                print(f"Shots {shot} Trial {trial} Acc score: {cider_score}")
                scores.append(cider_score)

            results = aggregate_results(scores, results, shot)


    if args.eval_ok_vqa:
        print("Evaluating on OK-VQA...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(range(args.num_trials), range(args.num_trials)):
                ok_vqa_score = evaluate_vqa(
                    model=flamingo,
                    tokenizer=tokenizer,
                    processor=processor,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    device=args.device,
                    seed=seed,
                    image_dir_path=args.ok_vqa_image_dir_path,
                    questions_json_path=args.ok_vqa_questions_json_path,
                    annotations_json_path=args.ok_vqa_annotations_json_path,
                    vqa_dataset="ok_vqa",
                    query_set_size=args.query_set_size,
                    output_log=args.output_log
                )
                print(f"Shots {shot} Trial {trial} OK-VQA score: {ok_vqa_score}")
                scores.append(ok_vqa_score)
            print(f"Shots {shot} Mean OK-VQA score: {np.mean(scores)}")
            results["ok_vqa"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)}
            )

    if args.eval_vqav2:
        if args.mode is not None and 'caption' in args.mode:
            max_generation_length = 30
        else:
            max_generation_length = 5

        print("Evaluating on VQAv2...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(range(args.num_trials), range(args.num_trials)):
                vqa_score = evaluate_vqa(
                    model=flamingo,
                    tokenizer=tokenizer,
                    processor=processor,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    device=args.device,
                    seed=seed,
                    image_dir_path=args.vqav2_image_dir_path,
                    questions_json_path=args.vqav2_questions_json_path,
                    annotations_json_path=args.vqav2_annotations_json_path,
                    vqa_dataset="vqa",
                    mode=args.mode,
                    query_set_size=args.query_set_size,
                    max_generation_length=max_generation_length,
                    text_prompt=args.text_prompt,
                    output_log=args.output_log,
                    args=args
                )
                print(f"Shots {shot} Trial {trial} VQA score: {vqa_score}")
                scores.append(vqa_score)
                
            results = aggregate_results(scores, results, shot)

    if args.eval_vqax:
        print("Evaluating on VQA-X...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(range(args.num_trials), range(args.num_trials)):
                vqa_score = evaluate_vqa(
                    model=flamingo,
                    tokenizer=tokenizer,
                    processor=processor,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    device=args.device,
                    seed=seed,
                    image_dir_path=args.vqav2_image_dir_path,
                    questions_json_path=args.vqav2_questions_json_path,
                    annotations_json_path=args.vqav2_annotations_json_path,
                    vqa_dataset="vqax",
                    mode=args.mode,
                    max_generation_length=20,
                    query_set_size=args.query_set_size,
                    query_image_dir_path=args.query_image_dir_path,
                    query_questions_json_path=args.query_questions_json_path,
                    query_annotations_json_path=args.query_annotations_json_path,
                    text_prompt=args.text_prompt,
                    output_log=args.output_log,
                    args=args
                )
                print(f"Shots {shot} Trial {trial} VQA score: {vqa_score}")
                scores.append(vqa_score)
            results = aggregate_results(scores, results, shot)


    if args.eval_vizwiz:
        print("Evaluating on VizWiz...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(range(args.num_trials), range(args.num_trials)):
                vqa_score = evaluate_vqa(
                    model=flamingo,
                    tokenizer=tokenizer,
                    processor=processor,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    device=args.device,
                    seed=seed,
                    image_dir_path=args.vqav2_image_dir_path,
                    questions_json_path=args.vqav2_questions_json_path,
                    annotations_json_path=args.vqav2_annotations_json_path,
                    vqa_dataset="vizwiz",
                    mode=args.mode,
                    max_generation_length=20,
                    query_set_size=args.query_set_size,
                    query_image_dir_path=args.query_image_dir_path,
                    query_questions_json_path=args.query_questions_json_path,
                    query_annotations_json_path=args.query_annotations_json_path,
                    text_prompt=args.text_prompt,
                    output_log=args.output_log,
                    previous_predictions=args.previous_predictions,
                    args=args
                )
                print(f"Shots {shot} Trial {trial} VQA score: {vqa_score}")
                scores.append(vqa_score)
            print(f"Shots {shot} Mean VQA score: {np.mean(scores)}")
            results["vqax"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)}
            )

    if args.eval_tdiuc:
        print("Evaluating on TDIUC...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(range(args.num_trials), range(args.num_trials)):
                vqa_score = evaluate_vqa(
                    model=flamingo,
                    tokenizer=tokenizer,
                    processor=processor,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    device=args.device,
                    seed=seed,
                    image_dir_path=args.vqav2_image_dir_path,
                    questions_json_path=args.vqav2_questions_json_path,
                    annotations_json_path=args.vqav2_annotations_json_path,
                    vqa_dataset="tdiuc",
                    mode=args.mode,
                    max_generation_length=20,
                    query_set_size=args.query_set_size,
                    query_image_dir_path=args.query_image_dir_path,
                    query_questions_json_path=args.query_questions_json_path,
                    query_annotations_json_path=args.query_annotations_json_path,
                    text_prompt=args.text_prompt,
                    output_log=args.output_log,
                    previous_predictions=args.previous_predictions,
                    args=args
                )
                print(f"Shots {shot} Trial {trial} VQA score: {vqa_score}")
                scores.append(vqa_score)
            results = aggregate_results(scores, results, shot)


    if args.eval_imagenet:
        print("Evaluating on ImageNet...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(range(args.num_trials), range(args.num_trials)):
                imagenet_score = evaluate_imagenet(
                    model=flamingo,
                    tokenizer=tokenizer,
                    processor=processor,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    device=args.device,
                    seed=seed,
                    imagenet_root=args.imagenet_root,
                )
                print(
                    f"Shots {shot} Trial {trial} " f"ImageNet score: {imagenet_score}"
                )
                scores.append(imagenet_score)
            print(f"Shots {shot} Mean ImageNet score: {np.mean(scores)}")
            results["imagenet"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)}
            )

    save_dir = os.path.dirname(args.results_file)
    os.makedirs(save_dir, exist_ok=True)
    print("final results: ", results)
    results_serializable = {key: list(value) for key, value in results.items()}
    if args.results_file is not None:
        with open(args.results_file, "w") as f:
            json.dump(results_serializable, f)





def evaluate_coco_flickr(
    model,
    tokenizer,
    processor,
    batch_size,
    image_dir_path,
    annotations_json_path,
    seed=42,
    max_generation_length=20,
    num_beams=3,
    length_penalty=-2.0,
    num_samples=5000,
    query_set_size=2048,
    num_shots=8,
    device=-1,
    is_flickr=False,
    mode=None,
    eval_chair=False,
    synonym_path=None,
    args=None,
    output_log='/data/mshukor/logs/open_flamingo',
    is_llava=False,
    skip_eval=False,
):
    """Evaluate a model on COCO dataset.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor : image processor for the model
        batch_size (int): batch size
        image_dir_path (str, optional): path to the directory containing the images.
        annotations_json_path (str, optional): path to the json file containing the annotations.
        seed (int, optional): seed for random number generator. Defaults to 42.
        max_generation_length (int, optional): maximum length of the generated caption. Defaults to 10.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_samples (int, optional): number of samples to evaluate on. Defaults to 5000.
        query_set_size (int, optional): number of samples to use for query set. Defaults to 2048.
        num_shots (int, optional): number of in-context samples to use. Defaults to 8.
        device (int, optional): device to use. Defaults to -1.
        num_workers (int, optional): number of workers to use for dataloader. Defaults to 4.
        is_flickr (bool): defines if that data is COCO or Flickr. Defaults to False (COCO).

    Returns:
        float: CIDEr score

    """
    do_sample=args.do_sample
    num_beams=args.num_beams

    instruction=args.instruction
    if mode is not None and "llava" in mode:
        is_llava = True 
        skip_eval = True
        llava_predictions = []
        
    if "length" in mode:
        long_captions = True if "long" in mode else False
    else:
        long_captions = None

    if args.test_annotations_json_path and not args.testset:
        annotations_json_path_ = args.test_annotations_json_path
    else:
        annotations_json_path_ = annotations_json_path

    if args.testset:

        test_dataset = COCOFlickrDataset(
            image_dir_path=args.test_image_dir_path,
            annotations_path=args.test_annotations_json_path,
            is_flickr=is_flickr,
            previous_query_predictions=args.previous_query_predictions,
            is_llava=is_llava,
            seed=seed,
            long_captions=long_captions
        )



    full_dataset = COCOFlickrDataset(
        image_dir_path=image_dir_path,
        annotations_path=annotations_json_path_,
        is_flickr=is_flickr,
        previous_query_predictions=args.previous_query_predictions,
        is_llava=is_llava,
        seed=seed,
        long_captions=long_captions
    )
    effective_num_shots = num_shots if num_shots > 0 else 2
    random_indices = get_random_indices(num_samples, query_set_size, full_dataset, seed)



    in_context_samples, eval_dataset = prepare_eval_samples_and_dataset(
        full_dataset=full_dataset,
        random_indices=random_indices,
        query_set_size=query_set_size,
        mode=mode
    )




    model.eval()

    from open_flamingo.eval import caption_utils

    


    
    predictions = defaultdict()

    desc = "Running inference Flickr30" if is_flickr else "Running inference COCO"
    print("mode", mode)

    if "evalquery" in mode:
        eval_dataset = in_context_samples
    
    if args.testset:
        eval_dataset = test_dataset

    print(len(eval_dataset), len(in_context_samples))

    for batch in more_itertools.chunked(tqdm(eval_dataset, desc=desc), batch_size):
        batch_demo_samples = sample_batch_demos_from_query_set(
            in_context_samples, effective_num_shots, len(batch), mode=mode
        )


        if mode is None:
        
            context_text = [
                get_context(
                    caption_utils.get_prompt,
                    in_context_samples=batch_demo_samples[i],
                    effective_num_shots=effective_num_shots,
                    mode=mode,
                    num_shots=num_shots, instruction=instruction, instruct_model="user_assistant" in args.mode
                )
                for i in range(len(batch))
            ]
            batch_text = [context_text[i] + [batch[i]["image"], f"Output:"] for i in range(len(batch))]


            postprocess_generation = caption_utils.postprocess_generation

        elif 'caption_objects' in mode:
            max_generation_length = 45

            objects_prompt=args.objects_prompt
            

            test_prompt = "Caption:"
            get_prompt = caption_utils.get_prompt_caption_objects
            postprocess_generation = caption_utils.postprocess_generation_caption_objects

            objects_first = False

                

            get_prompt_caption_objects_ = partial(get_prompt, objects_first=objects_first, objects_prompt=objects_prompt)
            postprocess_generation_ = partial(postprocess_generation, objects_first=objects_first, objects_prompt=objects_prompt)



            context_text = [
                get_context(
                    get_prompt_caption_objects_,
                    in_context_samples=batch_demo_samples[i],
                    effective_num_shots=effective_num_shots,
                    num_shots=num_shots,
                    neg_samples=in_context_samples,
                    mode=mode, instruction=instruction, instruct_model="user_assistant" in args.mode
                )
                for i in range(len(batch))
            ]

            batch_text = [context_text[i] + [batch[i]["image"], f"{test_prompt}"] for i in range(len(batch))]


            postprocess_generation = postprocess_generation_



        else:
        
            get_prompt = partial(caption_utils.get_prompt, prompt=args.prompt)
            
            context_text = [
                get_context(
                    get_prompt,
                    in_context_samples=batch_demo_samples[i],
                    effective_num_shots=effective_num_shots,
                    num_shots=num_shots,
                    mode=mode, instruction=instruction, instruct_model="user_assistant" in args.mode
                )
                for i in range(len(batch))
            ]

            batch_text = [context_text[i] + [s["image"]] + split_user_assistant(f"{args.prompt}", instruct_model="user_assistant" in args.mode, train=False, merge="merge" in mode) for i, s in enumerate(batch)]

            postprocess_generation = caption_utils.postprocess_generation
        


        
        inputs = processor(batch_text, return_tensors="pt").to(device)

        bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        inputs.update({"bad_words_ids": bad_words_ids})
        if "-instruct" in args.lm_path:
            exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
            inputs.update({"eos_token_id": exit_condition})


        if is_llava:
            max_generation_length = 500

        outputs = get_outputs(
            model=model,
            inputs=inputs,
            device=device,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            do_sample=do_sample
        )

        new_predictions = [
            postprocess_generation(out).replace('"', "").replace("\n", "")
            for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]


        new_predictions_raw = [out for out in tokenizer.batch_decode(outputs, skip_special_tokens=True) ]

        for i, sample in enumerate(batch):
            predictions[sample["image_id"]] = {
                "caption": new_predictions[i], "raw_text": new_predictions_raw[i]
            }


        if is_llava:

            new_predictions_raw = [out for out in tokenizer.batch_decode(outputs, skip_special_tokens=True) ]
            
            llava_predictions.extend(
            [
                {"text": p, "question_id": sample["question_id"], "answer_id": shortuuid.uuid(), "model_id": "OF", "metadata": {}, "prompt": sample["question"], "raw_text": raw_p}
                for p, sample, raw_p in zip(new_predictions, batch, new_predictions_raw)
            ]
        )
            
    # foe debugging
    print(batch, batch_text, new_predictions, tokenizer.batch_decode(outputs, skip_special_tokens=True))

    random_uuid = str(seed) #str(uuid.uuid4())
    os.makedirs(output_log, exist_ok=True)
    results_path = (
        f"flickrresults_{random_uuid}.json"
        if is_flickr
        else f"cocoresults_{random_uuid}.json"
    )
    results_path = os.path.join(output_log, results_path)
    with open(results_path, "w") as f:
        f.write(
            json.dumps(
                [
                    {"image_id": k, "caption": predictions[k]["caption"], "raw_text": predictions[k]["raw_text"]}
                    for k in predictions
                ],
                indent=4,
            )
        )
    if is_llava:
        
        results_path_llava = os.path.join(output_log, f"llava_results_{random_uuid}.json")
        with open(results_path_llava, "w") as f:
            f.write(json.dumps(llava_predictions, indent=4))

    if skip_eval:
        return 0 * 100.0
    
    metrics = compute_cider(
        result_path=results_path,
        annotations_path=annotations_json_path,
    )

    all_results = {"CIDEr": metrics["CIDEr"] * 100.0, }
    if eval_chair:
        from open_flamingo.eval.metrics.chair import load_generated_captions, CHAIR

        results_list = [{"image_id": k, "caption": predictions[k]["caption"]} for k in predictions]

        
        _, imids = load_generated_captions(results_list)

        coco_ann_path = '/'.join(annotations_json_path.split('/')[:-1])

        evaluator = CHAIR(imids, coco_ann_path, synonym_path=synonym_path) 
        try:
            evaluator.get_annotations()
            cap_dict = evaluator.compute_chair(results_list) 
            for k, v in cap_dict['overall_metrics'].items():
                all_results.update({k: v})
        except:
            print("Skip chair eval")
            pass

    print("save results to:", output_log, )

    return all_results


def evaluate_itm(
    model,
    tokenizer,
    processor,
    batch_size,
    image_dir_path,
    annotations_json_path,
    seed=42,
    max_generation_length=20,
    num_beams=3,
    length_penalty=-2.0,
    num_samples=5000,
    query_set_size=2048,
    num_shots=8,
    device=-1,
    is_flickr=False,
    mode=None,
    args=None,
    output_log='/data/mshukor/logs/open_flamingo',
    compos='sys',
    neg_type='atom',
    n_prod=4,
    text_prompt='',
    max_n_objs=10,
):
    """Evaluate a model on COCO dataset.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor : image processor for the model
        batch_size (int): batch size
        image_dir_path (str, optional): path to the directory containing the images.
        annotations_json_path (str, optional): path to the json file containing the annotations.
        seed (int, optional): seed for random number generator. Defaults to 42.
        max_generation_length (int, optional): maximum length of the generated caption. Defaults to 10.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_samples (int, optional): number of samples to evaluate on. Defaults to 5000.
        query_set_size (int, optional): number of samples to use for query set. Defaults to 2048.
        num_shots (int, optional): number of in-context samples to use. Defaults to 8.
        device (int, optional): device to use. Defaults to -1.
        num_workers (int, optional): number of workers to use for dataloader. Defaults to 4.
        is_flickr (bool): defines if that data is COCO or Flickr. Defaults to False (COCO).

    Returns:
        float: CIDEr score

    """

    instruction=args.instruction
    full_dataset = ITMDataset(
        image_dir_path=image_dir_path,
        annotations_path=annotations_json_path,
        compos=compos,
        neg_type=neg_type,
        n_prod=n_prod
    )
    effective_num_shots = num_shots if num_shots > 0 else 2
    random_indices = get_random_indices(num_samples, query_set_size, full_dataset, seed)



    in_context_samples, eval_dataset = prepare_eval_samples_and_dataset(
        full_dataset=full_dataset,
        random_indices=random_indices,
        query_set_size=query_set_size,
        mode=mode
    )




    model.eval()

    from open_flamingo.eval import itm_utils


    
    predictions = defaultdict()

    desc = "Running inference Flickr30" if is_flickr else "Running inference COCO"
    print("mode", mode)


    print(len(eval_dataset), len(in_context_samples))
    for batch in more_itertools.chunked(tqdm(eval_dataset, desc=desc), batch_size):
        batch_demo_samples = sample_batch_demos_from_query_set(
            in_context_samples, effective_num_shots, len(batch), mode=mode
        )

 


        if 'itm' in mode and "objects" in mode:

            
            use_objects= True if 'objects' in mode else False
            prompt = text_prompt
            max_n_objs = max_n_objs

            prompt_function = partial(itm_utils.get_prompt_itm_objects, use_objects=use_objects, 
                                    prompt=prompt, max_n_objs=max_n_objs)


            context_text = [
                get_context(
                    prompt_function,
                    in_context_samples=batch_demo_samples[i],
                    effective_num_shots=effective_num_shots,
                    mode=mode,
                    num_shots=num_shots, instruction=instruction, instruct_model="user_assistant" in args.mode
                )
                for i in range(len(batch))
            ]

            labels = ['yes' if random.random() > 0.5 else 'no' for i in range(len(batch))]
            batch_text = [context_text[i] + [s["image"]] + split_user_assistant(prompt_function(s, train=False, label=l), instruct_model="user_assistant" in args.mode, train=False, merge="merge" in mode) for i, (s, l) in enumerate(zip(batch, labels))]

            post_prompt = prompt.split(" ")[0]
            postprocess_generation = partial(itm_utils.postprocess_generation_itm_objects, prompt=post_prompt)

        elif 'itm' in mode:

            prompt_function = itm_utils.get_prompt_itm


            context_text = [
                get_context(
                    prompt_function,
                    in_context_samples=batch_demo_samples[i],
                    mode=mode,
                    effective_num_shots=effective_num_shots,
                    num_shots=num_shots, instruction=instruction, instruct_model="user_assistant" in args.mode
                )
                for i in range(len(batch))
            ]

            labels = ['yes' if random.random() > 0.5 else 'no' for i in range(len(batch))]
            batch_text = [context_text[i] + [s["image"]] + split_user_assistant(prompt_function(s, train=False, label=l), instruct_model="user_assistant" in args.mode, train=False, merge="merge" in mode) for i, (s, l) in enumerate(zip(batch, labels))]


            postprocess_generation = itm_utils.postprocess_generation_itm

        else:
        
            context_text = [
                get_context(
                    itm_utils.get_prompt,
                    in_context_samples=batch_demo_samples[i],
                    effective_num_shots=effective_num_shots,
                    mode=mode,
                    num_shots=num_shots, instruction=instruction, instruct_model="user_assistant" in args.mode
                )
                for i in range(len(batch))
            ]

            labels = ['a' if random.random() > 0.5 else 'b' for i in range(len(batch))]
            batch_text = [context_text[i] + [s["image"]] + split_user_assistant(itm_utils.get_prompt(s, train=False, label=l), instruct_model="user_assistant" in args.mode, train=False, merge="merge" in mode) for i, (s, l) in enumerate(zip(batch, labels))]


            postprocess_generation = itm_utils.postprocess_generation


    

        

        inputs = processor(batch_text, return_tensors="pt").to(device)

        bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        inputs.update({"bad_words_ids": bad_words_ids})
        if "-instruct" in args.lm_path:
            exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
            inputs.update({"eos_token_id": exit_condition})




        outputs = get_outputs(
            model=model,
            inputs=inputs,
            device=device,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

        new_predictions = [
            postprocess_generation(out).replace("\n", "").replace('"', "").replace(")", "").replace(".", "").strip()
            for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]

        new_predictions_raw = [out for out in tokenizer.batch_decode(outputs, skip_special_tokens=True) ]

        for i, sample in enumerate(batch):
            predictions[sample["image_id"]] = {
                "answer": new_predictions[i].strip(), "label": labels[i].strip(),
                "pos": sample['pos_caption'], "neg": sample['neg_caption'], 'raw_text': new_predictions_raw[i]
            }

    print(batch, batch_text, new_predictions, tokenizer.batch_decode(outputs, skip_special_tokens=True), labels)
    # save the predictions to a temporary file
    random_uuid = str(seed) #str(uuid.uuid4())
    os.makedirs(output_log, exist_ok=True)
    results_path = (
        f"itmresults_{random_uuid}.json"
    )
    results_path = os.path.join(output_log, results_path)
    with open(results_path, "w") as f:
        f.write(
            json.dumps(
                [
                    {"image_id": k, "answer": predictions[k]["answer"].strip(), "label": predictions[k]["label"].strip(), 
                     "raw_text": predictions[k]["raw_text"], "pos": predictions[k]["pos"], "neg": predictions[k]["neg"]}
                    for k in predictions
                ],
                indent=4,
            )
        )

    count = 0
    for p in predictions.values():
        if p['answer'] == p['label']:
            count+=1

    acc = count/len(predictions)

    return acc * 100.0


def evaluate_vqa(
    model,
    tokenizer,
    processor,
    batch_size,
    image_dir_path,
    annotations_json_path,
    questions_json_path=None,
    seed=42,
    max_generation_length=5,
    num_beams=3,
    length_penalty=-2.0,
    num_samples=5000,
    query_set_size=2048,
    num_shots=8,
    device=-1,
    vqa_dataset="vqa",
    mode=None,
    query_image_dir_path=None,
    query_questions_json_path=None,
    query_annotations_json_path=None,
    text_prompt='',
    output_log='/data/mshukor/logs/open_flamingo',
    previous_predictions=None,
    args=None,
    is_llava=False,
    skip_eval=False,
):
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.0.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor : image processor for the model
        batch_size (int): batch size
        image_dir_path (str): path to image directory
        questions_json_path (str): path to questions json file
        annotations_json_path (str): path to annotations json file
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_samples (int, optional): number of samples to evaluate on. Defaults to 5000 samples.
        query_set_size (int, optional): size of the query set. Defaults to 2048.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        device (int, optional): device to use. Defaults to -1 (cpu).
        num_workers (int, optional): number of workers to use. Defaults to 4.
        vqa_dataset (string): type of vqa dataset: currently supports vqa, ok_vqa. Defaults to vqa.
    Returns:
        float: accuracy score
    """
    do_sample= args.do_sample
    num_beams = args.num_beams
    instruction=args.instruction
    
    
    if "llava" in mode:
        is_llava = True 
        skip_eval = True 
        vqa_dataset = 'llava'
        llava_predictions = []

    if args.testset:
        test_dataset = VQADataset(
            image_dir_path=args.test_image_dir_path,
            question_path=questions_json_path,
            annotations_path=args.test_annotations_json_path,
            vqa_dataset=vqa_dataset,
            previous_predictions=previous_predictions,
            previous_query_predictions=args.previous_query_predictions,
            is_llava=is_llava,
            seed=seed,
            args=args,
        )

    full_dataset = VQADataset(
        image_dir_path=image_dir_path,
        question_path=questions_json_path,
        annotations_path=annotations_json_path,
        vqa_dataset=vqa_dataset,
        previous_predictions=previous_predictions,
        previous_query_predictions=args.previous_query_predictions,
        is_llava=is_llava,
        seed=seed,
        args=args
    )
    query_dataset = None
    if query_image_dir_path is not None:

        query_dataset = VQADataset(
            image_dir_path=query_image_dir_path,
            question_path=query_questions_json_path,
            annotations_path=query_annotations_json_path,
            vqa_dataset=vqa_dataset,
            args=args,
        )

    effective_num_shots = num_shots if num_shots > 0 else 2

    if num_samples + effective_num_shots > len(full_dataset):

        raise ValueError(
            f"num_samples {num_samples} + num_shots {effective_num_shots} must be less than or equal to {len(full_dataset)}"
        )

    random_indices = get_random_indices(num_samples, query_set_size, full_dataset, seed, query_dataset=query_dataset)

    query_indices=None
    if query_dataset is not None:
        query_indices = random_indices[1]
        random_indices = random_indices[0]

    

    in_context_samples, eval_dataset = prepare_eval_samples_and_dataset(
        full_dataset=full_dataset,
        random_indices=random_indices,
        query_set_size=query_set_size,
        query_indices=query_indices,
        query_dataset=query_dataset,
        mode=mode
    )

    model.eval()
    predictions = []

    if vqa_dataset == 'vqax' or (mode is not None and 'caption' in mode):
        import language_evaluation
        evaluator = language_evaluation.CocoEvaluator(verbose=False)
    predicted_exp = []
    ref_exp = []
    batches = []
    all_predictions = []
    ref_unans = []

    postprocess_generation_exp = None 
    postprocess_generation_unans = None

    print("mode: ", mode, "using text prompt:", text_prompt)
    count_corrected = 0
    if "evalquery" in mode:
        eval_dataset = in_context_samples

    if args.testset:
        eval_dataset = test_dataset

    print(len(eval_dataset), len(in_context_samples))
    for batch in more_itertools.chunked(
        tqdm(eval_dataset, desc="Running inference"), batch_size
    ):
        batch_demo_samples = sample_batch_demos_from_query_set(
            in_context_samples, effective_num_shots, len(batch), mode=mode
        )



        if mode is not None:


            if 'explain' in mode:

                prevqout = True if "prevqout" in mode else False
                contrastive = True if "contrastive" in mode else False
                

                bad_prompt="Bad Explanation:"
                good_prompt="Good Explanation:"


                if 'explainonly' in mode:
                    exp_prompt = 'Explanation:'
                    
                    get_prompt_explain_ = partial(vqa_utils.get_prompt_explain, onlyexplain=True, exp_prompt=exp_prompt, contrastive=contrastive, prevqout=prevqout, 
                                                      bad_prompt=bad_prompt, good_prompt=good_prompt)
                    
                    post_exp_prompt = exp_prompt if not contrastive else bad_prompt
                    postprocess_generation = partial(vqa_utils.postprocess_generation_vqa_exponly, exp_prompt=post_exp_prompt)
                    bad_prompt="Bad"
                    good_prompt="Good"
                    postprocess_generation_exp = partial(vqa_utils.postprocess_expgeneration_vqa_exponly, bad_prompt=bad_prompt, good_prompt=good_prompt)

                    
                else:


                    exp_prompt = ' because '  # '. Explanation:'
                    # get_prompt_explain_ = partial(get_prompt_explain, exp_prompt=exp_prompt)

                    get_prompt_explain_ = partial(vqa_utils.get_prompt_explain,  exp_prompt=exp_prompt, 
                                                      contrastive=contrastive, prevqout=prevqout, 
                                                      bad_prompt=bad_prompt, good_prompt=good_prompt)
                    
                    post_exp_prompt = exp_prompt if not contrastive else good_prompt
                    postprocess_generation = partial(vqa_utils.postprocess_generation_vqa_explain, exp_prompt=post_exp_prompt)
                    post_exp_prompt = exp_prompt if not contrastive else good_prompt
                    postprocess_generation_exp = partial(vqa_utils.postprocess_expgeneration_vqa_explain, exp_prompt=post_exp_prompt, 
                                                         contrastive=contrastive, good_prompt=good_prompt, bad_prompt=bad_prompt)

                context_text = [
                    get_context(
                        get_prompt_explain_,
                        in_context_samples=batch_demo_samples[i],
                        effective_num_shots=effective_num_shots,
                        num_shots=num_shots,
                        neg_samples=in_context_samples,
                        mode=mode,
                        instruction=instruction,
                        instruct_model="user_assistant" in args.mode
                    )
                    for i in range(len(batch))
                ]

                batch_text = [
                     context_text[i] + [s["image"]] + split_user_assistant( get_prompt_explain_(s, train=False), instruct_model="user_assistant" in args.mode, train=False, merge="merge" in mode) for i, s in enumerate(batch)
                ]



            elif 'vqa_abstention' in mode:
                
                if 'abstentiononly' in mode:
                    prompt_first = False

                    prompt = 'Is it possible to answer the previous question based on the image?'

                    if "before" in mode:
                        prompt_first = True
                        prompt = 'Is it possible to answer the following question based on the image?'
                    get_prompt= partial(vqa_utils.get_prompt_vqa_abstention, onlyunans=True, prompt=prompt, prompt_first=prompt_first)
                    postprocess_generation = partial(vqa_utils.postprocess_generation_vqa_abstention, prompt=prompt)
                    postprocess_generation_unans = partial(vqa_utils.postprocess_unansgeneration_vqa_abstention, prompt=prompt, 
                    onlyunans=True) 

                else:
                    prompt = 'Does the previous question describe the image?Answer:'
                    get_prompt = partial(vqa_utils.get_prompt_vqa_abstention, prompt=prompt)

                    prompt = 'Does the previous'
                    postprocess_generation = partial(vqa_utils.postprocess_generation_vqa_abstention, prompt=prompt)

                    prompt = 'image?Answer:'
                    postprocess_generation_unans = partial(vqa_utils.postprocess_unansgeneration_vqa_abstention, prompt=prompt) 

                context_text = [
                    get_context(
                        get_prompt,
                        in_context_samples=batch_demo_samples[i],
                        effective_num_shots=effective_num_shots,
                        mode=mode,
                        num_shots=num_shots, instruction=instruction, instruct_model="user_assistant" in args.mode
                    )
                    for i in range(len(batch))
                ]

                batch_text = [
                     context_text[i] + [s["image"]] + split_user_assistant( get_prompt(s, train=False), instruct_model="user_assistant" in args.mode, train=False) for i, s in enumerate(batch)
                ]


            else:
                get_prompt = partial(vqa_utils.get_prompt, long_ans=is_llava)
                context_text = [
                    get_context(
                        get_prompt,
                        in_context_samples=batch_demo_samples[i],
                        effective_num_shots=effective_num_shots,
                        mode=mode,
                        num_shots=num_shots, mode=mode, instruction=instruction, instruct_model="user_assistant" in args.mode
                    )
                    for i in range(len(batch))
                ]

                if num_shots == 1:
                    context_text = [[] for i, s in enumerate(batch)]
                batch_text = [
                     context_text[i] + [s["image"]] + split_user_assistant(get_prompt(s, train=False), instruct_model="user_assistant" in args.mode, train=False, merge="merge" in mode) for i, s in enumerate(batch)
                ]

                
                postprocess_generation = partial(vqa_utils.postprocess_generation_vqa, long_ans=is_llava)

        else:
            get_prompt = partial(vqa_utils.get_prompt, long_ans=is_llava)
            context_text = [
                get_context(
                    get_prompt,
                    in_context_samples=batch_demo_samples[i],
                    effective_num_shots=effective_num_shots,
                    num_shots=num_shots, mode=mode, instruction=instruction, instruct_model="user_assistant" in args.mode
                )
                for i in range(len(batch))
            ]

            if num_shots == 1:
                context_text = [[] for i, s in enumerate(batch)]
            batch_text = [
                context_text[i] + [s["image"]] + split_user_assistant(get_prompt(s, train=False), instruct_model="user_assistant" in args.mode, train=False, merge="merge" in mode) for i, s in enumerate(batch)
            ]


            postprocess_generation = partial(vqa_utils.postprocess_generation_vqa, long_ans=is_llava)

        



        inputs = processor(batch_text, return_tensors="pt").to(device)

        bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        inputs.update({"bad_words_ids": bad_words_ids})
        if "-instruct" in args.lm_path:
            exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
            inputs.update({"eos_token_id": exit_condition})


        if is_llava:
            max_generation_length = 500

        outputs = get_outputs(
            model=model,
            inputs=inputs,
            device=device,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            do_sample=do_sample
        )

        process_function = (
            postprocess_generation
            if vqa_dataset in ["vqa", "vqax", "vizwiz", "tdiuc", "llava"]
            else postprocess_ok_vqa_generation
        )

        new_predictions = [
            process_function(out).replace("\n", "")
            for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]

        if "correct" in mode:
            corrected_predictions = []

            
            for s, p in zip(batch, new_predictions):
                
                prev_unans_pred = s.get('prev_unans_pred', 'yes').strip() # re.findall("yes|no", s['prev_unans_pred'])[0]
                if prev_unans_pred in ["no", "absurd"]:
                    
                    if "tdiuc" in vqa_dataset:
                        p = "doesnotapply"
                        count_corrected+=1
                    else:
                        pass
                    
                corrected_predictions.append(p)
            
            new_predictions = corrected_predictions

        all_predictions.extend(new_predictions)

        new_predictions_raw = [out for out in tokenizer.batch_decode(outputs, skip_special_tokens=True) ]

        predictions.extend(
            [
                {"answer": p, "question_id": sample["question_id"], "raw_text": raw_p}
                for p, sample, raw_p in zip(new_predictions, batch, new_predictions_raw)
            ]
        )


        if is_llava:

            new_predictions_raw = [out for out in tokenizer.batch_decode(outputs, skip_special_tokens=True) ]
            
            llava_predictions.extend(
            [
                {"text": p, "question_id": sample["question_id"], "answer_id": shortuuid.uuid(), "model_id": "OF", "metadata": {}, "prompt": sample["question"],
                  "raw_text": raw_p, "image_id": sample['image_id']}
                for p, sample, raw_p in zip(new_predictions, batch, new_predictions_raw)
            ]
            )
        

        if postprocess_generation_exp:
            if 'explain' in mode:
                ref_exp.extend([b['explanation'] for b in batch])
            else:
                ref_exp.extend([b['captions'] for b in batch])

            predicted_exp.extend([
                postprocess_generation_exp(out).replace("\n", "")
                for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)])

        batches.extend([b for b in batch])
        ref_unans.extend(['no' if b['answer_type'] in ['unanswerable', "absurd", "doesnotapply"] else 'yes' for b in batch])

        if postprocess_generation_unans:
            ref_exp.extend(['no' if b['answer_type'] in ['unanswerable', "absurd"] else 'yes' for b in batch])

            predicted_exp.extend([
                postprocess_generation_unans(out).replace("\n", "").strip()
                for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)])
            
            
    
    print(batch, batch_text, tokenizer.batch_decode(outputs, skip_special_tokens=True), new_predictions)


    print("number of corrected answers:", count_corrected)
    # save the predictions to a temporary file
    random_uuid = str(seed) #str(uuid.uuid4())
    results_path = os.path.join(output_log, f"{vqa_dataset}_results_{random_uuid}.json")
    os.makedirs(output_log, exist_ok=True)
    with open(results_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))

    if is_llava:
        
        results_path_llava = os.path.join(output_log, f"llava_results_{random_uuid}.json")
        with open(results_path_llava, "w") as f:
            f.write(json.dumps(llava_predictions, indent=4))

    if skip_eval:
        return 0.
    
    acc = compute_vqa_accuracy(
        results_path,
        questions_json_path,
        annotations_json_path,
    )
    exp_results = None
    if postprocess_generation_exp:
        exp_results = evaluator.run_evaluation(predicted_exp, ref_exp)
        print(exp_results)

    print("save results to:", output_log, )
    if postprocess_generation_unans:
        count = 0
        unans = 0
        gt_unans = 0
        acc_preds = []
        acc_gts = []
        for r, p in zip(ref_exp, predicted_exp):
            if any([k in p for k in ['unanswerable', "unsuitable", "no", "absurd", "doesnotapply"]]):
                unans+=1
                post_p = "absurd"
            else:
                post_p = "valid"

            if any([k in r for k in ['unanswerable', "unsuitable", "no", "absurd", "doesnotapply"]]):
                gt_unans+=1
                post_r = "absurd" 
            else:
                post_r = "valid" 

            if post_r.strip() == post_p.strip():
                count+=1

            acc_preds.append(post_p)
            acc_gts.append(post_r)

        print("unanswerable accuracy:", count/len(ref_exp), "number of examples:", len(ref_exp), 
              "number of pred unans:", unans, "number of GT unans:", gt_unans)
        

    
        results = []

        results.extend(
            [
                {"img_path": s["img_path"], "image_id": s["image_id"], "question": s["question"], 
                 "question_id": s["question_id"], "answers": s["answers"], "prediction": p, "label": r,}
                for s, r, p in zip(batches, ref_exp, predicted_exp)
            ]
        )
        
        save_path_unans = os.path.join(output_log, f"{vqa_dataset}_unans_results_{str(seed)}.json")
        print("save results unans to:", save_path_unans)
        with open(save_path_unans, "w") as f:
            json.dump(results, f)

        if "tdiuc" in vqa_dataset and 'unansonly' in mode:
            ### Compute F1 score
            post_acc = count/len(predicted_exp)
            f1, precision, recall = f1_score(acc_preds, acc_gts, P_label='absurd', N_label='valid')
            return {"acc": acc, "acc": post_acc, "f1": f1, "precision": precision, "recall": recall}
        
    ## Absurd classif from pred
    post_count = 0
    post_gt_unans, post_unans = 0, 0
    ## post pred
    acc_preds = []
    acc_gts = []
    for b, p in zip(batches, all_predictions):
        p = p.replace(":", "").replace(".", "")

        r = b['answer_type']

        
        if any([k in p for k in ['unanswerable', "unsuitable", "doesnotapply", "absurd"]]):
            post_unans+=1
            post_p = "absurd"
        else:
            post_p = "valid"
        if any([k in r for k in ['unanswerable', "unsuitable",  "doesnotapply", "absurd"]]):
            post_gt_unans+=1
            post_r = "absurd" 
        else:
            post_r = "valid" 
        acc_preds.append(post_p)
        acc_gts.append(post_r)

        if post_p.strip() == post_r.strip():
            post_count+=1
    if len(all_predictions) > 0:
        post_acc = post_count/len(all_predictions)
        print("unanswerable post accuracy:", post_count/len(all_predictions), "number of examples:", len(all_predictions), 
                "number of pred unans:", post_unans, "number of GT unans:", post_gt_unans)
    if "tdiuc" in vqa_dataset:
        ### Compute F1 score
        f1, precision, recall = f1_score(acc_preds, acc_gts, P_label='absurd', N_label='valid')
        return {"acc": acc, "post_acc": post_acc, "f1": f1, "precision": precision, "recall": recall}


    if "vqax" in vqa_dataset:
        if exp_results is not None:
            return {"acc": acc, "CIDEr": exp_results['CIDEr'], "B4": exp_results['Bleu_4']}


    return acc


def evaluate_imagenet(
    model,
    tokenizer,
    processor,
    batch_size: int,
    imagenet_root: str,
    seed: int = 42,
    num_samples: int = 5000,
    num_shots: int = 8,
    device: int = -1,
):
    """
    Evaluate a model on ImageNet dataset.

    Args:
        model: model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor : image processor for the model
        batch_size (int): batch size
        imagenet_root (str): path to imagenet root for the specified split.
        seed (int, optional): random seed. Defaults to 42.
        num_samples (int, optional): number of samples to evaluate on. Defaults to 5000 samples.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        device (int, optional): device to use. Defaults to -1 (cpu).

    Returns:
        float: accuracy score
    """

    full_dataset = ImageNetDataset(root=imagenet_root)

    effective_num_shots = num_shots if num_shots > 0 else 2

    if num_samples + effective_num_shots > len(full_dataset):
        raise ValueError(
            f"num_samples {num_samples} + num_shots {effective_num_shots} must be less than or equal to "
            f"{len(full_dataset)} "
        )

    random_indices = get_random_indices(
        num_samples, effective_num_shots, full_dataset, seed
    )

    eoc_token = "<|endofchunk|>"
    eoc_token_id = tokenizer.additional_special_tokens_ids[
        tokenizer.additional_special_tokens.index(eoc_token)
    ]

    # Padding from right allows efficient precomputing of context activations.
    tokenizer.padding_side = "right"

    def _imagenet_prompt(class_name, is_context: bool = True):
        """Construct an imagenet prompt for a given label."""
        prefix = " A photo of a "
        if is_context:
            return prefix + class_name.strip()
        else:
            # Not a context example; insert EOS token before the class name
            # so that we can compute the loss on the class name tokens only.
            return prefix + tokenizer.eos_token + class_name.strip()

    def get_imagenet_prompt(x: dict, is_context: bool = True) -> str:
        """Construct an ImageNet prompt for an example, using its label."""
        return _imagenet_prompt(x["class_name"], is_context=is_context)

    in_context_samples, eval_dataset = prepare_eval_samples_and_dataset(
        full_dataset=full_dataset,
        random_indices=random_indices,
        query_set_size=effective_num_shots,  # NOTE: here we replace query_set_size with effective_num_shots but this is not the ideal evaluation setting.
        # TODO: We should add a query_set_size argument to the function and use it to randomly sample the context for each example.
        # This will be more consistent with the evaluation setting in the paper but will require some reworking of the caching.
    )

    device = device if device >= 0 else "cpu"

    model.eval()
    # Predictions based on the class target sequence with the maximal
    # predicted probability
    predictions_max_prob = []
    # Predictions based on the class target sequence with the minimal loss on
    # the model logits
    predictions_min_loss = []
    labels = []

    # context_images = [
    #     get_context_images(
    #         processor=processor,
    #         in_context_samples=in_context_samples,
    #         num_shots=num_shots,
    #     )
    #     for _ in range(batch_size)
    # ]

    context_text = get_context(
        get_imagenet_prompt,
        in_context_samples=in_context_samples,
        effective_num_shots=effective_num_shots,
        num_shots=num_shots,
    )

    # kwargs to use when calling tokenizer
    tokenizer_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": 256,
    }

    for i, batch in enumerate(more_itertools.chunked(eval_dataset, batch_size)):
        print(f"processing batch {i} of {ceil(len(eval_dataset) / batch_size)}")
        batch_per_class_probs = []
        batch_per_class_losses = []
        # batch_images = prepare_batch_images(
        #     batch=batch,
        #     processor=processor,
        #     context_images=context_images,
        #     num_shots=num_shots,
        # )

        # Process the images only once.
        batch_images = batch_images.to(device)
        model._encode_vision_x(vision_x=batch_images)

        # Process the context text only once.
        context_encodings = tokenizer([context_text] * batch_size, **tokenizer_kwargs)
        context_ids = context_encodings["input_ids"].to(device)
        context_len = context_ids.shape[-1]
        context_precomputed = model(
            None,
            context_ids,
            use_cached_vision_x=True,
            clear_conditioned_layers=False,
            use_cache=True,
        )

        # For each ImageNet class, construct the output prompt, compute a
        # forward pass, and store the results.
        for imagenet_class_name in tqdm(openai_imagenet_classnames):
            batch_text = [
                context_text + _imagenet_prompt(imagenet_class_name, False) + eoc_token
            ] * batch_size

            full_batch_encodings = tokenizer(batch_text, **tokenizer_kwargs)

            # full_batch_input_ids has shape [batch_size, seq_len], but we
            # only need to run inference on the [batch_size,
            # context_len:] inputs that have not been precomputed and
            # vary per class.
            full_batch_input_ids = full_batch_encodings["input_ids"].to(device)
            full_batch_attention_mask = full_batch_encodings["attention_mask"].to(
                device
            )

            # Sanity check that the encoded inputs with context are the same
            # as the encoded context alone, for every example in the batch
            assert torch.all(
                context_ids[0, :] == full_batch_input_ids[:, :context_len]
            ).item()

            # Clone the nested structure of the past key values
            past_key_values = tuple(
                [
                    tuple([x.clone() for x in inner])
                    for inner in context_precomputed.past_key_values
                ]
            )

            # Compute the outputs without recomputing context representations.
            outputs = model(
                vision_x=None,
                lang_x=full_batch_input_ids[:, context_len:],
                attention_mask=full_batch_attention_mask,
                use_cached_vision_x=True,
                clear_conditioned_layers=False,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = torch.concat((context_precomputed.logits, outputs.logits), 1)

            per_sample_probs = compute_per_sample_probs(
                encodings=full_batch_encodings,
                tokenizer=tokenizer,
                logits=logits,
                eoc_token_id=eoc_token_id,
            )
            per_sample_loss = compute_per_sample_loss(
                encodings=full_batch_encodings,
                tokenizer=tokenizer,
                logits=logits,
                eoc_token_id=eoc_token_id,
            )
            batch_per_class_probs.append(per_sample_probs.detach())
            batch_per_class_losses.append(per_sample_loss.detach())

        # Tensor of shape [batch_size, 1000] where the [i,j]th element is
        # the (probability or loss) for batch element i on imagenet class j.
        batch_probs = torch.stack(batch_per_class_probs, 1)
        batch_losses = torch.stack(batch_per_class_losses, 1)

        predictions_max_prob.extend(torch.argmax(batch_probs, 1).detach().tolist())
        predictions_min_loss.extend(torch.argmin(batch_losses, 1).detach().tolist())
        labels.extend(x["class_id"] for x in batch)

    acc_max_prob = (np.array(predictions_max_prob) == np.array(labels)).mean()
    acc_min_loss = (np.array(predictions_min_loss) == np.array(labels)).mean()
    print(f"[DEBUG] ImageNet accuracy with max prob method is {acc_max_prob}")
    print(f"[DEBUG] ImageNet accuracy with min loss method is {acc_min_loss}")
    print(f"[DEBUG] printing ImageNet predictions and labels:")
    for yhat_prob, yhat_loss, y in zip(
        predictions_max_prob, predictions_min_loss, labels
    ):
        print(
            " " * 30 + f"label: {IMAGENET_1K_CLASS_ID_TO_LABEL[y]}"
            f"\nprediction (max prob method): "
            f"{IMAGENET_1K_CLASS_ID_TO_LABEL[yhat_prob]}"
            f"\nprediction (min loss method): "
            f"{IMAGENET_1K_CLASS_ID_TO_LABEL[yhat_loss]}\n"
            "#" * 25
        )
    return acc_max_prob


if __name__ == "__main__":
    main()
