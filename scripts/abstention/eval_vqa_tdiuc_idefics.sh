#!/bin/bash

export LC_ALL=C  
export TRANSFORMERS_CACHE=path_to/.cache/huggingface/transformers
export CACHE=path_to/.cache/clip

DATA='/gpfsDATA/rech/dyf/ugz83ue'
LOGS='/gpfsLOGS/rech/dyf/ugz83ue'


DEVICE="0"



## Models
# LM_PATH='HuggingFaceM4/idefics-9b'
# LM_TOKENIZER_PATH='HuggingFaceM4/idefics-9b'
# MODEL="IDEFICS9B"

LM_PATH='HuggingFaceM4/idefics-9b-instruct'
LM_TOKENIZER_PATH='HuggingFaceM4/idefics-9b-instruct'
MODEL="IDEFICS9BINSTRUCT"





## Dataset
dataset="tdiuc"
split='val'
VQAV2_IMG_PATH="${DATA}/data/tdiuc/images/${split}2014"
VQAV2_ANNO_PATH="${DATA}/data/tdiuc/TDIUC/Annotations/mscoco_${split}2014_annotations.json"
VQAV2_QUESTION_PATH="${DATA}/data/tdiuc/TDIUC/Questions/OpenEnded_mscoco_${split}2014_questions.json"

NUM_SAMPLE=8000
QUERY_SIZE=2048 # for ICL examples




INSTRUCTION=''
PREVIOUS_PREDS=''

# Modes
# MODE="baseline" # ICL

# MODE="baseline_inst" # ICL + task instruction
# INSTRUCTION="Answer the following questions about the image, give short answers, if you do not know the asnwer or the question is not relevant to the image say doesnotapply. Here is few illustration examples: "

# MODE="vqa_abstention" # MT-ICL


# MODE="vqa_abstentiononly" # SC-ICL classif (done first for simplicity) 

# MODE="vqa_abstentiononly_inst" # SC-ICL classif + task instruction
# INSTRUCTION="You need to decide whether the question in '' can be answered from the image or not. Give only yes or no answers. Here is few illustration examples: "


# MODE="correct_shot" # SC-ICL (same shot)
# MODE="correct_32shot" # SC-ICL (32shot)





for shot in {0,4,8,16,32};do
    # PREVIOUS_PREDS="${LOGS}/logs/open_flamingo/tdiuc/vqa_abstentiononly/${MODEL}/${shot}/tdiuc_unans_results_seed.json"


    SAVE_DIR="${LOGS}/logs/open_flamingo/${dataset}/${MODE}/${MODEL}"
    OUTPUT_LOG="${SAVE_DIR}/${shot}"
    RESULTS_FILE="${SAVE_DIR}/results_${shot}.json"
    echo $RESULTS_FILE
    python -m accelerate.commands.launch --mixed_precision=bf16 --num_processes=1 --num_machines=1 open_flamingo/eval/eval_acc_idefics.py \
        --lm_path $LM_PATH \
        --lm_tokenizer_path $LM_TOKENIZER_PATH \
        --device $DEVICE \
        --vqav2_image_dir_path $VQAV2_IMG_PATH \
        --vqav2_annotations_json_path $VQAV2_ANNO_PATH \
        --results_file $RESULTS_FILE \
        --eval_tdiuc \
        --num_samples $NUM_SAMPLE \
        --shots $shot \
        --num_trials 3 \
        --low_cpu \
        --batch_size 2 \
        --vqav2_questions_json_path $VQAV2_QUESTION_PATH \
        --query_set_size $QUERY_SIZE \
        --mode $MODE \
        --output_log $OUTPUT_LOG \
        --cache_dir $CACHE \
        --instruction "$INSTRUCTION" 
        # --previous_predictions $PREVIOUS_PREDS 
done
