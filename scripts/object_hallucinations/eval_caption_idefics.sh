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

# LM_PATH='HuggingFaceM4/idefics-80b'
# LM_TOKENIZER_PATH='HuggingFaceM4/idefics-80b'
# MODEL="IDEFICS80B"

# LM_PATH='HuggingFaceM4/idefics-80b-instruct'
# LM_TOKENIZER_PATH='HuggingFaceM4/idefics-80b-instruct'
# MODEL="IDEFICS80BINSTRUCT"



## Dataset
dataset='coco'
split='val'
COCO_IMG_PATH="${DATA}/data/coco/${split}2014"
COCO_ANNO_PATH="${DATA}/data/coco/annotations/captions_${split}2014_objs_vinvl.json"
SYN_PATH="${DATA}/data/coco/hallucination/release/data/synonyms.txt"
NUM_SAMPLES=5000
QUERY_SIZE=2048



OBJECTS_PROMPT='There is only these objects:' # "The image contains only these objects:"
DO_SAMPLE=""
NUM_BEAMS=3
INSTRUCTION=''
PROMPT="Output:"
PREV_Q_OUT=''






## Modes
# MODE="baseline" # ICL

# MODE="baseline_inst" # ICL + task instruction
# INSTRUCTION="Describe the following images, do not include any object not present in the image. Here is few illustration examples: "

# MODE="caption_objects" # MT-ICL



for shot in {0,4,8,16,32};do
    SAVE_DIR="${LOGS}/logs/open_flamingo/${dataset}/${MODE}/${MODEL}"
    OUTPUT_LOG="${SAVE_DIR}/${shot}"
    RESULTS_FILE="${SAVE_DIR}/results_${shot}.json"
    echo $RESULTS_FILE

    python -m accelerate.commands.launch --mixed_precision=bf16 --num_processes=1 --num_machines=1 open_flamingo/eval/eval_acc_idefics.py \
        --lm_path $LM_PATH \
        --lm_tokenizer_path $LM_TOKENIZER_PATH \
        --device $DEVICE \
        --coco_image_dir_path $COCO_IMG_PATH \
        --coco_annotations_json_path $COCO_ANNO_PATH \
        --results_file $RESULTS_FILE \
        --eval_coco \
        --num_samples $NUM_SAMPLES \
        --query_set_size $QUERY_SIZE \
        --shots $shot \
        --num_trials 3 \
        --low_cpu \
        --batch_size 2 \
        --synonym_path $SYN_PATH \
        --objects_prompt "${OBJECTS_PROMPT}" \
        --output_log $OUTPUT_LOG \
        --mode $MODE \
        --cache_dir $CACHE \
        --instruction "$INSTRUCTION" \
        ${PREV_Q_OUT} \
        ${DO_SAMPLE} \
        --num_beams $NUM_BEAMS \
        --prompt "${PROMPT}"

done




