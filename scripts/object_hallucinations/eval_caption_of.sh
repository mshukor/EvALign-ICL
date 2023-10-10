#!/bin/bash

export LC_ALL=C  
export TRANSFORMERS_CACHE=path_to/.cache/huggingface/transformers
export CACHE=path_to/.cache/clip

DATA='/gpfsDATA/rech/dyf/ugz83ue'
LOGS='/gpfsLOGS/rech/dyf/ugz83ue'


DEVICE="0"


## Models
LM_TOKENIZER_PATH="${LOGS}/pretrained_models/new/7B"
VISION_ENCODER_NAME="ViT-L-14"
VISION_ENCODER_PRETRAINED="openai"
DEVICE="0"

# CKPT_PATH="${LOGS}/pretrained_models/open_flamingo/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt"
# CROSS_ATT=1
# LM_TOKENIZER_PATH="anas-awadalla/mpt-1b-redpajama-200b"
# LM_PATH="${LOGS}/pretrained_models/open_flamingo/mpt-1b-redpajama-200b"
# MODEL="3B"

# CKPT_PATH="${LOGS}/pretrained_models/open_flamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct/checkpoint.pt"
# CROSS_ATT=1
# LM_TOKENIZER_PATH="anas-awadalla/mpt-1b-redpajama-200b-dolly"
# # LM_TOKENIZER_PATH="${LOGS}/.cache/huggingface/transformers/models--anas-awadalla--mpt-1b-redpajama-200b-dolly"
# LM_PATH="${LOGS}/pretrained_models/open_flamingo/mpt-1b-redpajama-200b-dolly"
# MODEL="3Binst"

# CKPT_PATH="${LOGS}/pretrained_models/open_flamingo/OpenFlamingo-4B-vitl-rpj3b/checkpoint.pt"
# CROSS_ATT=2
# LM_TOKENIZER_PATH="togethercomputer/RedPajama-INCITE-Base-3B-v1"
# LM_PATH="${LOGS}/pretrained_models/open_flamingo/RedPajama-INCITE-Base-3B-v1"
# MODEL="4B"

# CKPT_PATH="${LOGS}/pretrained_models/open_flamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct/checkpoint.pt"
# CROSS_ATT=2
# LM_TOKENIZER_PATH="togethercomputer/RedPajama-INCITE-Instruct-3B-v1"
# LM_PATH="${LOGS}/pretrained_models/open_flamingo/RedPajama-INCITE-Instruct-3B-v1"
# MODEL="4Binst"

CKPT_PATH="${LOGS}/pretrained_models/open_flamingo/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt"
CROSS_ATT=4
LM_TOKENIZER_PATH="anas-awadalla/mpt-7b"
LM_PATH="${LOGS}/pretrained_models/open_flamingo/mpt-7b"
MODEL="9Bmpt"

# CKPT_PATH="${LOGS}/pretrained_models/open_flamingo/OpenFlamingo-9B/checkpoint.pt"
# CROSS_ATT=4
# LM_TOKENIZER_PATH="${LOGS}/pretrained_models/new/7B"
# LM_PATH="${LOGS}/pretrained_models/new/7B"
# MODEL="9B"



## Dataset
dataset='coco'
split='val'
COCO_IMG_PATH="${DATA}/data/coco/${split}2014"
COCO_ANNO_PATH="${DATA}/data/coco/annotations/captions_${split}2014_objs_vinvl.json"
SYN_PATH="${DATA}/data/coco/hallucination/release/data/synonyms.txt"
NUM_SAMPLES=5000
QUERY_SIZE=2048



OBJECTS_PROMPT='There is only these objects:' # "The image contains only these objects:"
INSTRUCTION=''
PROMPT="Output:"
PREV_Q_OUT=''


## Modes
MODE="baseline" # ICL

# MODE="baseline_inst" # ICL + task instruction
# INSTRUCTION="Describe the following images, do not include any object not present in the image. Here is few illustration examples: "

# MODE="caption_objects" # MT-ICL




for shot in {0,4,8,16,32};do
    SAVE_DIR="${LOGS}/logs/open_flamingo/${dataset}/${PREFIX}_${MODE}/${MODEL}"
    OUTPUT_LOG="${SAVE_DIR}/${shot}"
    RESULTS_FILE="${SAVE_DIR}/results_${shot}.json"
    echo $RESULTS_FILE

    python -m accelerate.commands.launch --mixed_precision=fp16 --num_processes=1 --num_machines=1 open_flamingo/eval/eval_acc.py \
        --lm_path $LM_PATH \
        --lm_tokenizer_path $LM_TOKENIZER_PATH \
        --vision_encoder_path $VISION_ENCODER_NAME \
        --vision_encoder_pretrained $VISION_ENCODER_PRETRAINED \
        --checkpoint_path $CKPT_PATH \
        --cross_attn_every_n_layers $CROSS_ATT \
        --device $DEVICE \
        --coco_image_dir_path $COCO_IMG_PATH \
        --coco_annotations_json_path $COCO_ANNO_PATH \
        --results_file $RESULTS_FILE \
        --eval_coco \
        --num_samples 5000 \
        --shots $shot \
        --num_trials 3 \
        --low_cpu \
        --precision float16 \
        --batch_size 2 \
        --synonym_path $SYN_PATH \
        --objects_prompt "${OBJECTS_PROMPT}" \
        --output_log $OUTPUT_LOG \
        --mode $MODE \
        --cache_dir $CACHE \
        --instruction "$INSTRUCTION"

done


