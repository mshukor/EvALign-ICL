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
# CROSS_ATT=13B
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





dataset='vg'
split='val'
IMG_PATH="${DATA}/data/visual_genome/images"
ANNO_PATH="${DATA}/data/crepe/syst_vg_hard_negs_seen_compounds_in_laion.json"
COMPOS="sys" # prod sys normal (baseline)
NEG_TYPE="atom" #  sys: 'atom' 'comp' 'combined', prod: 'hard_negs', baseline "caption"
PREFIX=f"${COMPOS}_${NEG_TYPE}"
NUM_SAMPLES=5000
QUERY_SIZE=2048


MAX_N_OBJS=15
num_trials=3


INSTRUCTION=''

MODE="itm" # ITM

# MODE="itm_inst" # ITM + task instruction
# INSTRUCTION="You need to find if the provided sentences acurately describe the image, if the composition of the sentence does not match the image then the sentence does not describe the image. Here is few illustration examples: "

# MODE="itm_objects" # ITM MT-ICL

# MODE="its" # ITS

for shot in {0,4,8,16,32};do
    SAVE_DIR="${LOGS}/logs/open_flamingo/${dataset}/${PREFIX}_${MODE}/${MODEL}"
    OUTPUT_LOG="${SAVE_DIR}/${shot}"
    RESULTS_FILE="${SAVE_DIR}/results_${shot}.json"


    python -m accelerate.commands.launch --mixed_precision=fp16 --num_processes=1 --num_machines=1 open_flamingo/eval/eval_acc.py \
        --lm_path $LM_PATH \
        --lm_tokenizer_path $LM_TOKENIZER_PATH \
        --vision_encoder_path $VISION_ENCODER_NAME \
        --vision_encoder_pretrained $VISION_ENCODER_PRETRAINED \
        --checkpoint_path $CKPT_PATH \
        --cross_attn_every_n_layers $CROSS_ATT \
        --device $DEVICE \
        --image_dir_path $IMG_PATH \
        --annotations_json_path $ANNO_PATH \
        --results_file $RESULTS_FILE \
        --eval_itm \
        --num_samples $NUM_SAMPLES \
        --shots $shot \
        --num_trials $num_trials \
        --low_cpu \
        --precision float16 \
        --batch_size 2 \
        --mode $MODE \
        --compos $COMPOS \
        --neg_type $NEG_TYPE \
        --n_prod $N_PROD \
        --output_log $OUTPUT_LOG \
        --query_set_size $QUERY_SIZE \
        --cache_dir $CACHE \
        --text_prompt 'because there is only these objects:' \
        --max_n_objs $MAX_N_OBJS \
        --instruction "$INSTRUCTION"

done
