#!/bin/bash

export LC_ALL=C  
export TRANSFORMERS_CACHE=path_to/.cache/huggingface/transformers
export CACHE=path_to/.cache/clip

DATA='/gpfsDATA/rech/dyf/ugz83ue'
LOGS='/gpfsLOGS/rech/dyf/ugz83ue'


DEVICE="0"



## Models
VISION_ENCODER_NAME="ViT-L-14"
VISION_ENCODER_PRETRAINED="openai"



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

# CKPT_PATH="${LOGS}/pretrained_models/open_flamingo/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt"
# CROSS_ATT=4
# LM_TOKENIZER_PATH="anas-awadalla/mpt-7b"
# LM_PATH="${LOGS}/pretrained_models/open_flamingo/mpt-7b"
# MODEL="9Bmpt"

CKPT_PATH="${LOGS}/pretrained_models/open_flamingo/OpenFlamingo-9B/checkpoint.pt"
CROSS_ATT=4
LM_TOKENIZER_PATH="${LOGS}/pretrained_models/new/7B"
LM_PATH="${LOGS}/pretrained_models/new/7B"
MODEL="9B"



## Dataset
dataset="vqax"
VQAV2_IMG_PATH="${SCRATCH}/data/coco/val2014"
VQAV2_ANNO_PATH="${SCRATCH}/data/XAI/Multimodal_Explanation_Dataset/VQA_X/vqaxtest_v2_mscoco_val2014_annotations.json"
split='test'
NUM_SAMPLE=1400
QUERY_SIZE=500 # for ICL examples


INSTRUCTION=''
PREV_Q_OUT=""


## Modes
# MODE='baseline'  # ICL

MODE="explainonly" # ICL explain only

# MODE="explainonly_inst" # ICL explain only + task instruction
# instruction="You will be given a question and answer, you need to give an explanation of the given answer based on the image. Here is few illustration examples: "
 
# MODE="answer_and_explain" # MT-ICL

# MODE="explainonly_evalquery" # ICL explain only for the query data

# MODE="explainonly_contrastive_prevqout_32shotvsgt" # CoH-ICL
# PREV_Q_OUT="--previous_query_predictions ${LOGS}/logs/open_flamingo/${dataset}/coh_randswap_explainonly_evalquery/${MODEL}/32/vqax_results_seed.json"


for shot in {0,4,8,16,32};do
    SAVE_DIR="${LOGS}/logs/open_flamingo/${dataset}/${MODE}/${MODEL}"
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
        --vqav2_image_dir_path $VQAV2_IMG_PATH \
        --vqav2_annotations_json_path $VQAV2_ANNO_PATH \
        --results_file $RESULTS_FILE \
        --eval_vqax \
        --num_samples $NUM_SAMPLE \
        --shots $shot \
        --num_trials 3 \
        --low_cpu \
        --precision float16 \
        --batch_size 2 \
        --vqav2_questions_json_path $VQAV2_QUESTION_PATH \
        --query_set_size $QUERY_SIZE \
        --cache_dir $CACHE \
        --mode $MODE \
        --output_log $OUTPUT_LOG \
        ${PREV_Q_OUT} \
        --instruction "$instruction"
done
