#!/bin/bash

export LC_ALL=C  
export TRANSFORMERS_CACHE=path_to/.cache/huggingface/transformers
export CACHE=path_to/.cache/clip

DATA='/gpfsDATA/rech/dyf/ugz83ue'
LOGS='/gpfsLOGS/rech/dyf/ugz83ue'


DEVICE="0"







## Models
LM_PATH='HuggingFaceM4/idefics-9b'
LM_TOKENIZER_PATH='HuggingFaceM4/idefics-9b'
MODEL="IDEFICS9B"

# LM_PATH='HuggingFaceM4/idefics-9b-instruct'
# LM_TOKENIZER_PATH='HuggingFaceM4/idefics-9b-instruct'
# MODEL="IDEFICS9BINSTRUCT"




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
# INSTRUCTION="You will be given a question and answer, you need to give an explanation of the given answer based on the image. Here is few illustration examples: "
 
# MODE="answer_and_explain" # MT-ICL

# MODE="explainonly_evalquery" # ICL explain only for the query data

# MODE="explainonly_contrastive_prevqout_32shotvsgt" # CoH-ICL
# PREV_Q_OUT="--previous_query_predictions ${WORK}/logs/open_flamingo/${dataset}/coh_randswap_explainonly_evalquery/${MODEL}/32/vqax_results_seed.json"




num_trials=3

for shot in {0,4,8,16,32};do
    SAVE_DIR="${WORK}/logs/open_flamingo/${dataset}/${MODE}/${MODEL}"
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
        --eval_vqax \
        --num_samples $NUM_SAMPLE \
        --shots $shot \
        --num_trials $num_trials \
        --low_cpu \
        --batch_size 2 \
        --mode $MODE \
        --query_set_size $QUERY_SIZE \
        --skip_use_local_files \
        --output_log $OUTPUT_LOG \
        ${PREV_Q_OUT} \
        --cache_dir $CACHE \
        --instruction "$INSTRUCTION"

done

