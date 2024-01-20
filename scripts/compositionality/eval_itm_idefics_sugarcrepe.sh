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
dataset='coco_sugarcrepe'
split='val'
IMG_PATH="${SCRATCH}/data/coco/val2017"



negtypes=("replace_att" "replace_obj" "replace_rel" "swap_att" "swap_obj" "add_obj" "add_att")
numsamples=(350 750 650 300 100 1000 300)
querysizes=(350 750 650 300 100 1000 300)


INSTRUCTION=''
batch_size=1
MAX_N_OBJS=15
num_trials=3

## Add user_assistant to MODE to use the IDEFICS-instruct template e.g. itm_user_assistant
## Modes
MODE="itm" # ITM

# MODE="itm_inst" # ITM + task instruction
# INSTRUCTION="You need to find if the provided sentences acurately describe the image, if the composition of the sentence does not match the image then the sentence does not describe the image. Here is few illustration examples: "

# MODE="itm_objects" # ITM MT-ICL

# MODE="its" # ITS


for index in {0,1,2,3,4,5,6};do


    for shot in {0,4,8,16,32};do


        NEG_TYPE="${negtypes[$index]}"
        NUM_SAMPLES=${numsamples[$index]}
        QUERY_SIZE=${querysizes[$index]} # 692

        ANNO_PATH="${SCRATCH}/data/sugarcrepe/${NEG_TYPE}.json"
        COMPOS="sugar" # prod sys
        PREFIX="${COMPOS}_${NEG_TYPE}"

        SAVE_DIR="${WORK}/logs/open_flamingo/${dataset}/${PREFIX}_${MODE}/${MODEL}"
        OUTPUT_LOG="${SAVE_DIR}/${shot}"
        RESULTS_FILE="${SAVE_DIR}/results_${shot}.json"


        echo $RESULTS_FILE

        python -m accelerate.commands.launch --mixed_precision=bf16 --num_processes=1 --num_machines=1 open_flamingo/eval/eval_acc_idefics.py \
            --lm_path $LM_TOKENIZER_PATH \
            --lm_tokenizer_path $LM_TOKENIZER_PATH \
            --device $DEVICE \
            --image_dir_path $IMG_PATH \
            --annotations_json_path $ANNO_PATH \
            --results_file $RESULTS_FILE \
            --eval_itm \
            --num_samples $NUM_SAMPLES \
            --shots $shot \
            --num_trials $num_trials \
            --low_cpu \
            --batch_size $batch_size \
            --mode $MODE \
            --compos $COMPOS \
            --neg_type $NEG_TYPE \
            --n_prod $N_PROD \
            --output_log $OUTPUT_LOG \
            --query_set_size $QUERY_SIZE \
            --cache_dir $CACHE \
            --instruction "$INSTRUCTION" \
            --text_prompt 'because there is only these objects:' \
            --max_n_objs $MAX_N_OBJS 
    done
done

