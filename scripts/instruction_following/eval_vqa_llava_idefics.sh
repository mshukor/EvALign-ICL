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
TESTSET='--testset' # set to '' if you want to test on llava training set
TEST_IMG_DIR="${DATA}/data/coco/val2014"



dataset='llava'
split="train"
VQAV2_IMG_PATH="${DATA}/data/coco/${split}2014"
QUERY_SIZE=200 # for ICL examples
NUM_SAMPLE=40

 

 
for shot in {1,0,2,4,8,16,32};do


    TEST_ANN_PATH="${DATA}/data/llava/test_llava_complex_reasoning_77k.json"

    MODE="llava_complex_questions"
    VQAV2_ANNO_PATH="${DATA}/data/llava/llava_complex_reasoning_77k.json"
    echo $MODE
    echo $TEST_ANN_PATH


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
        --eval_vqav2 \
        --num_samples $NUM_SAMPLE \
        --shots $shot \
        --num_trials 1 \
        --low_cpu \
        --batch_size 1 \
        --query_set_size $QUERY_SIZE \
        --mode $MODE \
        --output_log $OUTPUT_LOG \
        --test_annotations_json_path $TEST_ANN_PATH \
        --test_image_dir_path $TEST_IMG_DIR \
        --cache_dir $CACHE \
        $TESTSET \

        
    TEST_ANN_PATH="${DATA}/data/llava/test_llava_detail_23k.json"


    MODE="llava_detail_questions"
    VQAV2_ANNO_PATH="${DATA}/data/llava/llava_detail_23k.json"

    echo $MODE
    echo $TEST_ANN_PATH

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
        --eval_vqav2 \
        --num_samples $NUM_SAMPLE \
        --shots $shot \
        --num_trials 1 \
        --low_cpu \
        --batch_size 1 \
        --query_set_size $QUERY_SIZE \
        --mode $MODE \
        --output_log $OUTPUT_LOG \
        --test_annotations_json_path $TEST_ANN_PATH \
        --test_image_dir_path $TEST_IMG_DIR \
        --cache_dir $CACHE \
        $TESTSET \



    TEST_ANN_PATH="${DATA}/data/llava/test_llava_conversation_58k.json"




    MODE="llava_conv_questions"
    VQAV2_ANNO_PATH="${DATA}/data/llava/llava_conversation_58k.json"

    echo $MODE
    echo $TEST_ANN_PATH

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
        --eval_vqav2 \
        --num_samples $NUM_SAMPLE \
        --shots $shot \
        --num_trials 1 \
        --low_cpu \
        --batch_size 1 \
        --query_set_size $QUERY_SIZE \
        --mode $MODE \
        --output_log $OUTPUT_LOG \
        --test_annotations_json_path $TEST_ANN_PATH \
        --test_image_dir_path $TEST_IMG_DIR \
        --cache_dir $CACHE \
        $TESTSET \



    TEST_ANN_PATH="${DATA}/data/llava/test_llava_conversation_58k.json"


    MODE="llava_conv_dialog_questions"
    VQAV2_ANNO_PATH="${DATA}/data/llava/llava_conversation_58k.json"

    echo $MODE
    echo $TEST_ANN_PATH

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
        --eval_vqav2 \
        --num_samples $NUM_SAMPLE \
        --shots $shot \
        --num_trials 1 \
        --low_cpu \
        --batch_size 1 \
        --query_set_size $QUERY_SIZE \
        --mode $MODE \
        --output_log $OUTPUT_LOG \
        --test_annotations_json_path $TEST_ANN_PATH \
        --test_image_dir_path $TEST_IMG_DIR \
        --cache_dir $CACHE 

done




