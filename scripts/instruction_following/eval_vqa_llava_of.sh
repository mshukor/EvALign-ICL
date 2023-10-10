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
        --eval_vqav2 \
        --num_samples $NUM_SAMPLE \
        --shots $shot \
        --num_trials 1 \
        --low_cpu \
        --precision float16 \
        --batch_size 1 \
        --vqav2_questions_json_path $VQAV2_QUESTION_PATH \
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
        --eval_vqav2 \
        --num_samples $NUM_SAMPLE \
        --shots $shot \
        --num_trials 1 \
        --low_cpu \
        --precision float16 \
        --batch_size 1 \
        --vqav2_questions_json_path $VQAV2_QUESTION_PATH \
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
        --eval_vqav2 \
        --num_samples $NUM_SAMPLE \
        --shots $shot \
        --num_trials 1 \
        --low_cpu \
        --precision float16 \
        --batch_size 1 \
        --vqav2_questions_json_path $VQAV2_QUESTION_PATH \
        --query_set_size $QUERY_SIZE \
        --mode $MODE \
        --output_log $OUTPUT_LOG \
        --test_annotations_json_path $TEST_ANN_PATH \
        --test_image_dir_path $TEST_IMG_DIR \
        --cache_dir $CACHE \
        $TESTSET \


    MODE="llava_conv_dialog_questions"
    VQAV2_ANNO_PATH="${DATA}/data/llava/llava_conversation_58k.json"

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
        --eval_vqav2 \
        --num_samples $NUM_SAMPLE \
        --shots $shot \
        --num_trials 1 \
        --low_cpu \
        --precision float16 \
        --batch_size 1 \
        --vqav2_questions_json_path $VQAV2_QUESTION_PATH \
        --query_set_size $QUERY_SIZE \
        --mode $MODE \
        --output_log $OUTPUT_LOG \
        --cache_dir $CACHE \

done


