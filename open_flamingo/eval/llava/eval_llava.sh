#!/bin/bash


# 1: OPENAI_API_KEY="sk-***********************************" sh eval_lava.sh
# 2: python summarize_gpt_review.py --dir /data/mshukor/logs/open_flamingo/vqav2/gpt4_eval/ --files /data/mshukor/logs/open_flamingo/vqav2/gpt4_eval/combined_IDEFICS9BINSTRUCTuser_assistant_*_llava_results_0_review.json

model="IDEFICS80BINSTRUCTuser_assistant" # IDEFICS9B IDEFICS9BINSTRUCTuser_assistant 9Bmpt IDEFICS80B IDEFICS80BINSTRUCTuser_assistant
for shot in {1,0,2,4,8};do

    file_name="combined_${model}_${shot}_llava_results_0"
    result_path="/data/mshukor/logs/open_flamingo/vqav2/combined_files/$file_name.json"
    output_dir="/data/mshukor/logs/open_flamingo/vqav2/gpt4_eval"
    output_path="$output_dir/${file_name}_review.json"
    echo $result_path
    python eval_gpt_review_visual.py \
        --question ./qa90_questions.jsonl \
        --context ./caps_boxes_coco2014_val_80.jsonl \
        --answer-list \
        ./qa90_gpt4_answer.jsonl \
        $result_path \
        --rule ./rule.jsonl \
        --output $output_path 
        # --debug
done;

# python summarize_gpt_review.py \
# --dir $output_dir 
# # --files $output_path \
