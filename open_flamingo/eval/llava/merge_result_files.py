import jsonlines
import json 

model = "IDEFICS9B" # IDEFICS9B 9Bmpt
shot = 2 # 1, 0, 2, 4, 8


for shot in [1, 0, 2, 4, 8]:
    results = []
    question_ids = list(range(0, 90))
    for mode in ['llava_detail_questions', 'llava_complex_questions', 'llava_conv_questions']:
        res1_path = f"/data/mshukor/logs/open_flamingo/vqav2/{mode}/{model}/{shot}/llava_results_0.json"
        res_ = json.load(open(res1_path))
        # res = [r for r in res_ if r['text']]
        results+=res_

    qid_2_res = {r['question_id']: r for r in results}

    jsonl_file_path = f"/data/mshukor/logs/open_flamingo/vqav2/combined_files/combined_{model}_{shot}_llava_results_0.json"
    with jsonlines.open(jsonl_file_path, mode='w') as writer:

        for qid in question_ids:

            item = qid_2_res[qid]
            writer.write(item)

    print("save to:", jsonl_file_path)