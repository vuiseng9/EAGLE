import json
import argparse
from transformers import AutoTokenizer
import numpy as np
from collections import OrderedDict
from datetime import date, datetime

def check_output_equivalence(eagle_json, autoregressive_json): 
    def get_question_response(json_file):
        qa_dict = {}
        with open(json_file, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line)
                qa_dict[json_obj['question_id']] = json_obj['choices'][0]['turns'] # list of turn response
        return qa_dict
    
    ref_dict = get_question_response(autoregressive_json)
    sd_dict = get_question_response(eagle_json)

    non_intersecting_keys = set(sd_dict.keys()).symmetric_difference(set(ref_dict.keys()))

    if len(non_intersecting_keys) > 0:
        raise ValueError("Non-equivalent set of input questions")
    
    count_match = 0
    count_mismatch = 0
    mismatch_dict = []
    for qid, turns in sd_dict.items():
        for tid, resp in enumerate(turns):
            if ref_dict[qid][tid] == resp:
                count_match += 1
            else:
                count_mismatch += 1
                mismatch_dict.append(OrderedDict({
                    "question_id": qid,
                    "turn_id": tid,
                    "ar_resp": ref_dict[qid][tid],
                    "sd_resp": resp
                    }))
    ts = datetime.now().strftime('%y-%m-%d_%H-%M-%S')

    mismatch_dict.insert(0, {"ar_path": autoregressive_json, "sd_path": eagle_json, "match": count_match, "mismatch": count_mismatch})

    equivalence_rpt = f"{ts}_equivalence_check_match_{count_match},mismatch_{count_mismatch}.json"

    with open(equivalence_rpt, "w") as f:
        json.dump(mismatch_dict, f, indent=4)

    print(f"Path to mismatch rpt: {equivalence_rpt}")
    print("")

def main(tokenizer_id, eagle_json, autoregressive_json):
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_id)
    jsonl_file = eagle_json
    jsonl_file_base = autoregressive_json

    check_output_equivalence(jsonl_file, jsonl_file_base)

    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    speeds=[]
    for datapoint in data:
        qid=datapoint["question_id"]
        answer=datapoint["choices"][0]['turns']
        tokens=sum(datapoint["choices"][0]['new_tokens'])
        times = sum(datapoint["choices"][0]['wall_time'])
        speeds.append(tokens/times)

    data = []
    with open(jsonl_file_base, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    total_time=0
    total_token=0
    speeds0=[]
    for datapoint in data:
        qid=datapoint["question_id"]
        answer=datapoint["choices"][0]['turns']
        tokens = 0
        for i in answer:
            tokens += (len(tokenizer(i).input_ids) - 1)
        times = sum(datapoint["choices"][0]['wall_time'])
        speeds0.append(tokens / times)
        total_time+=times
        total_token+=tokens

    print("Speedup ratio",np.array(speeds).mean()/np.array(speeds0).mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ea_path",
        default=None,
        type=str,
        help="The file path of evaluated Speculative Decoding methods.",
    )
    parser.add_argument(
        "--ar_path",
        default=None,
        type=str,
        help="The file path of evaluated baseline.",
    )
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        type=str,
        help="tokenizer id",
    )
    args = parser.parse_args()

    main(args.tokenizer_path, args.ea_path, args.ar_path)
