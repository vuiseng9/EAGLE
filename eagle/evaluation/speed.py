import json
import argparse
from transformers import AutoTokenizer
import numpy as np

def main(tokenizer_id, eagle_json, autoregressive_json):
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_id)
    jsonl_file = eagle_json
    jsonl_file_base = autoregressive_json
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
