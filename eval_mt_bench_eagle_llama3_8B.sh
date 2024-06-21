#!/usr/bin/env bash

cd /data/vchua/dev/hgx1-240606-eagle-fork/eagle

CUDA_ID=0
# DEV="--question-begin 5 --question-end 8"
DEV=""

echo "[Info]: Running llama3-8B-instruct model in AutoRegressive mode on MT_BENCH"
ARLABEL=llama3-8B-instruct-fp16-autoregressive
CUDA_VISIBLE_DEVICES=${CUDA_ID} python -m eagle.evaluation.gen_baseline_answer_llama3instruct \
    --base-model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --ea-model-path /data2/vchua/run/hgx1-240606-eagle-fork/ea-llama3-8B-drafter/final \
    --temperature 0.0 \
    --model-id $ARLABEL \
    $DEV

echo "[Info]: Running llama3-8B-instruct model with local reproduce Eagle model on MT_BENCH"
EALABEL=llama3-8B-instruct-fp16-eagle-local-epoch20
CUDA_VISIBLE_DEVICES=${CUDA_ID} python -m eagle.evaluation.gen_ea_answer_llama3instruct \
    --base-model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --ea-model-path /data2/vchua/run/hgx1-240606-eagle-fork/ea-llama3-8B-drafter/final \
    --temperature 0.0 \
    --model-id $EALABEL \
    $DEV

echo "[Info]: Evaluating local reproduce speedup"
python eagle/evaluation/speed.py \
    --ea_path ./mt_bench/${EALABEL}-temperature-0.0.jsonl \
    --ar_path ./mt_bench/${ARLABEL}-temperature-0.0.jsonl \
    --tokenizer_path meta-llama/Llama-2-7b-chat-hf \
    --rpt_id llama3-8B-ea-local-epoch20
