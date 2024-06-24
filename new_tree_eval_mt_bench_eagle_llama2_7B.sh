#!/usr/bin/env bash

cd /data/vchua/dev/hgx1-240606-eagle-fork/eagle

CUDA_ID=0
# DEV="--question-begin 5 --question-end 8"
DEV=""

echo "[Info]: Running Llama2-7B Chat model in AutoRegressive mode on MT_BENCH"
ARLABEL=tree-a-llama-2-chat-7B-fp16-autoregressive
CUDA_VISIBLE_DEVICES=${CUDA_ID} python -m eagle.evaluation.gen_baseline_answer_llama2chat \
    --base-model-path meta-llama/Llama-2-7b-chat-hf \
    --ea-model-path yuhuili/EAGLE-llama2-chat-7B \
    --temperature 0.0 \
    --model-id $ARLABEL \
    $DEV

echo "[Info]: Running Llama2-7B Chat model with OFFICIAL Eagle model on MT_BENCH"
EALABEL=tree-a-llama-2-chat-7B-fp16-eagle-official
CUDA_VISIBLE_DEVICES=${CUDA_ID} python -m eagle.evaluation.gen_ea_answer_llama2chat \
    --base-model-path meta-llama/Llama-2-7b-chat-hf \
    --ea-model-path yuhuili/EAGLE-llama2-chat-7B \
    --temperature 0.0 \
    --model-id $EALABEL \
    $DEV
    
echo "[Info]: Evaluating OFFICIAL speedup"
python eagle/evaluation/speed.py \
    --ea_path ./mt_bench/${EALABEL}-temperature-0.0.jsonl \
    --ar_path ./mt_bench/${ARLABEL}-temperature-0.0.jsonl \
    --tokenizer_path meta-llama/Llama-2-7b-chat-hf \
    --rpt_id tree-a-llama2-7b-ea-official


# ---------------------------------------------------------------------------------------------
echo "[Info]: Running Llama2-7B Chat model with local reproduce Eagle model on MT_BENCH"
EALABEL=tree-a-llama-2-chat-7B-fp16-eagle-local-epoch20
CUDA_VISIBLE_DEVICES=${CUDA_ID} python -m eagle.evaluation.gen_ea_answer_llama2chat \
    --base-model-path meta-llama/Llama-2-7b-chat-hf \
    --ea-model-path /data2/vchua/run/hgx1-240606-eagle-fork/ea-llama2-7B-drafter/final \
    --temperature 0.0 \
    --model-id $EALABEL \
    $DEV

echo "[Info]: Evaluating local reproduce speedup"
python eagle/evaluation/speed.py \
    --ea_path ./mt_bench/${EALABEL}-temperature-0.0.jsonl \
    --ar_path ./mt_bench/${ARLABEL}-temperature-0.0.jsonl \
    --tokenizer_path meta-llama/Llama-2-7b-chat-hf \
    --rpt_id tree-a-llama2-7b-ea-local-epoch20
