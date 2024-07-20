#!/usr/bin/env bash

cd ~/vchua/dev/g5-vchua-240630-eagle2/eagle

CUDA_ID=0
# DEV="--question-begin 5 --question-end 8"
DEV=""

echo "[Info]: Running Llama2-7B Chat model in AutoRegressive mode on MT_BENCH"
ARLABEL=eagle2-llama-2-chat-7B-fp16-autoregressive
CUDA_VISIBLE_DEVICES=${CUDA_ID} python -m eagle.evaluation.gen_baseline_answer_llama2chat \
    --base-model-path meta-llama/Llama-2-7b-chat-hf \
    --ea-model-path yuhuili/EAGLE-llama2-chat-7B \
    --temperature 0.0 \
    --model-id $ARLABEL \
    $DEV

TOTAL_TOKEN=60
echo "[Info]: Running Llama2-7B Chat model with OFFICIAL Eagle model on MT_BENCH - total draft token ${TOTAL_TOKEN}"
EALABEL=eagle2-llama-2-chat-7B-fp16-eagle-official-default-total-$TOTAL_TOKEN
CUDA_VISIBLE_DEVICES=${CUDA_ID} python -m eagle.evaluation.gen_ea_answer_llama2chat \
    --base-model-path meta-llama/Llama-2-7b-chat-hf \
    --ea-model-path yuhuili/EAGLE-llama2-chat-7B \
    --temperature 0.0 \
    --model-id $EALABEL \
    --total-token $TOTAL_TOKEN \
    $DEV
    
echo "[Info]: Evaluating OFFICIAL speedup"
python eagle/evaluation/speed.py \
    --ea_path ./mt_bench/${EALABEL}-temperature-0.0.jsonl \
    --ar_path ./mt_bench/${ARLABEL}-temperature-0.0.jsonl \
    --tokenizer_path meta-llama/Llama-2-7b-chat-hf \
    --rpt_id llama2-7b-ea-official

TOTAL_TOKEN=-1
echo "[Info]: Running Llama2-7B Chat model with OFFICIAL Eagle model on MT_BENCH - total draft token ${TOTAL_TOKEN}"
EALABEL=eagle2-llama-2-chat-7B-fp16-eagle-official-default-total-$TOTAL_TOKEN
CUDA_VISIBLE_DEVICES=${CUDA_ID} python -m eagle.evaluation.gen_ea_answer_llama2chat \
    --base-model-path meta-llama/Llama-2-7b-chat-hf \
    --ea-model-path yuhuili/EAGLE-llama2-chat-7B \
    --temperature 0.0 \
    --model-id $EALABEL \
    --total-token $TOTAL_TOKEN \
    $DEV
    
echo "[Info]: Evaluating OFFICIAL speedup"
python eagle/evaluation/speed.py \
    --ea_path ./mt_bench/${EALABEL}-temperature-0.0.jsonl \
    --ar_path ./mt_bench/${ARLABEL}-temperature-0.0.jsonl \
    --tokenizer_path meta-llama/Llama-2-7b-chat-hf \
    --rpt_id llama2-7b-ea-official
