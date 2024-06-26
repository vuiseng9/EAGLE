
env: hgx1-240606-eagle-fork
python 3.11
transformers==4.41.2

https://github.com/vuiseng9/EAGLE
cd eagle
git remote add upstream https://github.com/SafeAILab/EAGLE
git checkout -b 240606-reproduce-L27B-chat-ea-drafter
pip install -e .
pip install datasets

# inference
```bash
CUDA_VISIBLE_DEVICES=0 python -m eagle.application.webui --ea-model-path yuhuili/EAGLE-llama2-chat-7B --base-model-path meta-llama/Llama-2-7b-chat-hf --model-type llama-2-chat

# local reproduce
CUDA_VISIBLE_DEVICES=1 python -m eagle.application.webui --ea-model-path /data2/vchua/run/hgx1-240606-eagle-fork/ea-llama2-7B-drafter/final --base-model-path meta-llama/Llama-2-7b-chat-hf --model-type llama-2-chat

# launch both aboth to see the delta, the lower one can only be done when training is complete
# also 
```

# where to get dataset? inferred from the fact that it is heavily based on medusa and some hardcoded path
```bash
based on medusa
https://github.com/FasterDecoding/Medusa?tab=readme-ov-file#prepare-the-data
git clone https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered
```

# data processing/collection
```bash
there is some hardcoding in the script, patched. see 
python -m eagle.ge_data.allocation --outdir /data/dataset/llama2-7B-chat-ShareGPT
```

# training
accelerate launch -m --mixed_precision=bf16 eagle.train.main --tmpdir /data/dataset/llama2-7B-chat-ShareGPT --cpdir /data2/vchua/run/hgx1-240606-eagle-fork/ea-llama2-7B-drafter --configpath eagle/train/llama_2_chat_7B_config.json

# PROMPT
Alan Turing theorized that computers would one day become

known issues or opens
1. epoch id is wrong
2. save ckpt is not intended for final usage, config.json missing
3. wandb random run_name


# training eagle drafter for llama3 instruction tuned
branch: 240607-L38B-ea-drafter

because of llama3 chat template, we update fschat from 0.2.31 to latest 
see commit here that integrate llama3
https://github.com/lm-sys/FastChat/commit/27a05b04a35510afb1d767ae7e5990cbd278f8fe

latest fschat-0.2.36 is released on feb24 release, llama3 template hasn't been updated. Therefore, we need a local installation.

git clone https://github.com/lm-sys/FastChat
cd FastChat
pip3 install -e ".[model_worker,webui]"
pip freeze | grep fschat
-e git+https://github.com/lm-sys/FastChat@c5223e34babd24c3f9b08205e6751ea6e42c9684#egg=fschat

Gradio is updated to latest but it will be break the current webui, 
therefore we revert gradio
uninstall gradio 4.36.0
install gradio-3.50.2
pip install gradio==3.50.2

# generate dataset
python -m eagle.ge_data.allocation --outdir /data3/vchua/dataset/llama3-8B-instruct-ShareGPT

# train
accelerate launch -m --mixed_precision=bf16 eagle.train.main --tmpdir /data3/vchua/dataset/llama3-8B-instruct-ShareGPT/ --cpdir /data2/vchua/run/hgx1-240606-eagle-fork/ea-llama3-8B-drafter --configpath /data/vchua/dev/hgx1-240606-eagle-fork/eagle/eagle/train/llama_3_instruct_8B_config.json

# inference
```bash
# 20 epoch
CUDA_VISIBLE_DEVICES=0 python -m eagle.application.webui --ea-model-path /data2/vchua/run/hgx1-240606-eagle-fork/ea-llama3-8B-drafter/final --base-model-path meta-llama/Meta-Llama-3-8B-Instruct --model-type llama-3

# 1st epoch
CUDA_VISIBLE_DEVICES=1 python -m eagle.application.webui --ea-model-path /data2/vchua/run/hgx1-240606-eagle-fork/ea-llama3-8B-drafter/state_0 --base-model-path meta-llama/Meta-Llama-3-8B-Instruct --model-type llama-3

# launch both aboth to see the delta, the lower one can only be done when training is complete
# also 
```

# Benchmarking notes
# benchmarks
1. baseline generation and perf count
```bash
CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_baseline_answer_llama2chat --base-model-path meta-llama/Llama-2-7b-chat-hf --ea-model-path yuhuili/EAGLE-llama2-chat-7B --model-id llama-2-chat-7B-fp16-baseline --temperature 0.0 --question-begin 5 --question-end 8
```
2. Eagle generation
```bash
CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama2chat --base-model-path meta-llama/Llama-2-7b-chat-hf --ea-model-path yuhuili/EAGLE-llama2-chat-7B --model-id llama-2-chat-7B-fp16-eagle --temperature 0.0 --question-begin 5 --question-end 8
```
3. Speedup calculation (dump mismatch report as well)
```bash
python eagle/evaluation/speed.py \
    --ea_path /pathto/mt_bench/llama-2-chat-7B-fp16-eagle-temperature-0.0.jsonl \
    --ar_path /pathto/mt_bench/llama-2-chat-7B-fp16-baseline-temperature-0.0.jsonl \
    --tokenizer_path meta-llama/Llama-2-7b-chat-hf 
```

4. Calculate Alpha
```bash
CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_alpha_llama2chat --base-model-path meta-llama/Llama-2-7b-chat-hf --ea-model-path yuhuili/EAGLE-llama2-chat-7B --model-id accept_len_llama-2-chat-7B-fp16-eagle-official --temperature 0.0
#  --question-begin 5 --question-end 8

CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_alpha_llama2chat --base-model-path meta-llama/Llama-2-7b-chat-hf --ea-model-path /data2/vchua/run/hgx1-240606-eagle-fork/ea-llama2-7B-drafter/final --model-id accept_len_llama-2-chat-7B-fp16-eagle-local-epoch20 --temperature 0.0
```

5. Quick profiling
```bash
CUDA_VISIBLE_DEVICES=0 python top_eagle.py
CUDA_VISIBLE_DEVICES=0 python top_autoreg.py
```

# Tree B (rolling acceptance length)
```bash
CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_alpha_llama2chat --base-model-path meta-llama/Llama-2-7b-chat-hf --ea-model-path yuhuili/EAGLE-llama2-chat-7B --model-id tree-b-accept_len_llama-2-chat-7B-fp16-eagle-official --temperature 0.0

CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_alpha_llama2chat --base-model-path meta-llama/Llama-2-7b-chat-hf --ea-model-path /data2/vchua/run/hgx1-240606-eagle-fork/ea-llama2-7B-drafter/final --model-id tree-b-accept_len_llama-2-chat-7B-fp16-eagle-local-epoch20 --temperature 0.0
```

understanding gap:
1. how do we get a autoregressive baseline? webui, untick eagle
2. what it is actually training? distillation or just standard cross entropy? data processing is just getting the features? cross entropy + feature regression
3. how data is getting processed? early look, 68000 sharegpt conversation is split into ngpu partition
4. what is sharegpt? crowd-source human conversations with ChatGPT/GPT4
5. technical question, is KV cache context being wasted from drafter
https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

bf16 or fp16?

llama3 system prompt is unknown. broader question is if this needs to be in a particular form? we are using the same as llama2
system prompt question
https://www.skool.com/4houraiworkweek/here-is-the-full-copy-of-meta-ais-new-llama-3-system-prompt

tokenizer will naturally split into slices of max_length, for eagle even in llama2, it is only 2k, should we stick to 2k or 4k (official), we stick to 2k for llama3 although it is supposed to be 8k

training opens - final model package is manual now. config.json and pytorch_model.bin are organized manually
how to do proper instruction-tuning?