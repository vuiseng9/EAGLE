
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

"prompt"
Alan Turing theorized that computers would one day become

known issues or opens
1. epoch id is wrong
2. save ckpt is not intended for final usage, config.json missing
3. wandb random run_name