

git checkout 240630-eval-eagle2
pip install -e .


# eagle2 webui 
```bash
# llama2-7B
CUDA_VISIBLE_DEVICES=0 python -m eagle.application.webui --ea-model-path yuhuili/EAGLE-llama2-chat-7B --base-model-path meta-llama/Llama-2-7b-chat-hf --model-type llama-2-chat --total-token -1
# llama3-8b
CUDA_VISIBLE_DEVICES=0 python -m eagle.application.webui --ea-model-path yuhuili/EAGLE-LLaMA3-Instruct-8B --base-model-path meta-llama/Meta-Llama-3-8B-Instruct --model-type llama-3-instruct --total-token -1
```



#  DRYRUN

CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_baseline_answer_llama2chat --base-model-path meta-llama/Llama-2-7b-chat-hf --ea-model-path yuhuili/EAGLE-llama2-chat-7B --model-id llama-2-chat-7B-fp16-baseline --temperature 0.0 --question-begin 5 --question-end 8

CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama2chat --base-model-path meta-llama/Llama-2-7b-chat-hf --ea-model-path yuhuili/EAGLE-llama2-chat-7B --model-id llama-2-chat-7B-fp16-eagle --temperature 0.0 --question-begin 5 --question-end 8

python eagle/evaluation/speed.py \
    --ea_path mt_bench/llama-2-chat-7B-fp16-eagle-temperature-0.0.jsonl \
    --ar_path mt_bench/llama-2-chat-7B-fp16-baseline-temperature-0.0.jsonl \
    --tokenizer_path meta-llama/Llama-2-7b-chat-hf \
    --rpt_id dryrun



CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_alpha_llama2chat --base-model-path meta-llama/Llama-2-7b-chat-hf --ea-model-path yuhuili/EAGLE-llama2-chat-7B --model-id accept_len_llama-2-chat-7B-fp16-eagle-official --temperature 0.0 --question-begin 5 --question-end 8
