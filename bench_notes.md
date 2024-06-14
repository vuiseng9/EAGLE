# benchmarks
1. baseline generation and perf count
```bash
CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_baseline_answer_llama2chat --base-model-path meta-llama/Llama-2-7b-chat-hf --ea-model-path yuhuili/EAGLE-llama2-chat-7B --model-id llama-2-chat-7B-fp16-baseline --temperature 0.0 --question-begin 5 --question-end 8
```
2. Eagle generation
```bash
CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama2chat --base-model-path meta-llama/Llama-2-7b-chat-hf --ea-model-path yuhuili/EAGLE-llama2-chat-7B --model-id llama-2-chat-7B-fp16-eagle --temperature 0.0 --question-begin 5 --question-end 8
```
3. Speedup calculation
```bash
python eagle/evaluation/speed.py \
    --ea_path /pathto/mt_bench/llama-2-chat-7B-fp16-eagle-temperature-0.0.jsonl \
    --ar_path /pathto/mt_bench/llama-2-chat-7B-fp16-baseline-temperature-0.0.jsonl \
    --tokenizer_path meta-llama/Llama-2-7b-chat-hf 
```
4. Equivalence Generation check