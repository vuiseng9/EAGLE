from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.model import get_conversation_template
import torch
import os

# from eagle.model.choices import mc_sim_7b_63

your_message="Alan Turing theorized that computers would one day become"

# your_message="Tell me a story of a hare and a tortoise"

use_llama_3_chat=False
use_llama_2_chat=True
use_vicuna=False

if use_llama_3_chat:
    target_model = "meta-llama/Meta-Llama-3-8B-Instruct"
    ea_drafter = "/data2/vchua/run/hgx1-240606-eagle-fork/ea-llama3-8B-drafter/final"
    conv = get_conversation_template(target_model)  
    sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    conv.system_message = sys_p
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


if use_llama_2_chat:
    target_model = "meta-llama/Llama-2-7b-chat-hf"
    ea_drafter = "yuhuili/EAGLE-llama2-chat-7B"
    conv = get_conversation_template("llama-2-chat")  
    sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    conv.system_message = sys_p
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " "


if use_vicuna:
    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


#load Hugging Face Transformers model with INT4 optimizations
# model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, torch_dtype = torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    target_model,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(target_model) 

model.eval()


import types
import inspect
import time
org_forward = model.forward
tok_latencies = []

def new_forward(self, *args, **kwargs):
    # Call the original method
    _t0 = time.time()
    ret = org_forward.__func__(self, *args, **kwargs)
    torch.cuda.synchronize()
    _t1 = time.time()
    tok_latencies.append(_t1 - _t0)
    return ret

new_forward.__signature__ = inspect.signature(model.forward)
model.forward = types.MethodType(new_forward, model)



input_ids=tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()
#warmup
output_ids = model.generate(input_ids, do_sample=False, top_p=None, num_beams=1, max_new_tokens=1024)
tok_latencies = [] #reset
t1 = time.time()
output_ids = model.generate(input_ids, do_sample=False, top_p=None, num_beams=1, max_new_tokens=1024)
t2 = time.time()
output = tokenizer.batch_decode(output_ids.cpu()) # alternative way
# output= tokenizer.decode(output_ids[0])
print(output)

dense_elapsed = t2 - t1
second_token_elapse_list = []

for idx, elapse in enumerate(tok_latencies):
    if idx > 0:
        second_token_elapse_list.append(elapse)
    else:
        first_latency = elapse

count_2nd_tokens_dense = len(second_token_elapse_list)
mean_2nd_tokens_dense = sum(second_token_elapse_list)/count_2nd_tokens_dense

print(f"in millisec, e2e: {sum(tok_latencies)*1000:6.3f}, 1st: {first_latency*1000:6.3f}, 2nd+: {mean_2nd_tokens_dense*1000:8.3f} ({1/mean_2nd_tokens_dense:5.1f} TPS)")
print("end.")
