from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch
import os

your_message="Alan Turing theorized that computers would one day become"

use_llama_2_chat=True
if use_llama_2_chat:
    target_model = "meta-llama/Llama-2-7b-chat-hf"
    ea_drafter = "yuhuili/EAGLE-llama2-chat-7B"
    conv = get_conversation_template("llama-2-chat")  
    sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    conv.system_message = sys_p
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " "

model = EaModel.from_pretrained(
    base_model_path=target_model,
    ea_model_path=ea_drafter,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1
)
model.eval()

torch.manual_seed(0)
input_ids=model.tokenizer([prompt]).input_ids
print("prompt shape", len(input_ids[0]))
input_ids = torch.as_tensor(input_ids).cuda()
output_ids=model.eagenerate(input_ids,temperature=0.0,max_new_tokens=512)
# output= model.tokenizer.batch_decode(output_ids.cpu())
output=model.tokenizer.decode(output_ids[0])

print(output, flush=True)
print("output_ids shape", output_ids.shape)
print("end.")