from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch
import os

# from eagle.model.choices import mc_sim_7b_63

# your_message="Alan Turing theorized that computers would one day become"

# your_message="Tell me a story of a hare and a tortoise"

your_message="Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."


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

prompt='''
[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
 
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
 
Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions. [/INST] Title: Aloha Hawaii: Exploring the Aloha State's Cultural Wonders and Natural Beauty
 
Introduction:
Hawaii, the Aloha State, is a tropical paradise known for its stunning natural beauty, rich cultural heritage, and warm hospitality. Recently, I had the opportunity to visit this enchanting archipelago, and I must say, it exceeded my expectations in every way. In this travel blog post, I will share some of the highlights of my trip, including the must-see attractions, cultural experiences, and the breathtaking landscapes that left me spellbound.
 
Must-see Attractions:
 
1. Waikiki Beach: The iconic Waikiki Beach is a must-visit destination in Hawaii. With its pristine white sand, crystal-clear waters, and stunning sunsets, it's no wonder why this beach is one of the most popular in the world. Take a leisurely stroll along the beach, enjoy the local cuisine, or simply soak up the sun and relax.
2. Pearl Harbor: Visit the USS Arizona Memorial and the Pacific Aviation Museum to learn about the historical significance of Pearl Harbor. It's a poignant reminder of the sacrifices made during World War II and a sobering reminder of the importance of peace and diplomacy.
3. Haleakala National Park: Watch the sunrise from the summit of Haleakala, a dormant volcano that offers breathtaking views of the island. The park is also home to diverse wildlife, including the endangered Hawaiian goat and the rare Hawaiian petrel.
4. The Road to Hana: This scenic drive takes you through lush rainforests, along rugged coastlines, and past cascading waterfalls. Stop at the picturesque towns of Hana and Ke'anae for a taste of local culture and cuisine.
 
Cultural Experiences:
 
1. Luau: Experience Hawaiian culture and tradition at a traditional luau. Watch the hula dancing, enjoy the local cuisine, and learn about the history and customs of the islands.
2. Visit a Hawaiian Temple: Explore the beautiful temples of Hawaii, such as the Byodo-In Temple in Kaneohe or the Kadphises Temple in Honolulu. These temples offer a glimpse into the spiritual beliefs and practices of the Hawaiian people.
3. Learn Hula Dancing: Take a hula dancing class to learn the traditional dance of Hawaii. You'll learn the basic steps and movements, as well as the cultural significance of this iconic dance.
4. Attend a Hawaiian Festival: Join in the fun at one of Hawaii's many festivals, such as the Merrie Monarch Festival or the Hawaii Volcanoes National Park Festival. These events celebrate Hawaiian culture, music, and art, and offer a unique and unforgettable experience.
 
Conclusion:
Hawaii is a truly special place that offers a unique blend of natural beauty, cultural heritage, and warm hospitality. Whether you're interested in exploring the islands' stunning landscapes, immersing yourself in local culture, or simply relaxing on the beach, Hawaii has something for everyone. So pack your bags, grab your sunscreen, and get ready for the adventure of a lifetime in the Aloha State! </s><s>[INST] Rewrite your previous response. Start every sentence with the letter A. [/INST]"
'''



# Must set before instantiation of EaModel
os.environ['PROFILE_EAGLE'] = 'ON'

model = EaModel.from_pretrained(
    base_model_path=target_model,
    ea_model_path=ea_drafter,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto", 
)
model.eval()

torch.manual_seed(0)
input_ids=model.tokenizer([prompt]).input_ids
print("prompt shape", len(input_ids[0]))
input_ids = torch.as_tensor(input_ids).cuda()
output_ids=model.eagenerate(input_ids,temperature=0.0,max_new_tokens=2048)
# output= model.tokenizer.batch_decode(output_ids.cpu())
output=model.tokenizer.decode(output_ids[0])

print(output, flush=True)
print("output_ids shape", output_ids.shape)

if model.tree_decoding_timer is not None:
    model.gen_cand_timer.current()
    model.tree_decoding_timer.current()
    model.eval_posterior_timer.current()
    model.draft_timer.current()

print("end")