import os
import gzip
import json
import openai
import tqdm

sys_content = "I will give you a question and you will convert the question into a declarative sentence and give me the declarative one only. " + \
        "However, when asked color or location of the objects, the declarative one must not contain 'color'. Answer must be started with 'there'. " +\
        "examples are below: Q: what color is the chair?,A: there is a chair. Q: where is the chair located?, A: there is a chair."

sys_content_name = 'given a question, extract an object name which is the most helpful to answer the question?. Give only the object name' +\
        'the object name must be in "chair", "table", "picture", "cabinet", "cushion", "sofa", ' +\
        '"bed", "chest_of_drawers", "plant", "sink", "toilet", "stool", "towel", "tv_monitor", ' +\
        '"shower", "bathtub", "counter", "fireplace", "gym_equipment", "seating", "clothes". ' +\
        '"examples are below: Q: what color is the chair?,A: chair. Q: where is the chair located?, A: chair."'
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system",
         "content": sys_content},
        {"role": "user", "content": "what color is the chair"}
    ]   
)

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system",
         "content": sys_content_name},
        {"role": "user", "content": "what is in front of the tv"}
    ]   
)

##############
split = "val"
episodes_file=f"data/datasets/eqa/mp3d/v1/{split}/{split}_origin.json.gz"
save_episodes_file = f"data/datasets/eqa/mp3d/v1/{split}/{split}.json.gz"


with gzip.open(episodes_file, 'r') as f:
    json_data = json.loads(
        f.read().decode('utf-8'))

save_data = json_data

for episode in tqdm.tqdm(save_data["episodes"]):
    question = episode["question"]["question_text"]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                "content": sys_content},
                {"role": "user", "content": question}
            ],
            max_tokens=20,
            timeout = 20,
            temperature=0.1
        )
    except Exception as e:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                "content": sys_content},
                {"role": "user", "content": question}
            ],
            max_tokens=20,
            timeout = 20,
            temperature=0.1
        )
        
    try:
        response_name = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                "content": sys_content_name},
                {"role": "user", "content": question}
            ],
            max_tokens=4,
            timeout = 20,
            temperature=0.1
        )
    except Exception as e:
        response_name = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                "content": sys_content_name},
                {"role": "user", "content": question}
            ],
            max_tokens=4,
            timeout = 30,
            temperature=0.1
        )
    episode["llm"] = {}
    episode["llm"]["declarative_text"] = response["choices"][0]["message"]["content"]
    episode["llm"]["object"] = response_name["choices"][0]["message"]["content"]


json_data_save = json.dumps(save_data, indent=2)
# Convert to bytes
encoded = json_data_save.encode('utf-8')
# Compress
compressed = gzip.compress(encoded)
with open(save_episodes_file, "wb") as f:
    f.write(compressed)
