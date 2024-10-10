import os
import gzip
import json
from PIL import Image
import requests
from io import BytesIO
import tqdm
import openai
from openai import AzureOpenAI
import argparse

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--apikey_path', type=str, default=None, required=True)
    parser.add_argument('--split', choices=["train", "val"], help='Choose dataset split to preprocess')
    parser.add_argument('--dataset', type=str, default="mp3d", required=True)
    parser.add_argument('--version', type=str, default="v1", required=True)
    parser.add_argument('--input_eqa_dataset', type=str, default=None, required=True)
    args = parser.parse_args()
    with open(args.apikey_path,'r') as f:
        apikey = json.load(f)
    os.environ["AZURE_OPENAI_ENDPOINT"] = apikey["AZURE_OPENAI_ENDPOINT"]
    os.environ["AZURE_OPENAI_KEY"] = apikey["AZURE_OPENAI_KEY"]
    del apikey

    split = args.split
    dataset = args.dataset
    version = args.version
    episodes_file=args.input_eqa_dataset
    save_episodes_dir = f"data/datasets/eqa/{dataset}/{version}/{split}"
    save_episodes_file = f"{save_episodes_dir}/{split}.json.gz"
    os.makedirs(save_episodes_dir, exist_ok=True)

    client = AzureOpenAI(
    api_key = os.getenv("AZURE_OPENAI_KEY"),  
    api_version = "2023-05-15",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    make_declarative_text_content = "I will give you a question and you will convert the question into a declarative sentence and give me the declarative one only." +\
        " However, when asked color or location of the objects, the declarative one must not contain 'color'." +\
        " Examples are below: Q: what color is the chair in the living room?,A: there is a chair in the living room." +\
        " Q: what is under the sink?, A: there is something under the sink."

    extract_object_cat_content = "Given a question, extract an target object name which is the most helpful to answer the question." +\
        " Give only the object name. Examples are below:" +\
        " Q: what color is the wooden chair in the living room?, A: chair." +\
        " Q: what is under the sink?, A: sink."


    if version=="v_zeroshot2":
        sys_content_name = "Given a question, extract an target object name from the question." +\
            " Give only the object name. Examples are below:" +\
            " Q: what color is the wooden chair in the living room?, A: chair." +\
            " Q: what is under the sink?, A: sink."


    with gzip.open(episodes_file, 'r') as f:
        json_data = json.loads(
            f.read().decode('utf-8'))

    save_data = json_data
    total_action_counter = 0

    for episode in tqdm.tqdm(save_data["episodes"]):
        question = episode["question"]["question_text"]
        while_loop_controler = 0
        temperature = 0.1
        total_action_counter += 1
        try:
            valid=True
            response = client.chat.completions.create(
                model="gpt-35-turbo",
                messages=[
                    {"role": "system",
                    "content": make_declarative_text_content},
                    {"role": "user", "content": question}
                ],
                max_tokens=20,
                temperature=temperature,
            )
        except Exception as e:
            valid=False
            total_action_counter += 1

        temperature = 0.1
        if valid:
            total_action_counter += 1
            try:
                response_object_cat = client.chat.completions.create(
                    model="gpt-35-turbo",
                    messages=[
                        {"role": "system",
                        "content": extract_object_cat_content},
                        {"role": "user", "content": question}
                    ],
                    max_tokens=4,
                    temperature=0.1,
                )
            except Exception as e:
                total_action_counter += 1
                valid=False
        if valid:
            episode["llm"] = {}
            episode["llm"]["declarative_text"] = response.choices[0].message.content
            episode["llm"]["object"] = response_object_cat.choices[0].message.content
            episode["llm"]["valid"]=True
        else:
            episode["llm"] = {}
            episode["llm"]["declarative_text"] = ""
            episode["llm"]["object"] = ""
            episode["llm"]["valid"]=False
        print(total_action_counter)

    json_data_save = json.dumps(save_data, indent=2)
    # Convert to bytes
    encoded = json_data_save.encode('utf-8')
    # Compress
    compressed = gzip.compress(encoded)
    with open(save_episodes_file, "wb") as f:
        f.write(compressed)

if __name__ == "__main__":
    main()