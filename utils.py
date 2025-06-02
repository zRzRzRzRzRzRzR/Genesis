import base64
import json
import os
import re

import numpy as np
import yaml
from openai import AzureOpenAI, OpenAI


def load_yaml_config(config_path, api_name, config_type="llm_config"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config_type not in config:
        raise ValueError(f"配置文件中未找到 '{config_type}' 配置")

    api_config = config[config_type].get(api_name)
    if not api_config:
        raise ValueError(f"未找到 '{api_name}' 的配置")

    return {"model": api_config["model"], "base_url": api_config["base_url"], "api_key": api_config["api_key"]}


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


def call_large_model_llm(messages, api_key=None, base_url=None, model=None, version="2024-12-01-preview"):
    if "azure" in base_url:
        client = AzureOpenAI(api_key=api_key, azure_endpoint=base_url, api_version=version)
    else:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=1000)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=8192,
            stream=False,
        )
        if response.choices[0].message.content.strip():
            return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error in call_large_model as {e}")
        return {}


def call_large_model(
    messages, api_key="EMPTY", base_url=None, model=None, version="2024-08-01-preview", video_path=None
):
    if "azure" in base_url:
        client = AzureOpenAI(api_key=api_key, base_url=base_url, api_version=version)
    else:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=100000)
    if video_path and os.path.exists(video_path):
        if "glm" in model:
            with open(video_path, "rb") as video_file:
                video_base = base64.b64encode(video_file.read()).decode("utf-8")
        video_path = "file://" + video_path
        for message in messages:
            if message["role"] == "user":
                message["content"] = [
                    {"type": "text", "text": message["content"]},
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": video_base if "glm" in model else video_path,
                        },
                    },
                ]
    for i in range(3):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=8192,
            stream=False,
        )
        if response.choices[0].message.content.strip():
            return response.choices[0].message.content.strip()

    return {}


def call_embedding(texts, api_key, base_url, model="embedding-3"):
    if isinstance(texts, list):
        texts = [" " if text == "" else text for text in texts]

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.embeddings.create(model=model, input=texts)
    embeddings = []
    for item in response.data:
        emb = np.array(item.embedding, dtype="float32")
        embeddings.append(emb)
    return embeddings


def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def clean_response(response_str):
    response_str = response_str.strip()
    if response_str.startswith("```") and response_str.endswith("```"):
        lines = response_str.split("\n")
        lines = lines[1:-1]
        response_str = "\n".join(lines).strip()
    return response_str


def parse_json_response(response):
    response_str = clean_response(response)

    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        pass

    json_block_match = re.search(r"```json\s*\n([\s\S]*?)\n```", response_str)
    if json_block_match:
        json_str = json_block_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    arr_match = re.search(r"\[[\s\S]*\]", response_str)
    if arr_match:
        arr_str = arr_match.group(0).strip()
        try:
            return json.loads(arr_str)
        except json.JSONDecodeError:
            pass

    obj_match = re.search(r"\{[\s\S]*\}", response_str)
    if obj_match:
        obj_str = obj_match.group(0).strip()
        try:
            return json.loads(obj_str)
        except json.JSONDecodeError:
            pass

    messages = [
        {
            "role": "system",
            "content": """
你将看到一个JSON输入 ，它包含一些格式错误。请帮助我修复它。这不是一个list，因此一定是单个json。
确保所有的键值对格式正确，并且所有括号和逗号都完整无误。返回修复后的有效JSON。

## 输出格式
```json
修复后的json
```

## 注意：

- 有时候，原始的输入包含不规则的符号或者连续超过100个数字，则应该把这个字段设为[]或者{}，因为这是输入错误。
- 不要输出额外内容。
- 所有输入的json冲重复的内容都需要删除。仅保留最简单的json，不要超过5000个token。

""",
        },
        {"role": "user", "content": f"原始JSON: \n{response_str}\n\n修复后的JSON:"},
    ]

    fixed_json_str = call_large_model_llm(
        messages, model="glm-4-plus", base_url="https://open.bigmodel.cn/api/paas/v4"
    )
    json_block_match = re.search(r"```json\s*\n([\s\S]*?)\n```", fixed_json_str)
    if json_block_match:
        json_str = json_block_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    return [{}]
