import argparse
import concurrent.futures
import json
import os

from tqdm import tqdm

from utils import call_large_model, load_yaml_config, parse_json_response


system_prompt = r"""
你是一名情绪事件分析助手。我将会给你一段视频，以及不同时间的字幕，包含说话人，说话内容和时间(分:秒)，我需要你根据我提供的视频和字幕，推断出该角色最重要、最核心的情绪事件。

字幕和视频应该一起参考，并做到：
1. 主要参考视频中的内容，字幕只是参考！视频中人物的表情变化才是重点，视频中的权重应该远远大于文字。
2. 一段对话中完整的历史，包含说话当事人和其他人，这些都是口语对话，因此语言非常口语化。一个人可能多多段连续的说话，这在日常对话中是很正常的。

输出格式要求:
1. 按照 JSON 格式输出，并且输出的最外层结构只包含当前角色一个 key，例如 `"1": { "events": [...] }`。
2. `"events` 为一个列表，每个元素包含 `"event"`（事件名称）和 `"emotions"`（角色针对该事件的情绪列表）。
3. `emotions` 是一个列表，每个元素至少包含：`{"state": "...", "reason": "..."}`。
   - `state` 必须在 `["positive", "negative", "neutral", "ambiguous", "doubt"]` 中选择。
   - `reason` 描述角色产生此情绪的原因或依据。
4. 一个人可能多有多个事件，如果这个事件是独立的，则必须单独展开。通常情况下，每个人都会有至少一个事件，因此不能输出空白。

具体格式如下输出格式如下:
```json
{
    "角色ID_1": {
        "events": [
            {
                "event": 事件名称,
                "emotions": [
                    {
                        "source_id": 角色ID(只有一个, 一个数字)，如果这个情感变化是由于某一个角色（一定是角色ID）导致，则输出对应角色ID。如果没有指定，或者由于上下文理解为自己的情感变化，则为自己的角色ID。这个句子一定出现在所有的holders内。
                        "state": "positive/negative/neutral/ambiguous/doubt",
                        "reason": 该事件产生该情绪的原因
                    }
                ]
            }
        ]
    },
    "角色ID_2": {
        ... 相同的结构
    }
}
```

我会给你提供一个例子,示例，我的输入会和这个例子相同
## 例子：
<VIDEO>
[视频内容如上所示，下面是视频字幕]
[
"1": "这口服液他坚决不能喝，整个一套它就是一欺诈行为，" (00:30)
"2": "我给孩子们弄点课程，让孩子们喝点口服液，我就成欺诈行为了。孩子们的事你就会在那和稀泥，他们犯了错你就可以跟他讲大道理，你这是啥行为？虚伪，" (00:40)
"1": "我为孩子好我还虚伪了."  (00:45)
"2": "这满嘴大道理不讲一点实际用处的行为就是虚伪，他们考不上大学咋办？一个个没出息去。" (00:47)
"1": "采油，怎么？油田怎么了？"  (00:51)
"1": "这些年我这俩孩子都是这油田养的，你别把自己办不成的事加到孩子身上行不行？" (00:53)
"1": "干嘛非逼着俩孩子去大城市，咱们是什么情况，咱们就过什么日子好不好？" (00:59)
"1": "孩子们有他们自己的命运对吧？你别非得他们要怎么样行不行？" (01:10)
"1": "我的意见。" (01:11)
]

此时如果要分析「说话人1」的情绪事件，可输出：

[示例输出]
{
  "1": {
    "events": [
      {
        "event": "质疑口服液",
        "emotions": [
          {
            "source_id": 2
            "state": "negative",
            "reason": "认为口服液是骗人的，没有效果"
          }
        ]
      }
    ]
  }
}

### 注意
- 同一事件中若情绪未发生变化，不应重复记录。
- 一个事件的情绪变化一般不会超过 2 次。你应该给出尽量少的event事件，具体的进展和情绪变化都放在emotions的state和reason字段中，event只是一个简单的大事件概述，不超过10个字。
- 必须根据事件主题，合并情绪变化的原因。一个事件通常很大，小的原因和事件进展都应该归类在 reason。
- 请注意，必须根据示例输出的结构输出， "1" 作为对应说话人的 ID，events 列表中每个事件都包含 event 和 emotions 字段，角色ID_1必须输出。

""".strip()

user_prompt_template = r"""
[视频内容如上所示，下面是视频字幕]
{chat_history}

我希望你根据上述的视频和字幕，，只分析说话人 {holder_id} 的情绪事件。请忽略其他说话人，不要输出与 {holder_id} 无关的事件。按照系统提示的格式要求输出。
""".strip()


def process_single_file(
    fname, input_dir, output_dir, video_dir, system_prompt, user_prompt_template, api_key, base_url, model_name
):
    try:
        out_fname = fname.replace("chat_", "emotions_")
        out_path = os.path.join(output_dir, out_fname)
        if os.path.exists(out_path):
            return f"Skipped: {fname}"

        file_path = os.path.join(input_dir, fname)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        video_name = fname.replace(".json", ".mp4")
        video_path = os.path.join(video_dir, video_name)

        lines = []
        holders_this_file = set()

        for idx, item in enumerate(data):
            sentence = item.get("sentence", "")
            holder = item.get("Holder", "")
            time = item.get("time", "")
            if holder and sentence:
                line_str = f'({idx + 1}) "{holder}": "{sentence}" ({time})'
                lines.append(line_str)
                holders_this_file.add(holder)

        chat_history_str = "[\n" + ",\n".join(lines) + "\n]"
        file_result = {}

        for holder_id in sorted(holders_this_file):
            try:
                user_prompt = user_prompt_template.format(holder_id=holder_id, chat_history=chat_history_str)
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                response_text = call_large_model(
                    messages, api_key=api_key, base_url=base_url, model=model_name, video_path=video_path
                )
                parsed_json = parse_json_response(response_text)
                if isinstance(parsed_json, dict) and holder_id in parsed_json:
                    file_result[holder_id] = parsed_json[holder_id]
                else:
                    file_result[holder_id] = {"events": []}
            except Exception as e:
                file_result[holder_id] = {"events": [], "error": str(e)}

        has_valid_results = False
        for holder_id, result in file_result.items():
            if "error" not in result and result.get("events") and len(result["events"]) > 0:
                has_valid_results = True
                break

        if has_valid_results:
            with open(out_path, "w", encoding="utf-8") as outf:
                json.dump(file_result, outf, ensure_ascii=False, indent=2)
            return f"Processed: {fname}"
        else:
            return f"No valid results: {fname}"

    except Exception as e:
        return f"Error: {fname} - {str(e)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="texts", help="Path to input JSON dir")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Path to save final JSON")
    parser.add_argument(
        "--video_dir", type=str, default="/mnt/paper/EmoCaustics_DB/videos", help="Path to video files"
    )
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--llm_model", type=str, required=True, help="LLM model name in config")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    llm_cfg = load_yaml_config(args.config_path, args.llm_model, "llm_config")
    api_key = llm_cfg["api_key"]
    base_url = llm_cfg["base_url"]
    model_name = llm_cfg["model"]

    model_output_dir = os.path.join(args.output_dir, args.llm_model)
    os.makedirs(model_output_dir, exist_ok=True)

    all_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".json"))

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch) as executor:
        for fname in all_files:
            futures.append(
                executor.submit(
                    process_single_file,
                    fname,
                    args.input_dir,
                    model_output_dir,
                    args.video_dir,
                    system_prompt,
                    user_prompt_template,
                    api_key,
                    base_url,
                    model_name,
                )
            )

        with tqdm(total=len(futures), desc="Processing files") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                pbar.update(1)
                if result:
                    if "Skipped" in result:
                        pbar.write(result)
                    elif "Error" in result:
                        pbar.write(f"❌ {result}")
                    elif "No valid results" in result:
                        pbar.write(f"⚠️ {result}")


if __name__ == "__main__":
    main()
