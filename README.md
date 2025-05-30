# Genesis: Multimodal Multi-Party Dataset for Emotional Causal Analysis

## 快速开始

### 环境准备

确保你拥有至少一张的 NVIDIA 显卡，使用`python>=3.12`版本。 接着运行

```shell
pip install -r requirements.txt
```

## 将字幕转为JSON格式

如果你有字幕文件（如.txt格式），可以使用以下脚本将其转换为JSON格式：

```shell
python convert_txt_json.py --input context --output texts
```

这将会读取`context`目录下的字幕文件，并将其转换为JSON格式，保存到`texts`目录下。为若干个`chat_*.json`文件。

```json
[
  {
    "sentence": "本场的辩题是，如果注定无法成功，该不该继续努力？",
    "Holder": "2",
    "Target": "",
    "Aspect": "",
    "Opinion": "",
    "Sentiment": "",
    "Rationale": "",
    "time": "00:00"
  }
]
```

### 使用VLM 情绪分析(不借助五元组)

1. 将所有字幕放到`texts`目录下，视频放到`videos`目录下，确保视频的命名格式为`chat_*.mp4`，文字的命名格式为`chat_*.json`。
   对于字幕/五元组，格式如下:

在程序运行时，仅读入`sentence`,`Holder`和`time`字段,其中`time`为分和秒。构成以下格式的字幕：

```json
"2": "本场的辩题是，如果注定无法成功，该不该继续努力？" (00: 00)
```

2. 启动大模型服务

首先，需要在`config_example.yaml`中配置模型路径和其他参数。包括大模型的Keys等信息，并改名为`config.yaml`。接着，运行vLLM来启动模型推理。

```shell
vllm  serve Qwen/Qwen2.5-VL-7B-Instruct  --allowed-local-media-path videos
```

3. 启动推理服务

```shell
python get_emo_vlm.py --llm_model qwen2.5-7b --config_path config.json --input_dir texts --video_dir videos --output_dir outputs
```

这将会保存所有的结果到`outputs/qwen2.5vl-7b`目录下。每个视频包含一个JSON文件`emotions_*.json`，格式如下：

```json
{
  "1": {
    "events": [
      {
        "event": "讨论资源分配与努力的价值",
        "emotions": [
          {
            "source_id": 1,
            "state": "positive",
            "reason": "认为努力的价值在于探索未知，而非成功的结果"
          },
          {
            "source_id": 1,
            "state": "negative",
            "reason": "反驳对方关于注定无法成功的人不应继续努力的观点"
          }
        ]
      }
    ]
  }
}
```
