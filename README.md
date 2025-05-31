# Genesis: Multimodal Multi-Party Dataset for Emotional Causal Analysis

> Dataset: [🤗 Huggingface](https://huggingface.co/datasets/zRzRzRzRzRzRzR/Genesis_Dataset), [🤖 ModelScope](https://modelscope.cn/datasets/zRzRzRzRzRzRzR/Genesis_Dataset)
>
> Paper: [arXiv (Comming Soon)]()

## Dataset Format

The dataset contains two folders: `texts` and `videos`.The `texts` folder stores subtitle files along with emotion
five-tuples for each sentence.The `videos` folder stores the corresponding video files.Each video corresponds to one
subtitle file, with the naming format `chat_*.json` and `chat_*.mp4`.Each sentence in the subtitle file follows the
format below:

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

## Quick Start

### Environment Setup

Make sure you have at least one NVIDIA GPU and are using `python>=3.12`. Then run:

```shell
pip install -r requirements.txt
```

### Using VLM for Sentiment Analysis (Without Five-Tuple Annotations)

1. Place all subtitle files into the `texts` directory and videos into the `videos` directory.Ensure video files follow
   the naming format `chat_*.mp4` and text files follow `chat_*.json`.

   For subtitles/five-tuples, use the following format:

   During program execution, only the `sentence`, `Holder`, and `time` fields are read.
   The `time` field should be in minutes and seconds format.
   Subtitles should follow this structure:

    ```json
    "2": "本场的辩题是，如果注定无法成功，该不该继续努力？" (00: 00)
    ```

2. Start the Large Model Service

   First, configure the model path and other parameters in `config_example.yaml`,
   including keys and other information required for the large model.
   Rename the file to `config.yaml` once editing is complete.
   Then, launch vLLM to start model inference.

    ```shell
    vllm  serve Qwen/Qwen2.5-VL-7B-Instruct  --allowed-local-media-path videos
    ```

3. Start Inference

    ```shell
    python get_emo_vlm.py --llm_model qwen2.5-7b --config_path config.json --input_dir texts --video_dir videos --output_dir outputs
    ```

   This will save all results to the `outputs/qwen2.5vl-7b` directory.
   Each video will generate a JSON file named `emotions_*.json`, with the following format:

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
