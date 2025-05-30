import argparse
import json
import os
import re


def parse_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.lstrip("\t").strip() for line in content.split("\n")]

    result = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if "原文" in line or (("年" in line or "月" in line or "日" in line) and len(line) > 10):
            i += 1
            continue

        speaker_match = re.match(r"发言人(\d+)\s+(\d{2}:\d{2})", line)
        if speaker_match:
            speaker_id = speaker_match.group(1)
            time = speaker_match.group(2)
            sentence_parts = []
            i += 1

            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    i += 1
                    continue
                if re.match(r"发言人\d+\s+\d{2}:\d{2}", next_line):
                    break
                if "原文" in next_line or (
                    ("年" in next_line or "月" in next_line or "日" in next_line) and len(next_line) > 10
                ):
                    i += 1
                    continue

                sentence_parts.append(next_line)
                i += 1
            if sentence_parts:
                sentence = "".join(sentence_parts)
                if sentence.strip():
                    entry = {
                        "sentence": sentence,
                        "Holder": speaker_id,
                        "Target": "",
                        "Aspect": "",
                        "Opinion": "",
                        "Sentiment": "",
                        "Rationale": "",
                        "time": time,
                    }
                    result.append(entry)
        else:
            i += 1

    return result


def convert_txt_to_json(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            json_data = parse_txt_file(input_path)
            output_filename = filename.replace(".txt", ".json")
            output_path = os.path.join(output_folder, output_filename)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Convert txt chat files to JSON format")
    parser.add_argument("--input", default="context", help="Input folder containing txt files")
    parser.add_argument("--output", default="texts", help="Output folder for JSON files")
    args = parser.parse_args()
    convert_txt_to_json(args.input, args.output)


if __name__ == "__main__":
    main()
