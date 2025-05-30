import os

# 根目录路径
root_dir = "outputs"

# 需要匹配的错误字符串
error_substring = "Error code: 404 - {'message': 'The model `Qwen/Qwen2.5-72B-Instruct` does not exist.'"

# 遍历所有子目录和文件
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".json"):
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if error_substring in content:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
