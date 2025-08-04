import re
import json


def extract_last_text_value(input_file, output_file="ai_goods_report.json"):
    try:
        # 1. 读取文件内容
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 2. 正则匹配所有text字段内容（处理转义字符和跨多行）
        # 匹配 "text": "```json ... ```" 结构，兼容转义符
        pattern = r'"text":\s*"```json(.*?)```"'
        matches = re.findall(pattern, content, re.DOTALL | re.UNICODE)

        if not matches:
            raise ValueError("未找到匹配的text字段内容")

        # 3. 取最后一个匹配项，处理转义字符（关键改进）
        last_match = matches[-1].strip()
        # 替换转义序列：将\\n转为实际换行，\\"转为"
        processed_content = last_match.replace("\\n", "\n").replace('\\"', '"')

        # 4. 解析为JSON并格式化（确保格式正确）
        try:
            json_data = json.loads(processed_content)
            formatted_json = json.dumps(json_data, ensure_ascii=False, indent=2)
        except json.JSONDecodeError as e:
            print(f"JSON解析警告: {e}，将使用原始处理内容")
            formatted_json = processed_content

        # 5. 控制台输出（标准格式）
        print("提取并格式化后的内容：\n")
        print(formatted_json)

        # 6. 保存到文件
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatted_json)

        print(f"\n成功保存到 {output_file}")

    except FileNotFoundError:
        print(f"错误：输入文件 {input_file} 未找到")
    except Exception as e:
        print(f"处理失败：{str(e)}")


if __name__ == "__main__":
    input_file_path = "raw_server_events.json"  # 替换为你的文件路径
    extract_last_text_value(input_file_path)