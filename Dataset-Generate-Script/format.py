import json

with open("output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to the new format
converted_data = []
for item in data:
    for message in item["messages"]:
        if message["role"] == "user":
            instruction = message["content"]
        elif message["role"] == "assistant":
            output = message["content"]
            converted_data.append({"instruction": instruction, "output": output})

with open("new.json", "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

print("Conversion complete! Output saved to new.json")
