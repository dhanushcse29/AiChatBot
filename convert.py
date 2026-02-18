import json
import os

def convert_to_qwen_format(input_file, output_file):
    """
    Convert Q&A JSON to Qwen 2.5 chat format
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]

    
    if isinstance(data, dict):
        data = [data]
    
    converted_data = []
    
    for item in data:
        # Qwen 2.5 chat format
        conversation = [
            {
                "role": "system",
                "content": "You are a technical assistant specializing in EW systems. Answer questions accurately and concisely based on your knowledge."
            },
            {
                "role": "user",
                "content": item['question']
            },
            {
                "role": "assistant",
                "content": item['answer']
            }
        ]
        
        converted_data.append({
            "messages": conversation,
            "id": item.get("id", ""),
            "difficulty": item.get("difficulty", ""),
            "tags": item.get("tags", [])
        })
    
    # Save as JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in converted_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✅ Converted {len(converted_data)} examples for Qwen 2.5")
    print(f"📁 Saved to: {output_file}")

if __name__ == "__main__":
    INPUT_FILE = r"C:/Users/pooji\OneDrive\Desktop\using_chroma-main\data\qa.json"
    OUTPUT_FILE = r"C:\Users\pooji\OneDrive\Desktop\using_chroma-main\data\qa_pairs_qwen.jsonl"
    
    convert_to_qwen_format(INPUT_FILE, OUTPUT_FILE)