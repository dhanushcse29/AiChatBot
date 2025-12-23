import fitz  # PyMuPDF
import json
import re
import random


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_into_chunks(text, min_length=40):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) >= min_length]


def generate_question(sentence):
    templates = [
        "Explain {}",
        "What is {}?",
        "Describe {}",
        "Define {}",
        "Give details about {}"
    ]
    key_phrase = " ".join(sentence.split()[:6])
    return random.choice(templates).format(key_phrase)


def generate_dataset(text, num_pairs):
    chunks = split_into_chunks(text)

    if num_pairs > len(chunks):
        raise ValueError("Requested more questions than content available.")

    selected_chunks = random.sample(chunks, num_pairs)

    dataset = []
    for chunk in selected_chunks:
        dataset.append({
            "instruction": generate_question(chunk),
            "output": chunk
        })

    return dataset


def save_to_json(data, file_name="train.json"):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    pdf_file = "input.pdf"

    print(" Reading PDF...")
    raw_text = extract_text_from_pdf(pdf_file)

    print(" Cleaning text...")
    cleaned_text = clean_text(raw_text)

    num_questions = int(input("Enter number of Q&A pairs to generate: "))

    print(" Generating dataset...")
    dataset = generate_dataset(cleaned_text, num_questions)

    save_to_json(dataset)

    print(f"\n {num_questions} instruction-output pairs saved to train.json")

if __name__ == "__main__":
    main()
