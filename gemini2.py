import json
import re
import os
import fitz
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from llama_cpp import Llama
from tqdm import tqdm
from collections import Counter

@dataclass
class Section:
    title: str
    level: int
    content: List[Any] = field(default_factory=list) # Can contain strings (paragraphs) or other Sections

class NestingUltraParser:
    def __init__(self, pdf_path: str, model_path: str):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path("output")
        self.image_dir = self.output_dir / "extracted_images"
        self.output_dir.mkdir(exist_ok=True)
        self.image_dir.mkdir(exist_ok=True)
        
        # Load local LLM
        self.llm = Llama(model_path=model_path, n_gpu_layers=35, n_ctx=4096, verbose=False)
        self.pages_data = []

    def get_font_stats(self, page_plumber) -> float:
        char_sizes = [round(char['size'], 1) for char in page_plumber.chars]
        return Counter(char_sizes).most_common(1)[0][0] if char_sizes else 10.0

    def parse_page(self, i, page_plumber, page_mupdf):
        text_full = page_plumber.extract_text() or ""
        if len(text_full.split()) < 10: return None

        body_font_size = self.get_font_stats(page_plumber)
        p_height = page_plumber.height
        
        # 1. Line Grouping
        lines_data = {}
        for obj in page_plumber.extract_words(extra_attrs=['size', 'fontname']):
            if 50 < obj['top'] < (p_height - 50): # Header/Footer clip
                lines_data.setdefault(round(obj['top']), []).append(obj)

        # 2. Hierarchical Processing with a Stack
        # Root section for the page
        root = Section(title=f"Page {i+1} Root", level=0)
        stack = [root]

        for y in sorted(lines_data.keys()):
            line_objs = lines_data[y]
            line_text = " ".join([o['text'] for o in line_objs]).strip()
            avg_size = sum([o['size'] for o in line_objs]) / len(line_objs)
            is_bold = any('bold' in o['fontname'].lower() for o in line_objs)
            
            # Determine Heading Level
            current_level = None
            if avg_size > (body_font_size + 2.5) or re.match(r'^(UNIT|CHAPTER|SECTION)\b', line_text, re.I):
                current_level = 1
            elif (is_bold and re.match(r'^\d+(\.\d+)+', line_text)) or (is_bold and avg_size > body_font_size):
                current_level = 2
            elif is_bold and len(line_text.split()) < 8:
                current_level = 3

            if current_level:
                new_section = Section(title=line_text, level=current_level)
                
                # Pop stack until we find the parent (level < current_level)
                while len(stack) > 1 and stack[-1].level >= current_level:
                    stack.pop()
                
                stack[-1].content.append(new_section)
                stack.append(new_section)
            else:
                # It's a paragraph, fix hyphenation and add to current active section
                clean_para = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', line_text)
                if len(clean_para) > 5:
                    stack[-1].content.append(clean_para)

        # 3. Post-Process Content (Join lines into sentences)
        self.reconstruct_sentences(root)

        # 4. Final Data Assembly
        return {
            "page_number": i+1,
            "structured_content": asdict(root)["content"],
            "tables": [{"header": t[0], "rows": t[1:]} for t in page_plumber.extract_tables() if t and len(t) > 1],
            "summary": self.generate_summary(text_full[:2000])
        }

    def reconstruct_sentences(self, section: Section):
        """Recursively joins strings in content into full sentences."""
        new_content = []
        temp_text = []
        
        for item in section.content:
            if isinstance(item, str):
                temp_text.append(item)
            else:
                # Before moving to a new section, flush gathered text as sentences
                if temp_text:
                    blob = " ".join(temp_text)
                    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', blob)
                    new_content.extend([s.strip() for s in sentences if len(s.strip()) > 5])
                    temp_text = []
                # Recursively process the sub-section
                self.reconstruct_sentences(item)
                new_content.append(item)
        
        # Final flush
        if temp_text:
            blob = " ".join(temp_text)
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', blob)
            new_content.extend([s.strip() for s in sentences if len(s.strip()) > 5])
            
        section.content = new_content

    def generate_summary(self, text: str) -> str:
        prompt = f"<|im_start|>system\nSummarize the page content concisely.<|im_end|>\n<|im_start|>user\n{text}\nSummary:<|im_end|>\n<|im_start|>assistant\n"
        res = self.llm(prompt, max_tokens=100, stop=["<|im_end|>"])
        return res['choices'][0]['text'].strip()

    def run(self):
        doc_m = fitz.open(self.pdf_path)
        doc_p = pdfplumber.open(self.pdf_path)
        try:
            for i in tqdm(range(len(doc_p.pages)), desc="Nested Parsing"):
                page_data = self.parse_page(i, doc_p.pages[i], doc_m[i])
                if page_data: self.pages_data.append(page_data)
                
            with open(self.output_dir / "output.json", "w", encoding="utf-8") as f:
                json.dump(self.pages_data, f, indent=2, ensure_ascii=False)
        finally:
            doc_m.close()
            doc_p.close()

if __name__ == "__main__":
    parser = NestingUltraParser("input.pdf", "models/qwen2.5-1.5b-instruct-q5_k_m.gguf")
    parser.run()