import json
import re
import os
import logging
import fitz
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from llama_cpp import Llama
from tqdm import tqdm
from collections import Counter

# --- Configuration ---
STRIP_MARGIN = 50
MIN_TEXT_DENSITY = 10

logging.basicConfig(filename='parser.log', level=logging.INFO)

class UltimateHybridParser:
    def __init__(self, pdf_path: str, model_path: str):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path("output")
        self.image_dir = self.output_dir / "extracted_images"
        self.output_dir.mkdir(exist_ok=True)
        self.image_dir.mkdir(exist_ok=True)
        
        # Initialize LLM with a larger context for layout analysis
        self.llm = Llama(model_path=model_path, n_gpu_layers=35, n_ctx=8192, verbose=False)
        self.pages_data = []

    def get_layout_audit(self, lines: List[Dict]) -> Dict[str, str]:
        """
        Sends the visual structure of the page to the LLM.
        The LLM identifies which lines are Headings and which are Subheadings.
        """
        # Prepare a lightweight representation of the page for the LLM
        layout_map = ""
        for i, l in enumerate(lines):
            style = "BOLD" if l['is_bold'] else "REGULAR"
            layout_map += f"ID:{i} | Size:{l['size']} | Style:{style} | Text: {l['text']}\n"

        prompt = f"""<|im_start|>system
You are a document structure expert. Analyze the visual metadata (Size/Style) and text content.
Identify the hierarchy. Return ONLY a JSON object mapping IDs to types: "H1", "H2", or "P" (Paragraph).
Example: {{"0": "H1", "1": "P", "5": "H2"}}<|im_end|>
<|im_start|>user
{layout_map[:3000]}
JSON Map:<|im_end|>
<|im_start|>assistant
"""
        try:
            res = self.llm(prompt, max_tokens=500, temperature=0.1)
            match = re.search(r'\{.*\}', res['choices'][0]['text'], re.DOTALL)
            return json.loads(match.group()) if match else {}
        except:
            return {}

    def parse_page(self, i, page_plumber, page_mupdf):
        p_height = page_plumber.height
        
        # 1. Collect lines with metadata
        lines_raw = []
        lines_map = {}
        for obj in page_plumber.extract_words(extra_attrs=['size', 'fontname']):
            if STRIP_MARGIN < obj['top'] < (p_height - STRIP_MARGIN):
                lines_map.setdefault(round(obj['top']), []).append(obj)

        for y in sorted(lines_map.keys()):
            objs = lines_map[y]
            lines_raw.append({
                "text": " ".join([o['text'] for o in objs]),
                "size": round(objs[0]['size'], 1),
                "is_bold": any('bold' in o['fontname'].lower() for o in objs),
                "top": y
            })

        if not lines_raw: return None

        # 2. LLM Layout Audit
        structure_map = self.get_layout_audit(lines_raw)

        # 3. Assemble Hierarchical JSON
        structured_content = []
        current_h1 = None
        current_h2 = None

        for idx, line in enumerate(lines_raw):
            l_type = structure_map.get(str(idx), "P")
            text = line['text']

            if l_type == "H1":
                current_h1 = {"heading": text, "subheadings": []}
                structured_content.append(current_h1)
                current_h2 = None
            elif l_type == "H2":
                current_h2 = {"subheading": text, "content": []}
                if current_h1:
                    current_h1['subheadings'].append(current_h2)
                else:
                    structured_content.append(current_h2)
            else:
                # Content (Paragraph)
                if current_h2:
                    current_h2['content'].append(text)
                elif current_h1:
                    # Content directly under H1
                    if "content" not in current_h1: current_h1["content"] = []
                    current_h1["content"].append(text)
                else:
                    structured_content.append({"text": text})

        # 4. Table & Image Integration (Spatial Anchoring)
        tables = [{"header": t[0], "rows": t[1:]} for t in page_plumber.extract_tables() if t and len(t) > 1]
        
        return {
            "page": i + 1,
            "hierarchy": structured_content,
            "tables": tables,
            "images": self.extract_images(i, page_mupdf)
        }

    def extract_images(self, page_num, page_mupdf):
        img_data = []
        for idx, img in enumerate(page_mupdf.get_images(), 1):
            xref = img[0]
            pix = page_mupdf.parent.extract_image(xref)
            fname = f"p{page_num+1}_img{idx}.{pix['ext']}"
            with open(self.image_dir / fname, "wb") as f: f.write(pix["image"])
            img_data.append({"filename": fname, "size": f"{pix['width']}x{pix['height']}"})
        return img_data

    def run(self):
        doc_m = fitz.open(self.pdf_path)
        doc_p = pdfplumber.open(self.pdf_path)
        results = []
        try:
            for i in tqdm(range(len(doc_p.pages)), desc="Hybrid Extraction"):
                data = self.parse_page(i, doc_p.pages[i], doc_m[i])
                if data: results.append(data)
            
            with open(self.output_dir / "output.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        finally:
            doc_m.close()
            doc_p.close()

if __name__ == "__main__":
    parser = UltimateHybridParser("input.pdf", "models/qwen2.5-1.5b-instruct-q5_k_m.gguf")
    parser.run()