import json
import re
import os
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from llama_cpp import Llama

# Configurable Constants
STRIP_MARGIN = 60  # Pixels to ignore at top and bottom (Headers/Footers)
MIN_WORD_COUNT = 10  # Minimum words to consider a page "not blank"

@dataclass
class PageData:
    page_number: int
    chunk_id: str
    source_file: str
    headings: List[str] = field(default_factory=list)
    paragraphs: List[str] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    document_text: str = ""
    summary: str = ""
    meta: Dict[str, Any] = field(default_factory=lambda: {"ocr": False, "is_blank": False})

class UltraPDFParser:
    def __init__(self, pdf_path: str, model_path: str):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path("output")
        self.image_dir = self.output_dir / "extracted_images"
        
        # Directory Management
        self.output_dir.mkdir(exist_ok=True)
        self.image_dir.mkdir(exist_ok=True)
        
        print(f"Initializing Local LLM (GGUF)...")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=35,
            n_ctx=4096,
            verbose=False
        )
        self.pages_data = []
        self.global_acronyms = {}

    def discover_acronyms(self, text: str):
        """Asks the LLM to find acronyms in a block of text and stores them."""
        if len(text.strip()) < 50: return
        
        prompt = f"""<|im_start|>system
Extract technical acronyms and their full forms from the text. 
Return ONLY a valid JSON object. Example: {{"RAM": "Random Access Memory"}}<|im_end|>
<|im_start|>user
Text: {text[:1000]}
JSON:<|im_end|>
<|im_start|>assistant
"""
        try:
            response = self.llm(prompt, max_tokens=200, temperature=0.1)
            raw = response['choices'][0]['text'].strip()
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                new_data = json.loads(match.group())
                self.global_acronyms.update(new_data)
        except:
            pass

    def apply_clean_and_expand(self, text: str) -> str:
        """Cleans watermarks and expands discovered acronyms using Regex."""
        # 1. Remove Watermarks
        text = re.sub(r'(CONFIDENTIAL|DRAFT|INTERNAL USE ONLY)', '', text, flags=re.IGNORECASE)
        
        # 2. Expand Acronyms (using \b to avoid partial word matches)
        for short, long in self.global_acronyms.items():
            if len(short) > 1: # Avoid single letter 'acronyms'
                text = re.sub(rf'\b{short}\b', f"{short} ({long})", text)
        return text.strip()

    def get_image_context(self, page_mupdf, img_rect) -> Tuple[str, str]:
        """Spatial search for text surrounding an image."""
        blocks = page_mupdf.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))
        
        before, after = [], []
        for b in blocks:
            b_text = b[4].strip()
            b_y_mid = (b[1] + b[3]) / 2
            if b_y_mid < img_rect[1]: before.append(b_text)
            elif b_y_mid > img_rect[3]: after.append(b_text)
        
        return " ".join(before)[-300:], " ".join(after)[:300]

    def extract_refined_tables(self, page_plumber) -> List[Dict]:
        """Extracts tables with settings to prevent fragmentation."""
        ts = {"vertical_strategy": "lines", "horizontal_strategy": "lines", "snap_tolerance": 3}
        tables = page_plumber.extract_tables(ts)
        output = []
        for t in tables:
            if t and len(t) > 1:
                clean_t = [[self.apply_clean_and_expand(str(c)) if c else "" for c in r] for r in t]
                output.append({"header": clean_t[0], "rows": clean_t[1:]})
        return output

    def parse_page(self, i, page_plumber, page_mupdf):
        text_full = page_plumber.extract_text() or ""
        
        # Skip blank pages
        if len(text_full.split()) < MIN_WORD_COUNT:
            return None

        # 1. Acronym Discovery (First 10 pages for efficiency)
        if i < 10: self.discover_acronyms(text_full)

        # 2. Extract Headings & Paragraphs with spatial filtering
        headings, paragraphs = [], []
        p_height = page_plumber.height
        words = page_plumber.extract_words()
        
        lines = {}
        for w in words:
            # Filter Headers/Footers
            if STRIP_MARGIN < w['top'] < (p_height - STRIP_MARGIN):
                y = round(w['top'])
                lines.setdefault(y, []).append(w)

        for y in sorted(lines.keys()):
            line_txt = self.apply_clean_and_expand(" ".join([w['text'] for w in lines[y]]))
            if line_txt.isupper() or re.match(r'^\d+(\.\d+)*', line_txt):
                headings.append(line_txt)
            elif len(line_txt) > 30:
                paragraphs.append(line_txt)

        # 3. Images
        images_data = []
        for img_idx, img in enumerate(page_mupdf.get_images(full=True), 1):
            xref = img[0]
            pix = page_mupdf.parent.extract_image(xref)
            fname = f"page_{i+1}_img_{img_idx}.{pix['ext']}"
            fpath = self.image_dir / fname
            with open(fpath, "wb") as f: f.write(pix["image"])
            
            # Get Context
            rects = page_mupdf.get_image_rects(xref)
            rect = rects[0] if rects else [0,0,0,0]
            cb, ca = self.get_image_context(page_mupdf, rect)
            
            images_data.append({
                "image_index": img_idx,
                "filename": fname,
                "file_path": str(fpath),
                "context_before": cb,
                "context_after": ca,
                "llm_description": "Inferred from context." # Can be LLM-generated
            })

        return PageData(
            page_number=i+1,
            chunk_id=f"page_{i+1}",
            source_file=self.pdf_path.name,
            headings=headings,
            paragraphs=paragraphs,
            tables=self.extract_refined_tables(page_plumber),
            images=images_data,
            document_text=" ".join(paragraphs)
        )

    def run(self):
        doc_m = fitz.open(self.pdf_path)
        doc_p = pdfplumber.open(self.pdf_path)
        
        try:
            print(f"Processing: {self.pdf_path.name}")
            for i in range(len(doc_p.pages)):
                data = self.parse_page(i, doc_p.pages[i], doc_m[i])
                if data: self.pages_data.append(data)
                print(f"Progress: {i+1}/{len(doc_p.pages)} pages", end='\r')
            
            # Final Export
            with open(self.output_dir / "output.json", "w", encoding="utf-8") as f:
                json.dump([asdict(p) for p in self.pages_data], f, indent=2)
            print(f"\nCompleted! JSON saved to {self.output_dir}/output.json")
            
        finally:
            doc_m.close()
            doc_p.close()

if __name__ == "__main__":
    PDF_FILE = "input.pdf"
    MODEL_FILE = "models/qwen2.5-1.5b-instruct-q5_k_m.gguf"
    
    parser = UltraPDFParser(PDF_FILE, MODEL_FILE)
    parser.run()