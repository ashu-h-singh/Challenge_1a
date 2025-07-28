import re
import json
import os
import pdfplumber
import numpy as np
from collections import defaultdict
import spacy
from langdetect import detect
from sklearn.cluster import KMeans
import sys
import time
import multiprocessing
from functools import partial

class OutlineExtractor:
    def __init__(self, language=None):
        self.language = language or 'auto'
        if self.language == 'en':
            model_name = "en_core_web_sm"
        elif self.language == 'ja':
            model_name = "ja_core_news_sm"
        else:
            model_name = "xx_ent_wiki_sm"

        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Warning: Failed to load '{model_name}', falling back to 'xx_ent_wiki_sm'")
            self.nlp = spacy.load("xx_ent_wiki_sm")

        self.heading_patterns_en = [
            r"^\s*(?:Chapter|CHAPTER)\s+\d+[\.:)]?\s+(.+)$",
            r"^\s*\d+\.\d+\.?\s+(.+)$",
            r"^\s*\d+\.\s+(.+)$",
            r"^\s*[A-Z][\w\s]+$",
            r"^\s*[IVXLCDM]+\.\s+(.+)$",
            r"^\d+(\.\d+)*\s+.+$",
            r"^[A-Z][A-Za-z\s\-]{3,}$"
        ]

        self.heading_patterns_ja = [
            r"^\u6587\s*\u6cd5\s*\d+",
            r"^[\u4e00-\u9fa5]+[\u3001.\uff0e]",
            r"^\u7b2c[\u4e00-\u9fa5]+[\u7ae0\u7bc0\u90e8]"
        ]

        self.stopwords = {'contents', 'index', '目次'}

    def extract_outline(self, pdf_path):
        try:
            text_blocks = self._extract_text_with_formatting(pdf_path)
            text_blocks = self._merge_small_vertical_blocks(text_blocks)

            if self.language == 'auto' and text_blocks:
                full_text = " ".join([b['text'] for b in text_blocks])
                self.language = detect(full_text)

            self.heading_patterns = self.heading_patterns_en + self.heading_patterns_ja if self.language.startswith('ja') else self.heading_patterns_en

            title = self._extract_title(text_blocks)
            candidate_headings = self._detect_heading_candidates(text_blocks)
            clustered_headings = self._assign_heading_levels(candidate_headings)
            optimized_headings = self._optimize_headings(clustered_headings)

            return {
                "title": title,
                "outline": optimized_headings,
                "language": self.language
            }
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return {
                "title": os.path.basename(pdf_path).replace('.pdf', ''),
                "outline": [],
                "language": self.language or 'unknown'
            }

    def _extract_text_with_formatting(self, pdf_path):
        blocks = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                chars = page.chars
                if not chars:
                    continue
                lines = defaultdict(list)
                for char in chars:
                    lines[round(char['top'])].append(char)
                for y, line_chars in sorted(lines.items()):
                    line_chars.sort(key=lambda x: x['x0'])
                    text = ''.join(c['text'] for c in line_chars)
                    if not text.strip():
                        continue
                    sizes = [c['size'] for c in line_chars if 'size' in c]
                    avg_size = np.mean(sizes) if sizes else 0
                    is_bold = any('Bold' in c.get('fontname', '') for c in line_chars)
                    blocks.append({
                        'text': text.strip(),
                        'page': page_num,
                        'font_size': avg_size,
                        'is_bold': is_bold,
                        'y_pos': y
                    })
        return blocks

    def _merge_small_vertical_blocks(self, blocks, y_threshold=5):
        merged = []
        prev = None
        for block in sorted(blocks, key=lambda b: (b['page'], b['y_pos'])):
            if prev and block['page'] == prev['page']:
                if abs(block['y_pos'] - prev['y_pos']) < y_threshold:
                    prev['text'] += ' ' + block['text']
                    prev['font_size'] = max(prev['font_size'], block['font_size'])
                    prev['is_bold'] = prev['is_bold'] or block['is_bold']
                    continue
            merged.append(block)
            prev = block
        return merged

    def _extract_title(self, blocks):
        if not blocks:
            return "Untitled Document"
        first_page_blocks = [b for b in blocks if b['page'] == 1]
        if not first_page_blocks:
            return "Untitled Document"
        sorted_blocks = sorted(first_page_blocks, key=lambda x: x['font_size'], reverse=True)
        for block in sorted_blocks[:5]:
            text = block['text']
            if len(text) < 3 or re.match(r'^\d+$', text):
                continue
            if len(text) > 100:
                continue
            doc = self.nlp(text)
            if not text.endswith('.') and any(token.pos_ in ['NOUN', 'PROPN'] for token in doc):
                return text
        for block in first_page_blocks:
            text = block['text']
            if len(text) > 3 and not re.match(r'^\d+$', text) and len(text) < 100:
                return text
        return "Untitled Document"

    def _detect_heading_candidates(self, blocks):
        candidates = []
        for block in blocks:
            text = block['text'].strip()
            if not text or len(text) > 200:
                continue
            if text.lower() in self.stopwords:
                continue
            is_heading = False
            for pattern in self.heading_patterns:
                if re.match(pattern, text):
                    is_heading = True
                    break
            if block['is_bold'] and len(text) < 100:
                is_heading = True
            if re.match(r'^\d+$', text):
                is_heading = False
            if is_heading:
                candidates.append(block)
        return candidates

    def _assign_heading_levels(self, heading_blocks):
        headings = []
        if not heading_blocks:
            return headings
        font_sizes = [b['font_size'] for b in heading_blocks if b['font_size'] > 0]
        unique_sizes = sorted(set(font_sizes))
        n_clusters = min(3, len(unique_sizes))
        if n_clusters == 0:
            return headings
        elif n_clusters == 1:
            cluster_to_level = {0: "H1"}
            size_to_cluster = {unique_sizes[0]: 0}
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            font_array = np.array(font_sizes).reshape(-1, 1)
            clusters = kmeans.fit_predict(font_array)
            size_to_cluster = {}
            for size, cluster in zip(font_sizes, clusters):
                size_to_cluster[size] = cluster
            cluster_centers = kmeans.cluster_centers_.flatten()
            sorted_clusters = sorted(range(n_clusters), key=lambda i: cluster_centers[i], reverse=True)
            cluster_to_level = {cluster: f"H{i+1}" for i, cluster in enumerate(sorted_clusters)}
        seen = set()
        for block in heading_blocks:
            size = block['font_size']
            cluster = size_to_cluster.get(size, -1)
            level = cluster_to_level.get(cluster, "H3")
            key = (block['text'], level)
            if key not in seen:
                headings.append({"level": level, "text": block['text'], "page": block['page']})
                seen.add(key)
        return headings

    def _optimize_headings(self, raw_headings):
        optimized = []
        i = 0
        while i < len(raw_headings):
            current = raw_headings[i]
            text = current["text"].strip()
            level = current["level"]
            page = current["page"]
            if text.endswith('.') and text[:-1].isdigit():
                if i + 1 < len(raw_headings):
                    next_text = raw_headings[i + 1]["text"].strip()
                    optimized.append({"level": level, "text": f"{text} {next_text}", "page": page})
                    i += 2
                    continue
            if text.startswith("(a)") or text.startswith("(b)") or text.lower().startswith("i declare"):
                level = "H3"
            optimized.append({"level": level, "text": text, "page": page})
            i += 1
        return optimized

    def save_outline(self, outline_dict, output_path, lang=None):
        if lang:
            base, ext = os.path.splitext(output_path)
            output_path = f"{base}_{lang}{ext}"
        final_output = {
            "title": outline_dict.get("title", "Untitled Document"),
            "outline": outline_dict.get("outline", [])
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

def process_single_pdf(pdf_file, input_dir, output_dir):
    extractor = OutlineExtractor()
    input_path = os.path.join(input_dir, pdf_file)
    output_filename = os.path.splitext(pdf_file)[0] + '.json'
    output_path = os.path.join(output_dir, output_filename)

    start_time = time.time()
    try:
        outline = extractor.extract_outline(input_path)
        extractor.save_outline(outline, output_path)
        elapsed_time = time.time() - start_time
        print(f"[DONE] Processed {pdf_file} in {elapsed_time:.2f} seconds. Found {len(outline['outline'])} headings.")
    except Exception as e:
        print(f"[ERROR] Failed to process {pdf_file}: {str(e)}")

def process_pdfs(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print("No PDF files found in the input directory.")
        return

    print(f"Found {len(pdf_files)} PDF files to process...")

    start_time = time.time()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(partial(process_single_pdf, input_dir=input_dir, output_dir=output_dir), pdf_files)

    total_time = time.time() - start_time
    print(f"All files processed in {total_time:.2f} seconds.")

def main():
    input_dir = os.path.join("sample_dataset", "pdfs")
    output_dir = os.path.join("sample_dataset", "outputs")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    process_pdfs(input_dir, output_dir)

if __name__ == "__main__":
    main()
