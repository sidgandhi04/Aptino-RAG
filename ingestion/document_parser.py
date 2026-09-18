import pymupdf
import os
import re

class DocumentParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        
    def parse(self):
        doc = pymupdf.open(self.pdf_path)
        
        section_pattern = re.compile(r'^(DEFINITIONS|WHAT WE COVER|WHAT WE EXCLUDE|EXTENSIONS|CLAIMS PROCEDURE|STANDARD TERMS AND CONDITIONS:?|[\d]+\.\s+[A-Z].+)$', re.MULTILINE)
        
        raw_text = ""
        page_map = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            text = text.replace("UNIVERSAL SOMPO GENERAL INSURANCE CO LTD", "")
            text = re.sub(r'\d+\s+CSC- Individual Health Insurance-Policy Wording\s+UNIHLIP18004V011718\s+IRDAI Reg No:134', '', text)
            
            start_idx = len(raw_text)
            raw_text += text + "\n"
            page_map.append((start_idx, len(raw_text), page_num + 1))
            
        sections = []
        for match in section_pattern.finditer(raw_text):
            sections.append((match.start(), match.group(1).strip()))
            
        chunks = []
        
        def get_page(idx):
            for start, end, p in page_map:
                if start <= idx < end:
                    return p
            return 1

        if not sections:
            sections = [(0, "General")]
        elif sections[0][0] > 0:
            sections.insert(0, (0, "General"))
            
        def split_text(text, chunk_size=1000, overlap=150):
            result = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                
                if end < len(text):
                    last_space = text.rfind(' ', start, end)
                    last_newline = text.rfind('\n', start, end)
                    split_idx = last_newline if last_newline > start + chunk_size // 2 else last_space
                    if split_idx != -1 and split_idx > start:
                        end = split_idx
                        
                result.append((start, text[start:end].strip()))
                start = end - overlap if end < len(text) else len(text)
            return result
            
        for i, (start_idx, heading) in enumerate(sections):
            end_idx = sections[i+1][0] if i + 1 < len(sections) else len(raw_text)
            section_text = raw_text[start_idx:end_idx].strip()
            
            if not section_text:
                continue
                
            sub_chunks = split_text(section_text)
            for j, (relative_start, text_chunk) in enumerate(sub_chunks):
                chunks.append({
                    "chunk_id": f"chunk_{i}_{j}",
                    "text": text_chunk,
                    "metadata": {
                        # A section can span several pages.  Cite the page on
                        # which this individual chunk begins, not its heading.
                        "page": get_page(start_idx + relative_start),
                        "section": heading if heading.isupper() else "Sub-section",
                        "heading": heading,
                        "source": os.path.basename(self.pdf_path)
                    }
                })
                
        return chunks

if __name__ == "__main__":
    import sys
    parser = DocumentParser(sys.argv[1])
    chunks = parser.parse()
    print(f"Parsed {len(chunks)} chunks.")
    for c in chunks[:3]:
        print({k: str(v).encode('ascii', 'ignore').decode() if isinstance(v, str) else v for k, v in c.items()})
