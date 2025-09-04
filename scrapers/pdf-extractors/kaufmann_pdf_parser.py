#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, csv, time, logging, sys
from typing import List, Dict, Any, Optional, Tuple
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ---------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = os.path.join("scrapers", "data", "lexicon")

try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.output_config import get_output_path, OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    def get_output_path(ext: str, prefix: str) -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return os.path.join(OUTPUT_DIR, f"{prefix}.{ext}")

# ---------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------
try:
    import pdfplumber
except Exception as e:
    raise SystemExit("Please install pdfplumber: pip install pdfplumber") from e

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def dehyphen(s: str) -> str:
    s = re.sub(r'(\w)-\s+([a-záéíóúñâêîôû])', r'\1\2', s, flags=re.I)
    return re.sub(r'\s+', ' ', s).strip()

def group_chars_into_lines(chars, y_tol: float = 2.0):
    chars = sorted(chars, key=lambda c: (round(c["top"], 1), c["x0"]))
    lines, current, cy = [], [], None
    for ch in chars:
        y = round(ch["top"], 1)
        if cy is None or abs(y - cy) > y_tol:
            if current:
                lines.append(current)
            current, cy = [ch], y
        else:
            current.append(ch)
    if current:
        lines.append(current)
    return lines

def text_of_line(line) -> str:
    return ''.join(ch["text"] for ch in sorted(line, key=lambda c: c["x0"]))

HEAD_RE = re.compile(r'\s*([A-Za-z0-9ÁÉÍÓÚÑáéíóúâêîôû’ʼ\-\.\u02BC]+)\s*[,—(]')

def initial_bold_head(line) -> Optional[Tuple[str, float, float]]:
    seg = sorted(line, key=lambda c: c["x0"])
    if not seg:
        return None

    run = []
    for ch in seg:
        if ch["text"].isspace() and not run:
            continue
        font = ch.get("fontname", "").lower()
        is_bold = any(tag in font for tag in ("bold", "black", "heavy"))
        if not run and not is_bold:
            return None
        if is_bold:
            run.append(ch)
            continue
        break

    if not run:
        return None

    head_txt = ''.join(c["text"] for c in run)
    tail = ''.join(ch["text"] for ch in seg[len(run):len(run)+6])
    cand = head_txt + tail

    m = HEAD_RE.match(cand)
    if not m:
        return None

    head = m.group(1).strip()
    avg_size = sum(c["size"] for c in run) / len(run)
    if avg_size > 12:
        return None
    x0 = min(c["x0"] for c in run)
    return head, x0, avg_size

ROOT_RE = re.compile(r'(?:Dim\.|Freq\.)\s*(?:and\s*)?of\s+([A-Za-zÁÉÍÓÚÑáéíóúâêîôû’ʼ\-]+)', re.I)
SEE_RE = re.compile(r'^(?:See|See also)\s+(.+)', re.I)

def find_root(head: str, definition: str) -> Optional[Dict[str, str]]:
    if m := ROOT_RE.search(definition):
        return {"form": m.group(1), "via": "dim/freq cue"}
    if m := SEE_RE.match(definition):
        return {"form": m.group(1).strip(), "via": "see-pointer"}
    if head.startswith("-"):
        return None
    return None

# ---------------------------------------------------------------------
# Main Parser
# ---------------------------------------------------------------------
class KaufmannPDFParser:
    def __init__(self, pdf_path: str, output_dir: Optional[str] = None, delay: float = 0.0):
        self.pdf_path = pdf_path
        self.output_dir = output_dir or OUTPUT_DIR
        self.delay = delay
        os.makedirs(self.output_dir, exist_ok=True)
        self.log = logging.getLogger("kaufmann")

    def extract_entries_from_page(self, page) -> List[Dict[str, Any]]:
        lines = group_chars_into_lines(page.chars)
        col_w = page.width / 3.0

        heads = []
        for i, line in enumerate(lines):
            got = initial_bold_head(line)
            if not got:
                continue
            head, x0, size = got
            if len(head) == 1 and size > 12:
                continue
            col = int(x0 // col_w)
            heads.append((i, head, col))

        entries: List[Dict[str, Any]] = []
        for j, (start_i, head, col) in enumerate(heads):
            end_i = heads[j+1][0] if j+1 < len(heads) else len(lines)
            block = ' '.join(text_of_line(lines[k]) for k in range(start_i, end_i))
            m = re.search(re.escape(head) + r'\s*[,—(]\s*(.*)', block)
            definition = dehyphen(m.group(1) if m else block)

            root_info = find_root(head, definition)
            entries.append({
                "word": head,
                "definition": definition,
                "page": int(page.page_number),
                "column": col,
                "is_bold": True,
                "is_hiligaynon_root": True,
                "root_info": root_info or "",
            })
        return entries

    def parse(self, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
        all_entries: List[Dict[str, Any]] = []
        with pdfplumber.open(self.pdf_path) as pdf:
            n = len(pdf.pages)
            pages = range(n if not max_pages else min(max_pages, n))
            for idx in pages:
                page = pdf.pages[idx]
                page_entries = self.extract_entries_from_page(page)
                all_entries.extend(page_entries)
                if self.delay:
                    time.sleep(self.delay)
        return all_entries

    def save(self, entries: List[Dict[str, Any]]) -> Tuple[str, str]:
        jpath = get_output_path("json", "kaufmann")
        cpath = get_output_path("csv", "kaufmann")

        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

        keys = ["word", "page", "column", "is_bold", "is_hiligaynon_root", "definition", "root_info"]
        with open(cpath, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for e in entries:
                row = e.copy()
                if isinstance(row.get("root_info"), dict):
                    row["root_info"] = json.dumps(row["root_info"], ensure_ascii=False)
                w.writerow({k: row.get(k, "") for k in keys})

        return jpath, cpath

# ---------------------------------------------------------------------
# Tkinter UI
# ---------------------------------------------------------------------
def run_ui():
    def choose_pdf():
        path = filedialog.askopenfilename(title="Select Kaufmann PDF", filetypes=[("PDF files", "*.pdf")])
        if path:
            pdf_var.set(path)

    def run_parser():
        pdf_path = pdf_var.get()
        if not pdf_path:
            messagebox.showerror("Error", "Please select a PDF file first.")
            return
        try:
            parser = KaufmannPDFParser(pdf_path=pdf_path, output_dir=DEFAULT_OUTPUT_DIR)
            entries = parser.parse()
            jpath, cpath = parser.save(entries)
            messagebox.showinfo("Done", f"Extracted {len(entries)} entries.\n\nSaved:\n{jpath}\n{cpath}")

            # show first few entries in table
            for row in tree.get_children():
                tree.delete(row)
            for e in entries[:50]:  # show only first 50 to avoid overload
                tree.insert("", "end", values=(e["word"], e["definition"][:50], e["page"], e["column"]))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    root = tk.Tk()
    root.title("Kaufmann Dictionary Parser")
    root.geometry("800x600")

    pdf_var = tk.StringVar()

    tk.Label(root, text="PDF Path:").pack(anchor="w", padx=10, pady=5)
    tk.Entry(root, textvariable=pdf_var, width=80).pack(side="left", padx=10)
    tk.Button(root, text="Browse", command=choose_pdf).pack(side="left")

    tk.Button(root, text="Parse PDF", command=run_parser).pack(pady=10)

    # results table
    cols = ("Word", "Definition (truncated)", "Page", "Column")
    tree = ttk.Treeview(root, columns=cols, show="headings")
    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=150)
    tree.pack(fill="both", expand=True)

    root.mainloop()

# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_ui()
