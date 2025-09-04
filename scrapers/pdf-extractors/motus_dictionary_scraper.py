#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motus Hiligaynon Dictionary Extractor — ONE FILE

• Pick the PDF manually (no embedded paths)
• Simple Tkinter UI with live log
• Adjustable start page & continuation depth
• Exports CSV or JSON

Dependencies (any one PDF reader path will work):
    pip install pymupdf pdfminer.six pandas
"""

import io
import json
import csv
import sys
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# ---------- UI ----------
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---------- Optional deps (we'll detect at runtime) ----------
_HAS_PYMUPDF = False
_HAS_PDFMINER = False
_HAS_PANDAS = False

try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    _HAS_PYMUPDF = False

try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer
    _HAS_PDFMINER = True
except Exception:
    _HAS_PDFMINER = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False


# ===========================
#        PDF READER
# ===========================

def load_pdf_as_pages(pdf_file_or_bytes) -> List[Tuple[int, str]]:
    """
    Return: list of (page_index, text) tuples.
    Accepts path-like, file-like, or raw bytes.
    Uses PyMuPDF if available, else pdfminer.six.
    """
    # read bytes
    if hasattr(pdf_file_or_bytes, "read"):
        pdf_bytes = pdf_file_or_bytes.read()
    elif isinstance(pdf_file_or_bytes, (bytes, bytearray)):
        pdf_bytes = bytes(pdf_file_or_bytes)
    else:
        with open(pdf_file_or_bytes, "rb") as f:
            pdf_bytes = f.read()

    # Try PyMuPDF first (fast & layout-friendly)
    if _HAS_PYMUPDF:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = []
            for i, page in enumerate(doc):
                text = page.get_text("text")
                pages.append((i, text))
            return pages
        except Exception:
            pass

    # Fallback to pdfminer.six
    if _HAS_PDFMINER:
        pages = []
        for i, layout in enumerate(extract_pages(io.BytesIO(pdf_bytes))):
            lines = []
            for el in layout:
                if isinstance(el, LTTextContainer):
                    lines.append(el.get_text())
            pages.append((i, "".join(lines)))
        return pages

    raise RuntimeError(
        "No PDF engine available. Please install at least one of:\n"
        "  pip install pymupdf\n"
        "  pip install pdfminer.six"
    )


# ===========================
#        CLEANING
# ===========================

DIACRITICS = "âêîôûáéíóúàèìòùÁÉÍÓÚÀÈÌÒÙ’'"

def dehyphenate_wraps(text: str) -> str:
    # join hyphenated line-breaks: "ba-\nlay" -> "balay"
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def join_broken_sentences(text: str) -> str:
    # if a line ends with lowercase & next starts lowercase, join with space
    return re.sub(r"([a-z" + DIACRITICS + r"])\n([a-z" + DIACRITICS + r"])", r"\1 \2", text)

def split_sentences(paragraph: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", paragraph)
    return [p.strip() for p in parts if p.strip()]


# ===========================
#        PARSER (Motus)
# ===========================

# POS from "List of Symbols and Abbreviations" (Motus)
POS_ABBR = {
    "n": "noun", "v": "verb", "adj": "adjective", "adv": "adverb", "pr": "pronoun",
    "pt": "particle", "con": "conjunction", "num": "numeral", "id": "idiom",
    "intj": "interjection", "d": "deictic", "va": "verbal-affix", "na": "noun-formative", "aa": "adjective-formative"
}

SECTION_HEADER = re.compile(r"^\s*[A-Z]\s*$")
HEADING_WORDS = {
    "CONTENTS","PREFACE","INTRODUCTION","NOTES","BIBLIOGRAPHY",
    "LIST OF SYMBOLS AND ABBREVIATIONS"
}
HEADWORD = re.compile(rf"^([A-Za-z{DIACRITICS}\-]+)\s+(.*)$")
ALL_CAPS = re.compile(r"^[A-Z][A-Z\s\-]+$")

POS_TOKEN = re.compile(r"\b(?:intj|con|num|adj|adv|va|na|id|pt|pr|d|n|v)\b")
AFFIX_BLOCK = re.compile(r"/[^/\n]+/")  # /mag-,-un/, /ka-…-an/, etc.

def looks_like_headword(line: str) -> bool:
    if not line or len(line) < 2:
        return False
    if ALL_CAPS.match(line.strip()):
        return False
    if line.strip() in HEADING_WORDS:
        return False
    m = HEADWORD.match(line)
    if not m:
        return False
    word = m.group(1)
    # avoid single-letter/punct headwords
    if len(re.sub(r"[^A-Za-z" + DIACRITICS + r"]", "", word)) < 2:
        return False
    return True

def extract_pos(chunk: str) -> List[str]:
    tags = []
    for tok in POS_TOKEN.findall(chunk):
        long = POS_ABBR.get(tok, tok)
        if long not in tags:
            tags.append(long)
    return tags

def extract_affixes(chunk: str) -> List[str]:
    blocks, seen = [], set()
    for m in AFFIX_BLOCK.finditer(chunk):
        txt = m.group(0).strip()
        if 2 <= len(txt) <= 80 and txt not in seen:
            seen.add(txt)
            blocks.append(txt)
    return blocks[:10]

def extract_examples(chunk: str) -> List[str]:
    sents = split_sentences(chunk)
    markers = (" ang ", " si ", " nga ", " mga ", " sang ", " sa ")
    ex = []
    for s in sents:
        low = " " + s.lower() + " "
        if any(m in low for m in markers) or re.search(rf"[{DIACRITICS}]", s):
            ex.append(s)
    return ex[:5]

@dataclass
class Entry:
    word: str
    parts_of_speech: List[str]
    affixes: List[str]
    definitions: List[str]
    examples: List[str]
    page: int
    raw_text: str

def parse_pages(pages: List[Tuple[int, str]], start_page: int = 25, max_continuation: int = 12) -> List[Entry]:
    entries: List[Entry] = []

    for pno, raw in pages:
        if pno < start_page:
            continue

        # normalize page text
        text = dehyphenate_wraps(raw)
        text = join_broken_sentences(text)
        text = normalize_whitespace(text)

        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        i = 0
        while i < len(lines):
            line = lines[i]

            # skip section headers and headings
            if SECTION_HEADER.match(line) or line in HEADING_WORDS:
                i += 1
                continue

            if looks_like_headword(line):
                m = HEADWORD.match(line)
                head = m.group(1).strip()
                rest = m.group(2).strip()
                chunk_lines = [rest]

                # greedy continuation until next headword/section/heading
                j = i + 1
                cont = 0
                while j < len(lines) and cont < max_continuation:
                    nxt = lines[j]
                    if looks_like_headword(nxt) or SECTION_HEADER.match(nxt):
                        break
                    if nxt in HEADING_WORDS or ALL_CAPS.match(nxt):
                        break
                    chunk_lines.append(nxt)
                    cont += 1
                    j += 1

                chunk = " ".join(chunk_lines)
                pos = extract_pos(chunk)
                aff = extract_affixes(chunk)

                # definitions: sentences, trimming leading POS tag if present
                sents = split_sentences(chunk)
                defs: List[str] = []
                for s in sents:
                    if re.match(r"^(?:intj|con|num|adj|adv|va|na|id|pt|pr|d|n|v)\b", s):
                        s2 = re.sub(r"^(?:intj|con|num|adj|adv|va|na|id|pt|pr|d|n|v)\b\s*", "", s).strip()
                        if s2:
                            defs.append(s2)
                    else:
                        defs.append(s.strip())
                defs = defs[:8]

                exs = extract_examples(chunk)

                entries.append(Entry(
                    word=head,
                    parts_of_speech=pos,
                    affixes=aff,
                    definitions=defs,
                    examples=exs,
                    page=pno+1,  # 1-based for humans
                    raw_text=f"{head} {chunk}".strip()
                ))
                i = j
            else:
                i += 1

    # Deduplicate by (word, first POS)
    seen = set()
    deduped: List[Entry] = []
    for e in entries:
        key = (e.word.lower(), (e.parts_of_speech[0] if e.parts_of_speech else ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)
    return deduped


# ===========================
#        SAVE HELPERS
# ===========================

def entries_to_csv_df(entries: List[Entry]):
    if not _HAS_PANDAS:
        return None
    rows = []
    for e in entries:
        rows.append({
            "word": e.word,
            "parts_of_speech": "; ".join(e.parts_of_speech),
            "affixes": "; ".join(e.affixes),
            "definitions": " | ".join(e.definitions),
            "examples": " | ".join(e.examples),
            "page": e.page
        })
    return pd.DataFrame(rows)

def save_csv(entries: List[Entry], out_path: Path) -> None:
    """
    CSV without pandas (fallback), to keep this single-file truly portable.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["word","parts_of_speech","affixes","definitions","examples","page"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for e in entries:
            w.writerow({
                "word": e.word,
                "parts_of_speech": "; ".join(e.parts_of_speech),
                "affixes": "; ".join(e.affixes),
                "definitions": " | ".join(e.definitions),
                "examples": " | ".join(e.examples),
                "page": e.page
            })

def save_json(entries: List[Entry], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in entries], f, ensure_ascii=False, indent=2)


# ===========================
#           UI
# ===========================

class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master: tk.Tk = master
        self.master.title("Motus Hiligaynon Extractor (One File)")
        self.master.geometry("860x560")

        # state
        self.pdf_var = tk.StringVar()
        self.out_var = tk.StringVar()
        self.format_var = tk.StringVar(value="CSV")  # CSV or JSON
        self.start_page_var = tk.StringVar(value="26")   # human 1-based default
        self.max_cont_var = tk.StringVar(value="12")

        # layout
        pad = {"padx": 8, "pady": 6}
        self.grid(column=0, row=0, sticky="nsew")
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        # input
        fin = ttk.Frame(self); fin.grid(row=0, column=0, sticky="ew", **pad); fin.columnconfigure(1, weight=1)
        ttk.Label(fin, text="Input PDF").grid(row=0, column=0, sticky="w")
        ttk.Entry(fin, textvariable=self.pdf_var).grid(row=0, column=1, sticky="ew", padx=(8,8))
        ttk.Button(fin, text="Browse…", command=self.pick_pdf).grid(row=0, column=2)

        # output
        fout = ttk.Frame(self); fout.grid(row=1, column=0, sticky="ew", **pad); fout.columnconfigure(1, weight=1)
        ttk.Label(fout, text="Output file").grid(row=0, column=0, sticky="w")
        ttk.Entry(fout, textvariable=self.out_var).grid(row=0, column=1, sticky="ew", padx=(8,8))
        ttk.Button(fout, text="Save as…", command=self.pick_output).grid(row=0, column=2)

        # options
        fopt = ttk.LabelFrame(self, text="Options"); fopt.grid(row=2, column=0, sticky="ew", **pad)
        for c in range(4): fopt.columnconfigure(c, weight=1)
        ttk.Label(fopt, text="Output format").grid(row=0, column=0, sticky="w")
        ttk.Combobox(fopt, textvariable=self.format_var, values=["CSV","JSON"], width=8, state="readonly").grid(row=0, column=1, sticky="w")
        ttk.Label(fopt, text="Start page (1-based)").grid(row=1, column=0, sticky="w", pady=(6,0))
        ttk.Entry(fopt, textvariable=self.start_page_var, width=10).grid(row=1, column=1, sticky="w", pady=(6,0))
        ttk.Label(fopt, text="Max continuation lines").grid(row=1, column=2, sticky="w", pady=(6,0))
        ttk.Entry(fopt, textvariable=self.max_cont_var, width=10).grid(row=1, column=3, sticky="w", pady=(6,0))

        # run
        frun = ttk.Frame(self); frun.grid(row=3, column=0, sticky="ew", **pad); frun.columnconfigure(0, weight=1)
        self.btn_run = ttk.Button(frun, text="Run Extraction", command=self.run)
        self.btn_run.grid(row=0, column=0, sticky="e")

        # log
        flog = ttk.LabelFrame(self, text="Log"); flog.grid(row=4, column=0, sticky="nsew", **pad)
        flog.rowconfigure(0, weight=1); flog.columnconfigure(0, weight=1)
        self.txt = tk.Text(flog, height=16, wrap="word"); self.txt.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(flog, orient="vertical", command=self.txt.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.txt.configure(yscrollcommand=yscroll.set)

        # theme polish (optional)
        try:
            self.master.call("tk", "scaling", 1.2)
            style = ttk.Style()
            if "vista" in style.theme_names(): style.theme_use("vista")
            elif "clam" in style.theme_names(): style.theme_use("clam")
        except Exception:
            pass

        # env info
        self.log(f"PDF engines: PyMuPDF={_HAS_PYMUPDF}, pdfminer.six={_HAS_PDFMINER}")
        self.log(f"Pandas available: {_HAS_PANDAS}")

    # helpers
    def log(self, msg: str):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)
        self.txt.update_idletasks()

    def pick_pdf(self):
        path = filedialog.askopenfilename(
            title="Select Motus PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if path:
            self.pdf_var.set(path)
            suggested = Path(path).with_name(Path(path).stem + "_extracted.csv")
            if not self.out_var.get():
                self.out_var.set(str(suggested))

    def pick_output(self):
        fmt = self.format_var.get().upper()
        defext = ".csv" if fmt == "CSV" else ".json"
        path = filedialog.asksaveasfilename(
            title="Save output as…",
            defaultextension=defext,
            filetypes=[("CSV", "*.csv"), ("JSON", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.out_var.set(path)

    def run(self):
        pdf_path = self.pdf_var.get().strip()
        out_path = self.out_var.get().strip()
        fmt = self.format_var.get().upper()

        if not pdf_path or not Path(pdf_path).exists():
            messagebox.showerror("Missing input", "Please choose a valid input PDF.")
            return
        if not out_path:
            messagebox.showerror("Missing output", "Please choose where to save the output file.")
            return

        # parse numeric options
        try:
            start_1b = int(self.start_page_var.get().strip())
            if start_1b < 1: raise ValueError
            start_idx = start_1b - 1
        except Exception:
            messagebox.showerror("Invalid input", "Start page must be a positive integer.")
            return

        try:
            max_cont = int(self.max_cont_var.get().strip())
            if max_cont < 1: raise ValueError
        except Exception:
            messagebox.showerror("Invalid input", "Max continuation lines must be a positive integer.")
            return

        self.btn_run.config(state="disabled")
        self.master.after(60, self._do_run, Path(pdf_path), Path(out_path), fmt, start_idx, max_cont)

    def _do_run(self, pdf_path: Path, out_path: Path, fmt: str, start_idx: int, max_cont: int):
        try:
            self.log(f"Reading PDF: {pdf_path}")
            pages = load_pdf_as_pages(pdf_path)
            self.log(f"Loaded {len(pages)} pages")

            self.log(f"Parsing (start page={start_idx+1}, max_cont={max_cont}) …")
            entries = parse_pages(pages, start_page=start_idx, max_continuation=max_cont)
            self.log(f"Parsed {len(entries)} entries")

            if not entries:
                messagebox.showwarning("No entries", "Finished but no entries were found. Try adjusting start page/continuation.")
                return

            if fmt == "CSV":
                # Use pandas if present for convenience; fallback to pure-CSV writer
                if _HAS_PANDAS:
                    df = entries_to_csv_df(entries)
                    df.to_csv(out_path, index=False)
                else:
                    save_csv(entries, out_path)
                self.log(f"Saved CSV → {out_path}")
            else:
                save_json(entries, out_path)
                self.log(f"Saved JSON → {out_path}")

            messagebox.showinfo("Done", f"Extraction complete.\nSaved to:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Extraction failed:\n{e}")
        finally:
            self.btn_run.config(state="normal")


def main():
    root = tk.Tk()
    root.columnconfigure(0, weight=1); root.rowconfigure(0, weight=1)
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
