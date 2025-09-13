# Write a full, hardened extractor **with GUI preserved** to /mnt/data/motus_dictionary_scraper_ui.py
from pathlib import Path

code = r''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motus Hiligaynon Dictionary Extractor — HARDENED+++ (GUI kept) + smarter PDF ordering

Why this version?
• Fixes mixed/shifted rows like: headword text becoming the meaning, or English gloss
  being attached to the wrong word (your sample CSV showed that).
• Handles both POS abbreviations (n, v, adj, intj, …) **and** full English names
  (noun, verb, adjective, interjection, particle, conjunction, numeral, pronoun, deictic, idiom).
• Uses PyMuPDF block coordinates to rebuild two-column page order (left column top→down
  then right column top→down), reducing interleaving like “ákun … April.” appearing together.
  Falls back gracefully to pdfminer.six or PyPDF2 if PyMuPDF is unavailable.
• Joins hyphenated line-breaks but preserves affix blocks (/mag-,-un/ etc.).
• Strips running headers/footers like “HILIGAYNON DICTIONARY”, stray “A 12”, etc.
• Repairs glued tokens (“Thebook” → “The book”, “angbáta’” → “ang báta’”) cautiously.
• Accepts stacked layout (headword on line 1; POS + sense start on line 2).
• CSV saving uses strong quoting (QUOTE_ALL) to prevent column drift in spreadsheet apps.

CLI examples:
  python motus_dictionary_scraper_ui.py --nogui -i /path/ceceilmotus.pdf -o out.csv --auto-start
  python motus_dictionary_scraper_ui.py --nogui -i input.pdf -o out_raw.json --schema RAW

GUI: double-click or run `python motus_dictionary_scraper_ui.py`.
"""

import io
import re
import csv
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# ---------------- Optional Deps ----------------
_HAS_PYMUPDF = False
_HAS_PDFMINER = False
_HAS_PYPDF2 = False
_HAS_PANDAS = False

try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    pass

try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LTTextBox, LAParams
    _HAS_PDFMINER = True
except Exception:
    pass

try:
    import PyPDF2
    _HAS_PYPDF2 = True
except Exception:
    pass

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    pass

# ---------------- Constants ----------------
DIACRITICS = "âêîôûáéíóúàèìòùÁÉÍÓÚÀÈÌÒÙ’'ˊ"
ALPHA_DIAC = r"A-Za-z" + DIACRITICS

# Accept both abbreviations and full names
POS_ABBR2LONG = {
    # abbreviations -> long form
    "n": "noun", "v": "verb", "adj": "adjective", "adv": "adverb",
    "pr": "pronoun", "pt": "particle", "con": "conjunction", "num": "numeral",
    "id": "idiom", "intj": "interjection", "d": "deictic",
    "va": "verbal affix", "na": "noun formative", "aa": "adjective formative",
    # short alt spellings
    "int": "interjection", "intj.": "interjection", "adj.": "adjective",
    "adv.": "adverb", "conj": "conjunction", "conj.": "conjunction"
}
POS_FULL_SET = {
    "noun", "verb", "adjective", "adverb",
    "pronoun", "particle", "conjunction", "numeral",
    "idiom", "interjection", "deictic",
    "verbal affix", "noun formative", "adjective formative"
}
POS_TOKEN = r"(?:intj(?:\.?)|interjection|con(?:j\.?)?|conjunction|num|numeral|adj(?:\.?)|adjective|adv(?:\.?)|adverb|va|verbal affix|na|noun formative|aa|adjective formative|id|idiom|pt|particle|pr|pronoun|d|deictic|n|noun|v|verb)"

INLINE_HEAD = re.compile(
    rf"^(?P<word>[{ALPHA_DIAC}\-]+)\s+(?P<pos>{POS_TOKEN})\b(?P<tail>.*)$",
    flags=re.IGNORECASE
)
STACKED_POS_ONLY = re.compile(
    rf"^(?P<pos>{POS_TOKEN})\b(?:\s+(?P<affix>/[^/]+/))?\s*(?P<rest>.*)$",
    flags=re.IGNORECASE
)
POS_ANCHOR = re.compile(rf"(?<!\S){POS_TOKEN}\b", flags=re.IGNORECASE)
AFFIX_BLOCK = re.compile(r"/[^/\n]+/")
SECTION_HEADER = re.compile(r"^\s*[A-Z]\s*$")
ALL_CAPS = re.compile(r"^[A-Z][A-Z\s\-]+$")

# ---------------- Heuristics ----------------
def unspace_letters(s: str) -> str:
    """Collapse OCR letter-spaced words like 't h i s' -> 'this' (>=4 letters)."""
    def _join(match: re.Match) -> str:
        chunk = match.group(0)
        return re.sub(r"\s+", "", chunk)
    return re.sub(r"(?:\b[ A-Za-z’']\s+){3,}[A-Za-z’']\b", _join, s)

def diacritic_ratio(s: str) -> float:
    return sum(ch in DIACRITICS for ch in s) / max(1, len(s))

def looks_hiligaynon(s: str) -> bool:
    low = " " + s.lower() + " "
    markers = (" ang ", " si ", " nga ", " mga ", " sang ", " sa ", " kag ", " nag", " gin", " mag", " naga", " gina", " gin’", " nag’")
    if any(m in low for m in markers): return True
    if diacritic_ratio(s) >= 0.02: return True
    return False

def looks_english(s: str) -> bool:
    return diacritic_ratio(s) < 0.02

def split_sentences(paragraph: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [p.strip() for p in parts if p.strip()]

def split_meaning_and_examples(text: str):
    sents = split_sentences(text)
    if not sents:
        return text.strip().strip(";:, "), "", ""
    ex_en = ""
    ex_hil = ""
    en_idx = None
    # last English-like as translation
    for i in range(len(sents)-1, -1, -1):
        if looks_english(sents[i]):
            ex_en = sents[i]; en_idx = i; break
    if en_idx is not None:
        for j in range(en_idx-1, -1, -1):
            if looks_hiligaynon(sents[j]):
                ex_hil = sents[j]; break
        cutoff = (en_idx-1) if ex_hil else en_idx
        meaning = " ".join(sents[:cutoff]).strip().strip(";:, ")
        return meaning, ex_hil, ex_en
    # fallback: no English, try last HIL
    for i in range(len(sents)-1, -1, -1):
        if looks_hiligaynon(sents[i]):
            ex_hil = sents[i]
            meaning = " ".join(sents[:i]).strip().strip(";:, ")
            return meaning, ex_hil, ""
    return " ".join(sents).strip().strip(";:, "), "", ""

# ---------------- Text cleanup utilities ----------------
RX_HILIGAYNON_DICTIONARY = re.compile(
    r"h\s*i\s*l\s*i\s*g\s*a\s*y\s*n\s*o\s*n\s+d\s*i\s*c\s*t\s*i\s*o\s*n\s*a\s*r\s*y\s*\d*",
    re.IGNORECASE
)
RX_A_NUM = re.compile(r"\bA\s?\d+\b")

def fix_punct_spacing(s: str) -> str:
    # Remove spaces before punctuation
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    # Ensure one space after punctuation if followed by a letter/digit
    s = re.sub(r"([.,;:!?])(?=[A-Za-z0-9ÁÉÍÓÚáéíóú’])", r"\1 ", s)
    # Normalize multiple spaces
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def smart_space_repair(s: str) -> str:
    # HIL function words glued to content
    s = re.sub(r"\b(ang|sa|si|sang|mga)([A-Za-zÁÉÍÓÚáéíóú’])", r"\1 \2", s, flags=re.IGNORECASE)
    # EN function words glued to next token (3+ letters to avoid "He" + "len")
    s = re.sub(r"\b(The|He|She|It|They|We|You|I)([a-z]{3,})", r"\1 \2", s)
    s = re.sub(r"\b(to|of|and|but|or|for|from|with|that|this|these|those)([a-z]{3,})", r"\1 \2", s, flags=re.IGNORECASE)
    return s

def strip_running_headers(s: str) -> str:
    s = RX_HILIGAYNON_DICTIONARY.sub("", s)
    s = RX_A_NUM.sub("", s)
    return s

def join_hyphenation(txt: str) -> str:
    # Merge hyphenated line breaks: end-with-letter + "-\n" + lowercase
    return re.sub(r"(?<=\w)-\n(?=[a-záéíóú])", "", txt)

# ---------------- PDF Extraction (with 2-column rebuild) ----------------
def _blocks_to_text_2col(blocks: List[Tuple[float,float,float,float,str]], page_width: float) -> str:
    """Given PyMuPDF-like blocks (x0,y0,x1,y1, text), reconstruct 2-column reading order."""
    # Filter valid text blocks and normalize newlines
    bnorm = []
    for (x0,y0,x1,y1,txt) in blocks:
        if not txt or not txt.strip():
            continue
        t = txt.replace("\r\n","\n").replace("\r","\n")
        bnorm.append((x0,y0,x1,y1,t))
    if not bnorm:
        return ""

    # Decide midline by page width
    mid = page_width / 2.0
    left_blocks = [(y0, x0, t) for (x0,y0,x1,y1,t) in bnorm if (x0 + x1)/2.0 <= mid]
    right_blocks = [(y0, x0, t) for (x0,y0,x1,y1,t) in bnorm if (x0 + x1)/2.0 > mid]

    left_blocks.sort(key=lambda r: (round(r[0],1), round(r[1],1)))
    right_blocks.sort(key=lambda r: (round(r[0],1), round(r[1],1)))

    left_text  = "\n".join([t for (_,_,t) in left_blocks])
    right_text = "\n".join([t for (_,_,t) in right_blocks])
    # Return left column then right column, with a spacer to avoid accidental run-on
    return left_text.rstrip() + "\n\n" + right_text.lstrip()

def extract_with_pymupdf(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    pages = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        # Prefer blocks to keep spatial info for columns
        raw_blocks = page.get_text("blocks")
        blocks = []
        for b in raw_blocks:
            if len(b) >= 5 and isinstance(b[4], str):
                x0,y0,x1,y1,txt = b[0], b[1], b[2], b[3], b[4]
                blocks.append((float(x0), float(y0), float(x1), float(y1), txt))
        if blocks:
            t = _blocks_to_text_2col(blocks, page.rect.width)
        else:
            t = page.get_text("text") or ""
        pages.append((i, t))
    return pages

def extract_with_pdfminer(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    pages = []
    laparams = LAParams()
    for pageno, layout in enumerate(extract_pages(io.BytesIO(pdf_bytes), laparams=laparams)):
        blocks = []
        for el in layout:
            if isinstance(el, (LTTextContainer, LTTextBox)):
                txt = el.get_text()
                # approximate columns by x0,x1
                x0, y0, x1, y1 = el.bbox
                blocks.append((float(x0), float(y0), float(x1), float(y1), txt))
        if blocks:
            # Determine page width by max x1
            page_width = max([b[2] for b in blocks]) if blocks else 1000.0
            t = _blocks_to_text_2col(blocks, page_width)
        else:
            t = ""
        pages.append((pageno, t))
    return pages

def extract_with_pypdf2(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    pages = []
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    for i in range(len(reader.pages)):
        pages.append((i, reader.pages[i].extract_text() or ""))
    return pages

def spaced_letter_score(sample: str) -> float:
    hits = len(re.findall(r"(?:\b[ A-Za-z’']\s+){3,}[A-Za-z’']\b", sample))
    return 1000.0 * hits / max(1, len(sample))

def join_collision_score(sample: str) -> float:
    # Count likely glued tokens (HIL & EN function words) per 1k chars
    patterns = [
        r"\b(ang|sa|si|sang|mga)[A-Za-zÁÉÍÓÚáéíóú’]",
        r"\b(The|He|She|It|They|We|You|I)[a-z]{3,}",
        r"\b(to|of|and|but|or|for|from|with|that|this|these|those)[a-z]{3,}",
    ]
    hits = 0
    for p in patterns:
        hits += len(re.findall(p, sample))
    return 1000.0 * hits / max(1, len(sample))

def choose_best_engine(candidates: Dict[str, List[Tuple[int, str]]], sample_pages=(27,28,29)) -> str:
    best = None
    best_score = 1e9
    for name, pages in candidates.items():
        if not pages: continue
        buf = []
        for pno in sample_pages:
            if 0 <= pno < len(pages):
                buf.append(pages[pno][1])
        sample = "\n".join(buf)
        score = 0.6 * spaced_letter_score(sample) + 0.4 * join_collision_score(sample)
        if score < best_score:
            best_score = score; best = name
    return best

def load_pdf_pages(path: Path) -> List[Tuple[int, str]]:
    pdf_bytes = Path(path).read_bytes()
    candidates: Dict[str, List[Tuple[int, str]]] = {}
    if _HAS_PYMUPDF:
        try: candidates["pymupdf"] = extract_with_pymupdf(pdf_bytes)
        except Exception: candidates["pymupdf"] = []
    if _HAS_PDFMINER:
        try: candidates["pdfminer"] = extract_with_pdfminer(pdf_bytes)
        except Exception: candidates["pdfminer"] = []
    if _HAS_PYPDF2:
        try: candidates["pypdf2"] = extract_with_pypdf2(pdf_bytes)
        except Exception: candidates["pypdf2"] = []

    if not any(candidates.values()):
        raise RuntimeError("No PDF engine succeeded. Install one of: PyMuPDF, pdfminer.six, PyPDF2.")

    best = choose_best_engine(candidates)
    pages = candidates[best]

    # Clean & unspace per page
    cleaned = []
    for i, txt in pages:
        t = txt.replace("\r\n", "\n").replace("\r", "\n")
        t = join_hyphenation(t)
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        t = unspace_letters(t)
        cleaned.append((i, t.strip()))
    return cleaned

# ---------------- Data Models ----------------
@dataclass
class Sense:
    pos: str
    affixes: List[str]
    meaning: str
    ex_hil: str
    ex_en: str

@dataclass
class Entry:
    word: str
    senses: List[Sense]
    page: int
    raw: str

# ---------------- POS normalization ----------------
def normalize_pos_token(tok: str) -> str:
    if not tok:
        return ""
    t = tok.strip().lower().rstrip(".")
    if t in POS_ABBR2LONG:
        return POS_ABBR2LONG[t]
    # if already full name or close variant
    for full in POS_FULL_SET:
        if t == full or t.replace("-", " ") == full:
            return full
    return tok.strip()

# ---------------- Parsing ----------------
def page_text_to_lines(txt: str, do_strip_headers=True, do_fix_punct=True, do_smart_space=True) -> List[str]:
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    if do_strip_headers:
        txt = strip_running_headers(txt)
    lines = [ln.strip() for ln in txt.split("\n")]
    out = []
    for ln in lines:
        if not ln: continue
        if do_fix_punct:
            ln = fix_punct_spacing(ln)
        if do_smart_space:
            ln = smart_space_repair(ln)
        out.append(ln)
    return out

def diacritic_word_ok(head: str) -> bool:
    # Reject capitalized plain-English sentence starters with no diacritics
    if head and head[0].isupper() and diacritic_ratio(head) < 0.02:
        # Allow month names and proper nouns with diacritics or short
        return False
    alpha = re.sub(rf"[^A-Za-z{DIACRITICS}]", "", head)
    return len(alpha) >= 2

def parse_page(lines: List[str], page_num_1b: int, max_cont: int = 12) -> List[Entry]:
    i = 0
    out: List[Entry] = []

    def collect_block(start_i: int):
        line = lines[start_i]
        m = INLINE_HEAD.match(line)
        if m and diacritic_word_ok(m.group("word")):
            word = m.group("word")
            pos = normalize_pos_token(m.group("pos"))
            tail = m.group("tail").strip()
            # Store POS in-line for block parsing
            first = f"{pos} {tail}".strip() if tail else pos
            block = [first]
            j = start_i + 1; cont = 0
            while j < len(lines) and cont < max_cont:
                nxt = lines[j]
                if INLINE_HEAD.match(nxt): break  # new headword inline
                # New stacked headword? head line followed by pos line
                if diacritic_word_ok(nxt.split(" ",1)[0]) and j+1 < len(lines) and STACKED_POS_ONLY.match(lines[j+1]):
                    break
                block.append(nxt); cont += 1; j += 1
            return word, block, j

        # Stacked: headword line, POS on next line
        head = line.strip()
        if diacritic_word_ok(head) and (start_i+1) < len(lines):
            m2 = STACKED_POS_ONLY.match(lines[start_i+1])
            if m2:
                pos = normalize_pos_token(m2.group("pos"))
                rest = m2.group("rest") or ""
                affx = m2.group("affix") or ""
                first = f"{pos} {affx} {rest}".strip()
                j = start_i + 2; block = [first]; cont = 0
                while j < len(lines) and cont < max_cont:
                    nxt = lines[j]
                    if INLINE_HEAD.match(nxt): break
                    if diacritic_word_ok(nxt.split(" ",1)[0]) and j+1 < len(lines) and STACKED_POS_ONLY.match(lines[j+1]):
                        break
                    block.append(nxt); cont += 1; j += 1
                return head, block, j

        return "", [], start_i + 1

    def split_senses(block_text: str) -> List[Sense]:
        bt = re.sub(r"\s+", " ", block_text).strip()
        # Replace any abbr/full POS tokens with canonical spacing for anchor finding
        anchors = list(POS_ANCHOR.finditer(bt))
        if not anchors:
            aff = list(dict.fromkeys(AFFIX_BLOCK.findall(bt)))
            meaning, ex_hil, ex_en = split_meaning_and_examples(bt)
            return [Sense("", aff, meaning, ex_hil, ex_en)]
        indices = [a.start() for a in anchors] + [len(bt)]
        senses: List[Sense] = []
        for k, a in enumerate(anchors):
            pos_tok = normalize_pos_token(a.group(0))
            seg = bt[a.end():indices[k+1]].strip()
            # consume immediate leading affix blocks
            leading = []
            while True:
                m = re.match(r"^(/[^/]+/)\s*", seg)
                if not m: break
                leading.append(m.group(1)); seg = seg[m.end():].strip()
            all_aff = list(dict.fromkeys(leading + AFFIX_BLOCK.findall(seg)))
            meaning, ex_hil, ex_en = split_meaning_and_examples(seg)
            senses.append(Sense(pos_tok, all_aff, meaning, ex_hil, ex_en))
        return senses

    while i < len(lines):
        line = lines[i]
        if not line or SECTION_HEADER.match(line) or ALL_CAPS.match(line):
            i += 1; continue

        word, block_lines, next_i = collect_block(i)
        if not word:
            i += 1; continue

        block_text = " ".join([b for b in block_lines if b])
        # Must contain at least one POS token to be a valid entry
        if not POS_ANCHOR.search(block_text):
            i = next_i; continue

        senses = split_senses(block_text)
        out.append(Entry(word=word, senses=senses, page=page_num_1b, raw=(word + " " + block_text)))
        i = next_i

    return out

def parse_pages(pages: List[Tuple[int, str]], start_index: int, max_cont: int = 12,
                do_strip_headers=True, do_fix_punct=True, do_smart_space=True) -> List[Entry]:
    entries: List[Entry] = []
    for pno, text in pages:
        if pno < start_index: continue
        lines = page_text_to_lines(text, do_strip_headers=do_strip_headers,
                                   do_fix_punct=do_fix_punct, do_smart_space=do_smart_space)
        entries.extend(parse_page(lines, pno+1, max_cont=max_cont))
    # de-dup by (word, first-sense pos+meaning) to dampen minor ordering changes
    seen = set(); deduped = []
    for e in entries:
        first_pos = e.senses[0].pos if e.senses else ""
        first_mean = e.senses[0].meaning[:120] if e.senses else ""
        key = (e.word.lower(), first_pos, first_mean)
        if key in seen: continue
        seen.add(key); deduped.append(e)
    return deduped

def find_auto_start(pages: List[Tuple[int, str]], probe_from: int = 0, min_entries: int = 8, window: int = 2) -> int:
    """Find first page where a small window yields enough entries."""
    for idx in range(probe_from, len(pages)-window):
        total = 0
        for j in range(window+1):
            _, text = pages[idx+j]
            lines = page_text_to_lines(text)
            total += len(parse_page(lines, pages[idx+j][0]+1))
        if total >= min_entries:
            return idx
    return probe_from

# ---------------- Output ----------------
def entries_to_wide_rows(entries: List[Entry], include_affix_entries=False) -> List[Dict[str, str]]:
    rows = []
    for e in entries:
        if not include_affix_entries and re.match(r"^[0-9\-]", e.word):
            continue
        for s in e.senses:
            if not include_affix_entries and s.pos in ("verbal affix","adjective formative","noun formative"):
                continue
            rows.append({
                "Word": e.word,
                "Part of speech": s.pos,
                "Affixation": "; ".join(s.affixes),
                "Meaning": s.meaning,
                "Example": s.ex_hil,
                "English example": s.ex_en,
                "page": str(e.page),
            })
    return rows

def entries_to_raw_rows(entries: List[Entry]) -> List[Dict[str, str]]:
    out = []
    for e in entries:
        for s in e.senses:
            out.append({
                "word": e.word,
                "pos": s.pos,
                "affixes": " | ".join(s.affixes),
                "meaning": s.meaning,
                "example_hil": s.ex_hil,
                "example_en": s.ex_en,
                "page": str(e.page),
                "raw": e.raw
            })
    return out

def save_rows(rows: List[Dict[str, str]], out_path: Path, schema: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_PANDAS:
        df = pd.DataFrame(rows)
        if out_path.suffix.lower() == ".json":
            df.to_json(out_path, orient="records", force_ascii=False, indent=2)
        else:
            # Force quoting to avoid column drift in Excel/consumers
            df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
        return
    # No pandas fallback
    if out_path.suffix.lower() == ".json":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        return
    # CSV fallback with strong quoting
    if rows:
        headers = list(rows[0].keys())
    else:
        headers = ["Word","Part of speech","Affixation","Meaning","Example","English example","page"] if schema.upper()=="WIDE" \
                  else ["word","pos","affixes","meaning","example_hil","example_en","page","raw"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ---------------- GUI ----------------
def try_import_tk():
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        return tk, ttk, filedialog, messagebox
    except Exception:
        return None, None, None, None

class AppGUI:
    def __init__(self, master):
        tk, ttk, filedialog, messagebox = self.tmods
        self.master = master
        self.master.title("Motus Hiligaynon Extractor (Hardened+++)")
        self.master.geometry("1080x700")
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        # State
        self.pdf_var = tk.StringVar()
        self.out_var = tk.StringVar()
        self.schema_var = tk.StringVar(value="WIDE")
        self.start_var = tk.StringVar(value="28")
        self.max_cont_var = tk.StringVar(value="12")
        self.include_affix_var = tk.BooleanVar(value=False)
        self.auto_start_var = tk.BooleanVar(value=False)
        self.fix_punct_var = tk.BooleanVar(value=True)
        self.strip_headers_var = tk.BooleanVar(value=True)

        root = ttk.Frame(self.master); root.grid(sticky="nsew", padx=12, pady=12)
        for c in range(3):
            root.columnconfigure(c, weight=1)

        # Row 0: Input
        ttk.Label(root, text="Input PDF").grid(row=0, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.pdf_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(root, text="Browse…", command=self.pick_pdf).grid(row=0, column=2, sticky="e")

        # Row 1: Output
        ttk.Label(root, text="Output file").grid(row=1, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.out_var).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(root, text="Save as…", command=self.pick_output).grid(row=1, column=2, sticky="e")

        # Options
        opt = ttk.LabelFrame(root, text="Options"); opt.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10,8))
        for c in range(8): opt.columnconfigure(c, weight=1)

        ttk.Label(opt, text="Schema").grid(row=0, column=0, sticky="w")
        ttk.Combobox(opt, textvariable=self.schema_var, values=["WIDE","RAW"], width=10, state="readonly").grid(row=0, column=1, sticky="w")

        ttk.Label(opt, text="Start page (1-based)").grid(row=0, column=2, sticky="w")
        ttk.Entry(opt, textvariable=self.start_var, width=10).grid(row=0, column=3, sticky="w")

        ttk.Label(opt, text="Max continuation").grid(row=0, column=4, sticky="w")
        ttk.Entry(opt, textvariable=self.max_cont_var, width=10).grid(row=0, column=5, sticky="w")

        ttk.Checkbutton(opt, text="Auto-start (find first lexicon page)", variable=self.auto_start_var).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6,0))
        ttk.Checkbutton(opt, text="Include affix/formative entries (va/aa/na, hyphen/number headwords)", variable=self.include_affix_var).grid(row=1, column=3, columnspan=3, sticky="w", pady=(6,0))

        ttk.Checkbutton(opt, text="Fix punctuation spacing", variable=self.fix_punct_var).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6,0))
        ttk.Checkbutton(opt, text="Strip running headers/footers", variable=self.strip_headers_var).grid(row=2, column=3, columnspan=3, sticky="w", pady=(6,0))

        # Run + Log
        runf = ttk.Frame(root); runf.grid(row=3, column=0, columnspan=3, sticky="ew")
        self.btn_run = ttk.Button(runf, text="Run Extraction", command=self.run)
        self.btn_run.pack(side="right")

        logf = ttk.LabelFrame(root, text="Log"); logf.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=(10,0))
        logf.rowconfigure(0, weight=1); logf.columnconfigure(0, weight=1)
        self.txt = tk.Text(logf, height=18, wrap="word")
        self.txt.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(logf, orient="vertical", command=self.txt.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.txt.configure(yscrollcommand=yscroll.set)

        self.log(f"PDF engines available: PyMuPDF={_HAS_PYMUPDF}, pdfminer.six={_HAS_PDFMINER}, PyPDF2={_HAS_PYPDF2}")
        self.log(f"Pandas available: {_HAS_PANDAS}")

    @property
    def tmods(self):
        return try_import_tk()

    def log(self, msg: str):
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.txt.update_idletasks()

    def pick_pdf(self):
        tk, ttk, filedialog, messagebox = self.tmods
        path = filedialog.askopenfilename(title="Select Motus PDF", filetypes=[("PDF files","*.pdf"),("All files","*.*")])
        if path:
            self.pdf_var.set(path)
            if not self.out_var.get():
                self.out_var.set(str(Path(path).with_name(Path(path).stem + "_extracted.csv")))

    def pick_output(self):
        tk, ttk, filedialog, messagebox = self.tmods
        path = filedialog.asksaveasfilename(
            title="Save output as…",
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("JSON","*.json"),("All files","*.*")]
        )
        if path:
            self.out_var.set(path)

    def run(self):
        tk, ttk, filedialog, messagebox = self.tmods
        in_path = self.pdf_var.get().strip()
        out_path = self.out_var.get().strip()
        schema = self.schema_var.get().upper()
        include_affix = bool(self.include_affix_var.get())
        auto_start = bool(self.auto_start_var.get())
        do_fix_punct = bool(self.fix_punct_var.get())
        do_strip_headers = bool(self.strip_headers_var.get())

        if not in_path or not Path(in_path).exists():
            messagebox.showerror("Missing input", "Please choose a valid input PDF.")
            return
        if not out_path:
            messagebox.showerror("Missing output", "Please choose where to save the output file.")
            return

        try:
            start_1b = int(self.start_var.get().strip())
            if start_1b < 1: raise ValueError
            start_idx = start_1b - 1
        except Exception:
            messagebox.showerror("Invalid input", "Start page must be a positive integer.")
            return

        try:
            max_cont = int(self.max_cont_var.get().strip())
            if max_cont < 1: raise ValueError
        except Exception:
            messagebox.showerror("Invalid input", "Max continuation must be a positive integer.")
            return

        self.btn_run.config(state="disabled")
        self.master.after(50, self._do_run, Path(in_path), Path(out_path), schema, include_affix, auto_start, start_idx, max_cont, do_fix_punct, do_strip_headers)

    def _do_run(self, in_path: Path, out_path: Path, schema: str, include_affix: bool, auto_start: bool, start_idx: int, max_cont: int, do_fix_punct: bool, do_strip_headers: bool):
        tk, ttk, filedialog, messagebox = self.tmods
        try:
            self.log(f"Reading PDF: {in_path}")
            pages = load_pdf_pages(in_path)
            self.log(f"Loaded {len(pages)} pages")

            if auto_start:
                start_idx = find_auto_start(pages, probe_from=0, min_entries=8, window=2)
                self.log(f"Auto-start chose page {start_idx+1} (1-based)")

            self.log(f"Parsing (start page={start_idx+1}, max_cont={max_cont}) …")
            entries = parse_pages(
                pages, start_index=start_idx, max_cont=max_cont,
                do_strip_headers=do_strip_headers, do_fix_punct=do_fix_punct, do_smart_space=True
            )
            self.log(f"Parsed {len(entries)} entries")

            if not entries:
                messagebox.showwarning("No entries", "No entries found. Adjust Start/Auto-start and try again.")
                self.btn_run.config(state="normal")
                return

            rows = entries_to_raw_rows(entries) if schema == "RAW" else entries_to_wide_rows(entries, include_affix_entries=include_affix)
            save_rows(rows, out_path, schema=schema)

            self.log(f"Saved → {out_path}")
            messagebox.showinfo("Done", f"Extraction complete.\nSaved to:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Extraction failed:\n{e}")
        finally:
            self.btn_run.config(state="normal")

# ---------------- CLI ----------------
def build_argparser():
    ap = argparse.ArgumentParser(description="Motus Hiligaynon Dictionary Extractor — HARDENED+++ + GUI")
    ap.add_argument("-i", "--input", help="Input PDF path")
    ap.add_argument("-o", "--output", help="Output file (.csv or .json)")
    ap.add_argument("--schema", choices=["WIDE","RAW"], default="WIDE", help="Output schema (default: WIDE)")
    ap.add_argument("--start", type=int, default=28, help="Start page (1-based, default 28). Ignored with --auto-start.")
    ap.add_argument("--auto-start", action="store_true", help="Auto-detect first dictionary page by entry density")
    ap.add_argument("--max-cont", type=int, default=12, help="Max continuation lines per entry (default 12)")
    ap.add_argument("--include-affixes", action="store_true", help="Include va/aa/na and hyphen/number headwords")
    ap.add_argument("--nogui", action="store_true", help="Headless mode (requires -i and -o)")
    ap.add_argument("--no-fix-punct", action="store_true", help="Disable punctuation spacing fixer")
    ap.add_argument("--no-strip-headers", action="store_true", help="Disable running header/footer scrub")
    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()

    # If --nogui or both -i and -o provided → CLI mode
    if args.nogui or (args.input and args.output):
        if not args.input or not args.output:
            print("For --nogui you must provide both --input and --output.", flush=True)
            return
        in_path = Path(args.input)
        out_path = Path(args.output)
        pages = load_pdf_pages(in_path)
        start_idx = find_auto_start(pages) if args.auto_start else max(0, args.start - 1)
        entries = parse_pages(
            pages, start_index=start_idx, max_cont=args.max_cont,
            do_strip_headers=(not args.no_strip_headers),
            do_fix_punct=(not args.no_fix_punct),
            do_smart_space=True
        )
        rows = entries_to_raw_rows(entries) if args.schema == "RAW" else entries_to_wide_rows(entries, include_affix_entries=args.include_affixes)
        save_rows(rows, out_path, schema=args.schema)
        print(f"Parsed {len(entries)} entries, saved → {out_path}")
        print(f"Start page used (1-based): {start_idx+1}")
        return

    # Otherwise, launch GUI
    tk = try_import_tk()[0]
    if not tk:
        print("Tkinter is not available in this environment. Run with --nogui and -i/-o.", flush=True)
        return
    root = tk.Tk()
    AppGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

