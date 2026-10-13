#!/usr/bin/env python3
"""
extract_items.py -- turn REB Pupil's Books and Teacher's Guides into a structured
table of End-of-Unit Assessment items and Additional-Activity items, with answers.

    python extract_items.py                      # defaults (see below)
    python extract_items.py <pdf_dir> <out.csv>  # custom folders

Default layout (paths are relative to the repo root, one level above this script in src/):
    data/raw/raw_pdfs/*.pdf (or *.txt)  ->  data/raw/items.csv
                                                data/raw/extraction_report.csv
Put the Pupil's Book AND the Teacher's Guide for the same class/subject in the same
folder. The script tells them apart from the page header ("Subject | Primary N |
Pupil's Book" / "... | Teacher's Guide"); if the header is missing it falls back to the
file name (e.g. P5_Mathematics_PB.pdf, P5_Mathematics_TG.pdf).

What comes from where
    Pupil's Book   -> the End-of-Unit Assessment QUESTIONS (stem, options, word bank...).
    Teacher's Guide-> the End-of-Unit Assessment ANSWERS ("II. Answers to the end of unit
                      assessment"), matched to the questions by unit + question number;
                      plus the Additional activities (Remedial / Consolidation /
                      Extended) as complete question+answer items (ids ...-R01, -C01, -E01).
    A Teacher's Guide on its own still works: you get the additional activities, and the
    end-of-unit answers as "answer only" rows (empty stem, review_flag question_text_missing).

Pupil's Book layout relied on: a heading "N.M. END OF UNIT ASSESSMENT", then sections
"A. Knowledge and understanding", "B. Skills", "C. Attitudes and values" with sequentially
numbered questions. Instruction lines ("Choose and circle the correct answer", "Fill in the
blanks using the provided terms: ...", "Answer by True or False") set the item type.

Teacher's Guide layout relied on: "N.M. End (of) unit assessment" -> "II. Answers to ... end
unit assessment" -> sections (with or without "A."/bullet) -> numbered answers ("1. B. Jack",
"10.Toolbox", or several on one line "1) a) B 2)d) Home"); then "N.M. Additional activities"
-> "Remedial / Consolidation / Extended activities" -> numbered questions -> "Answers to ...
activities" -> numbered answers.

Multiple-choice items get option_labels: one lowercase letter per detected option, e.g. four
options -> ["a", "b", "c", "d"]. For MCQ rows answer_key is that letter (lowercase).

No LLM is used: every rule below is a regex you can read and change. Check the printed
diagnostics and the review_flag column before trusting the output for a new book.
"""
import csv
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter

HDR = re.compile(r"^(?:(?:\d+|[ivxlc]+)\s+)?(?P<subj>.+?)\s*\|\s*Primary\s*(?P<cls>\d)\s*\|\s*"
                 r"(?P<kind>[Pp]upil.?s [Bb]ook|[Tt]eacher.?s [Gg]uide)(?:\s+\d+)?\s*$")
UNIT_TOC = re.compile(r"^(?:UNIT|Unit)\s+(\d+)\s*:?\s+(.+?)\s+\d+$")
END_HDR = re.compile(r"^(\d+)\.\d+\.?\s*END (?:OF )?UNIT ASSESSMENT\s*$", re.I)
SEC_HDR = re.compile(r"^([ABC])\.\s*(Knowledge and understanding|Skills|Attitudes and values)\s*$", re.I)
# Teacher's Guide section headings: letter and bullet are optional ("A. Skills", "• Skills", "Skills")
SEC_TG = re.compile(r"^(?:[\u2022\-\u2013]\s*)?(?:[ABC][.)]\s*)?"
                    r"(Knowledge and understanding|Skills|Attitudes and values)\s*:?\s*$", re.I)
NUM = re.compile(r"^(\d+)\s*[.)]\s*(.*)$")
INSTR_FILL = re.compile(r"^Fill in the blanks?\b[^:]*:?\s*(.*)$", re.I)
INSTR_MCQ = re.compile(r"^Choose (?:and circle )?the correct answers?\b", re.I)
INSTR_TF = re.compile(r"^Answer by true or false", re.I)
INSTR_OPEN = re.compile(r"^Answer the questions?\b", re.I)
STOP = re.compile(r"^(Environmental action message|Key unit competence|Bibliography)", re.I)
GRID = re.compile(r"^(?:[A-Z]\s+){4,}[A-Z]$")
BLANK = re.compile(r"_{3,}|[.\u2026]{4,}|\u2026{2,}")
OPT = re.compile(r"(?:(?<=\s)|^)([A-Fa-f])[.)]\s+")
VISUAL = re.compile(r"\b(diagram|picture|image|figure|draw|drawing|sketch|label|puzzle|below|as shown)\b", re.I)
COMPUTER_I = re.compile(r"scratch|turtle|excel|gnumeric|abiword|spreadsheet|worksheet|workbook|pictoblox|machinelearningforkids", re.I)
COMPUTER_C = re.compile(r"\bWord\b")

# --- Teacher's Guide patterns ---
ANS_HDR = re.compile(r"^(?:II\.\s*)?Answers?\s+to\s+(?:the\s+)?end\b.*assessment", re.I)
ADD_TITLE = re.compile(r"^(\d+)\.\d+\.?\s*Additional activities\s*:?\s*$", re.I)
ADD_SUB = re.compile(r"^(?:\(?[A-Ca-c][.)]\s*)?(Remedial|Consolidation|Extended|Extension)\s+Activit(?:y|ies)\b.*$", re.I)
ADD_ANS = re.compile(r"^Answers?\s+to\s+(?:the\s+)?(Remedial|Consolidation|Extended|Extension)\s+Activit(?:y|ies)\b.*$", re.I)
TG_STOP = re.compile(r"^(?:Unit\s+\d+|BIBLIOGRAPHY)\s*$", re.I)   # next unit's title page / end of book
UPPER = re.compile(r"^(?:[A-Z][A-Z\s&,]+|UNIT \d+)$")              # unit-title lines left at a block's tail
BULLET = re.compile(r"^[\u2022\-\u2013\u00b7]\s*")
SKIP_LINE = re.compile(r"^(?:Choose (?:and circle )?the correct answers?|Questions?)\s*:?\s*$", re.I)
LETTER_ANS = re.compile(r"^([A-Fa-f])\s*(?:[.)]\s*|$)(.*)$")       # "B. Jack", "b) Anther", "C"

SECTION_NAMES = {"A": "Knowledge and understanding", "B": "Skills", "C": "Attitudes and values"}
SECTION_BY_NAME = {"knowledge and understanding": "Knowledge and understanding",
                   "skills": "Skills", "attitudes and values": "Attitudes and values"}
ADD_CODE = {"Remedial": "R", "Consolidation": "C", "Extended": "E"}
CODE_ORDER = {"Q": 0, "R": 1, "C": 2, "E": 3}

FIELDS = ["item_id", "subject", "class_level", "unit_no", "unit_title", "section", "q_no",
          "item_type", "stem", "options", "option_labels", "n_options", "word_bank", "n_blanks",
          "n_subitems", "needs_visual", "needs_computer", "sensitive_topic", "n_words_stem",
          "sms_text", "sms_chars", "sms_segments", "uses_unicode",
          "answer_key", "answer_text", "answer_key_source", "stem_source", "review_flag"]


def _have(cmd):
    return shutil.which(cmd) is not None


def pdf_text(pdf):
    """Text of a .txt export, or of a PDF via pdftotext (poppler) with a pypdf fallback."""
    if pdf.lower().endswith(".txt"):
        out = open(pdf, encoding="utf-8", errors="replace").read()
    elif _have("pdftotext"):
        out = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True, check=True).stdout
    else:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("Need poppler's pdftotext or `pip install pypdf` to read PDFs "
                               "(or export the book to .txt)")
        out = "\n\x0c\n".join((p.extract_text() or "") for p in PdfReader(pdf).pages)
    # Wingdings bullets come through as private-use glyphs (e.g. before "Answers to ... activities")
    out = re.sub(r"[\ue000-\uf8ff]", "", out)
    return [l.strip() for l in out.replace("\x0c", "\n").split("\n")]


def pdf_pages(pdf):
    if pdf.lower().endswith(".txt"):
        return None
    try:
        if _have("pdfinfo"):
            info = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout
            return int(re.search(r"Pages:\s+(\d+)", info).group(1))
        from pypdf import PdfReader
        return len(PdfReader(pdf).pages)
    except Exception:
        return None


def strip_headers(lines):
    meta, kept = {}, []
    for l in lines:
        m = HDR.match(l)
        if m:
            meta.setdefault("class_level", int(m.group("cls")))
            meta.setdefault("subject", m.group("subj").strip())
            meta.setdefault("doc_kind", "TG" if m.group("kind").lower().startswith("teacher") else "PB")
            continue
        kept.append(l)
    return kept, meta


def unit_titles(lines):
    t = {}
    for l in lines:
        m = UNIT_TOC.match(l)
        if m:
            t.setdefault(int(m.group(1)), m.group(2).strip().title())
    return t


# ----------------------------------------------------------------------------------------
# Pupil's Book: end-of-unit questions
# ----------------------------------------------------------------------------------------
def find_blocks(lines):
    out = {}
    for i, ln in enumerate(lines):
        m = END_HDR.match(ln)
        if not m:
            continue
        unit = int(m.group(1))
        j = next((k for k in range(i + 1, min(i + 200, len(lines)))
                  if SEC_HDR.match(lines[k]) and lines[k][0].upper() == "A"), None)
        if j is None or unit in out:
            continue
        k = next((k for k in range(j + 1, len(lines))
                  if STOP.match(lines[k]) or END_HDR.match(lines[k])), len(lines))
        out[unit] = lines[j:k]
    return out


def parse_block(block):
    items, cur = [], None
    section, mode, bank, collecting, expected = None, "OPEN", "", False, 1
    for ln in block:
        if not ln or GRID.match(ln):
            continue
        m = SEC_HDR.match(ln)
        if m:
            if cur:
                items.append(cur)
                cur = None
            section = SECTION_NAMES[m.group(1).upper()]
            mode, bank, collecting = "OPEN", "", False
            continue
        n = NUM.match(ln)
        if n and int(n.group(1)) == expected:
            if cur:
                items.append(cur)
            collecting = False
            cur = {"q_no": expected, "section": section, "mode": mode,
                   "bank": bank if mode == "FILL" else "",
                   "lines": [n.group(2)] if n.group(2) else []}
            expected += 1
            continue
        if INSTR_TF.match(ln) or INSTR_MCQ.match(ln) or INSTR_OPEN.match(ln):
            if cur:
                items.append(cur)
                cur = None
            mode = "TF" if INSTR_TF.match(ln) else "MCQ" if INSTR_MCQ.match(ln) else "OPEN"
            bank, collecting = "", False
            continue
        f = INSTR_FILL.match(ln)
        if f:
            if cur:
                items.append(cur)
                cur = None
            mode, bank, collecting = "FILL", f.group(1), True
            continue
        if collecting:
            bank += " " + ln
        elif cur is not None:
            cur["lines"].append(ln)
    if cur:
        items.append(cur)
    return items


# ----------------------------------------------------------------------------------------
# Teacher's Guide: end-of-unit answers and additional activities
# ----------------------------------------------------------------------------------------
def sec_name(ln):
    m = SEC_TG.match(ln)
    return SECTION_BY_NAME[m.group(1).lower()] if m else None


def trim_tail(block):
    """Drop blank lines and the next unit's title ("WORD PROCESSING", "AND SPREADSHEET") from a block's tail."""
    block = list(block)
    while block and (not block[-1] or UPPER.match(block[-1])):
        block.pop()
    return block


def find_tg_blocks(lines):
    """unit -> {"answers": [...lines...], "additional": [...lines...]} (either key may be missing)."""
    out = {}
    n = len(lines)

    def end_of(start, extra=()):
        return next((k for k in range(start, n)
                     if TG_STOP.match(lines[k]) or END_HDR.match(lines[k]) or ADD_TITLE.match(lines[k])
                     or any(p.match(lines[k]) for p in extra)), n)

    for i, ln in enumerate(lines):
        m = END_HDR.match(ln)
        if m:
            unit = int(m.group(1))
            if "answers" in out.get(unit, {}):
                continue
            stop = end_of(i + 1, (ANS_HDR,))          # the answers heading must come before the next unit/heading
            if stop >= n or not ANS_HDR.match(lines[stop]):
                continue
            # start at the first "Knowledge and understanding" heading: skips the project answers
            s = next((k for k in range(stop + 1, min(stop + 300, n))
                      if sec_name(lines[k]) == "Knowledge and understanding"), None)
            if s is None:
                continue
            out.setdefault(unit, {})["answers"] = lines[s:end_of(s + 1)]
            continue
        m = ADD_TITLE.match(ln)
        if m:
            unit = int(m.group(1))
            if "additional" not in out.get(unit, {}):
                out.setdefault(unit, {})["additional"] = lines[i + 1:end_of(i + 1)]
    return out


def split_inline(text, expected):
    """'a) B 2)d) Home 3)c) X' (expected=2) -> ['a) B', 'd) Home', 'c) X']: several answers on one line."""
    parts, pos, nxt = [], 0, expected
    for m in re.finditer(r"(?<=\s)(\d+)\)\s*", text):
        if int(m.group(1)) == nxt:
            parts.append(text[pos:m.start()].strip())
            pos = m.end()
            nxt += 1
    parts.append(text[pos:].strip())
    return parts


def parse_answers(block):
    """End-of-unit answers -> {q_no: {"text": str, "section": str}}. Numbering runs on across sections."""
    out, cur, section, expected = {}, None, None, 1

    def close():
        nonlocal cur
        if cur is not None:
            out[cur["q_no"]] = {"text": re.sub(r"\s+", " ", " ".join(cur["lines"])).strip(),
                                "section": cur["section"]}
        cur = None

    for ln in block:
        if not ln or GRID.match(ln) or SKIP_LINE.match(ln):
            continue
        s = sec_name(ln)
        if s:
            close()
            section = s
            continue
        n = NUM.match(ln)
        if n and int(n.group(1)) == expected:
            for part in split_inline(n.group(2), expected + 1):
                close()
                cur = {"q_no": expected, "section": section, "lines": [part] if part else []}
                expected += 1
            continue
        if cur is not None:
            cur["lines"].append(ln)
    close()
    return out


def parse_additional(block):
    """Additional activities -> [{"kind": "Remedial"|"Consolidation"|"Extended", "Q": {n: lines}, "A": {n: lines}}]."""
    subs, sub, cur, phase, expected = [], None, None, "Q", 1

    def kind_of(word):
        return "Extended" if word.lower().startswith("exten") else word.capitalize()

    def close():
        nonlocal cur
        if cur is not None and sub is not None:
            sub[phase][cur["n"]] = cur["lines"]
        cur = None

    for ln in block:
        if not ln or GRID.match(ln) or SKIP_LINE.match(ln):
            continue
        m = ADD_ANS.match(ln)
        if m:
            close()
            kind = kind_of(m.group(1))
            sub = next((s for s in reversed(subs) if s["kind"] == kind), None)
            if sub is None:
                sub = {"kind": kind, "Q": {}, "A": {}}
                subs.append(sub)
            phase, expected = "A", 1
            continue
        m = ADD_SUB.match(ln)
        if m:
            close()
            sub = {"kind": kind_of(m.group(1)), "Q": {}, "A": {}}
            subs.append(sub)
            phase, expected = "Q", 1
            continue
        if sub is None:
            continue
        n = NUM.match(BULLET.sub("", ln))
        # while reading answers never go past the last question number: a numbered list INSIDE
        # an answer ("1. Use less cars ... 7. Reduce waste") must not become answers 3..7
        if n and int(n.group(1)) == expected and (phase == "Q" or expected <= len(sub["Q"])):
            close()
            cur = {"n": expected, "lines": [n.group(2)] if n.group(2) else []}
            expected += 1
            continue
        if cur is not None:
            cur["lines"].append(ln)
    close()
    return subs


# ----------------------------------------------------------------------------------------
# Item building
# ----------------------------------------------------------------------------------------
def find_chain(text):
    chain, want = [], "a"
    for m in OPT.finditer(text):
        if m.group(1).lower() == want:
            chain.append(m)
            want = chr(ord(want) + 1)
            if want > "f":
                break
    return chain if len(chain) >= 2 else []


def to_ascii(s):
    return (s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"')
             .replace("\u201d", '"').replace("\u2013", "-").replace("\u2014", "-")
             .replace("\u2026", "..."))


def sms_stats(txt):
    txt = to_ascii(txt)
    ucs = any(ord(c) > 127 for c in txt)
    single, multi = (70, 67) if ucs else (160, 153)
    n = len(txt)
    return n, (1 if n <= single else math.ceil(n / multi)), ucs


def render(label, stem, options, qtype, bank):
    s = f"{label}. {stem}"
    if options:
        s += " " + " ".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(options))
    if qtype == "TF" and "true or false" not in stem.lower():
        s += " (True/False)"
    if bank:
        s += " Words: " + ", ".join(bank)
    return s


def subject_code(subj, fallback):
    subj = subj or fallback
    caps = "".join(w[0] for w in subj.split() if w[0].isupper())
    if len(subj.split()) == 1 or len(caps) < 2:
        return re.sub(r"[^A-Za-z]", "", subj)[:4].upper() or "SUBJ"
    return caps


def make_id(meta, unit_no, code, q_no):
    return f"{subject_code(meta.get('subject', ''), '')}-P{meta.get('class_level', 0)}-U{unit_no:02d}-{code}{q_no:02d}"


def finish(it, unit_no, unit_title, meta, last, source=""):
    """Turn a raw parsed item into an output row. it: q_no, section, mode, bank, lines (+ code, min_chain)."""
    lines = list(it["lines"])
    code = it.get("code", "Q")
    if last:
        while lines and (re.fullmatch(r"[A-Z][A-Z\s&]+", lines[-1]) or re.fullmatch(r"UNIT \d+", lines[-1])):
            lines.pop()
    bank_raw, inline = it["bank"], False
    if lines:
        fm = INSTR_FILL.match(lines[0])
        if fm:
            inline, bank_raw, lines = True, fm.group(1), lines[1:]
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    chain = find_chain(text)
    mode = it["mode"]

    if inline:
        qtype = "FILL"
    elif re.search(r"true or false", text, re.I) or (mode == "TF" and not text.rstrip().endswith("?")):
        qtype = "TF"
    elif re.search(r"\bmatch\b", text, re.I):
        qtype = "MATCH"
    elif len(chain) >= it.get("min_chain", 2):
        qtype = "MCQ"
    elif BLANK.search(text):
        qtype = "FILL"
    else:
        qtype = "OPEN"

    stem, options = text, []
    if qtype == "MCQ":
        stem = text[:chain[0].start()].strip()
        options = [text[m.end(): (chain[i + 1].start() if i + 1 < len(chain) else len(text))].strip()
                   for i, m in enumerate(chain)]
    labels = [chr(97 + i) for i in range(len(options))]          # 4 options -> a, b, c, d
    bank = []
    if qtype == "FILL":
        bank = [w.strip(" .") for w in bank_raw.split(",") if w.strip(" .")]
    n_blanks = len(BLANK.findall(text)) if qtype == "FILL" else 0
    n_sub = len(chain) if qtype in ("TF", "FILL") else 0

    sms_text = render(f"{code}{it['q_no']}", stem, options, qtype, bank)
    chars, segs, ucs = sms_stats(sms_text)
    return {
        "item_id": make_id(meta, unit_no, code, it["q_no"]),
        "subject": meta.get("subject", ""), "class_level": meta.get("class_level"),
        "unit_no": unit_no, "unit_title": unit_title, "section": it["section"], "q_no": it["q_no"],
        "item_type": qtype, "stem": stem, "options": json.dumps(options, ensure_ascii=False),
        "option_labels": json.dumps(labels), "n_options": len(options),
        "word_bank": json.dumps(bank, ensure_ascii=False),
        "n_blanks": n_blanks, "n_subitems": n_sub,
        "needs_visual": bool(VISUAL.search(text)),
        "needs_computer": bool(COMPUTER_I.search(text) or COMPUTER_C.search(text)),
        "sensitive_topic": bool(re.search(r"human\s+reproduct", unit_title, re.I)),
        "n_words_stem": len(stem.split()),
        "sms_text": sms_text, "sms_chars": chars, "sms_segments": segs, "uses_unicode": ucs,
        "answer_key": "", "answer_text": "", "answer_key_source": "", "stem_source": source,
        "review_flag": "",
    }


def norm_answer(text, qtype, n_options=0):
    """-> (answer_key, answer_text, flag). MCQ: lowercase letter; TF: True/False; else the text."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return "", "", "no_answer"
    if qtype == "MCQ":
        m = LETTER_ANS.match(text)
        if m:
            key = m.group(1).lower()
            flag = "answer_letter_outside_options" if n_options and ord(key) - 96 > n_options else ""
            return key, m.group(2).strip(), flag
        return text, text, "mcq_answer_not_a_letter"
    if qtype == "TF":
        toks = re.findall(r"\b(true|false)\b", text, re.I)
        if toks:                                  # several statements -> "False, True, False"
            return ", ".join(t.capitalize() for t in toks), text, ""
        return text, text, "tf_answer_not_true_or_false"
    return text, text, ""


def guess_type(answer_text):
    t = answer_text.strip()
    if re.fullmatch(r"(?i)(true|false)\.?", t):
        return "TF"
    return "MCQ" if LETTER_ANS.match(t) else "OPEN"


def add_flag(row, flag):
    if flag:
        row["review_flag"] = ";".join(x for x in (row["review_flag"], flag) if x)


def meta_from_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"P(?:rimary)?[\s_-]*(\d)[\s_-]+(.+)", stem, re.I)
    if not m:
        return {}
    return {"class_level": int(m.group(1)), "subject": re.sub(r"[_-]+", " ", m.group(2)).replace("PB", "").strip()}


def process_book(path):
    """-> (rows, answers, report). answers: {item_id: {...}} for end-of-unit answers found in a Teacher's Guide."""
    fname = os.path.basename(path)
    raw_lines = pdf_text(path)
    lines, meta = strip_headers(raw_lines)
    for k, v in meta_from_name(path).items():
        meta.setdefault(k, v)
    kind = meta.get("doc_kind") or ("TG" if re.search(r"(?<![A-Za-z])TG(?![A-Za-z])|teacher", fname, re.I) else "PB")
    titles = unit_titles(lines)
    rows, answers, units, n_add = [], {}, 0, 0

    if kind == "PB":
        blocks = find_blocks(lines)
        units = len(blocks)
        for unit in sorted(blocks):
            parsed = parse_block(blocks[unit])
            for i, it in enumerate(parsed):
                r = finish(it, unit, titles.get(unit, f"Unit {unit}"), meta, i == len(parsed) - 1, fname)
                r["_kind"] = "end"
                rows.append(r)
    else:
        tg = find_tg_blocks(lines)
        units = len(tg)
        for unit in sorted(tg):
            utitle = titles.get(unit, f"Unit {unit}")
            for q, a in parse_answers(trim_tail(tg[unit].get("answers", []))).items():
                answers[make_id(meta, unit, "Q", q)] = {**a, "unit": unit, "q_no": q, "meta": dict(meta),
                                                        "unit_title": utitle, "source": fname}
            for sub in parse_additional(trim_tail(tg[unit].get("additional", []))):
                code = ADD_CODE[sub["kind"]]
                flag = ""
                if len(sub["Q"]) != len(sub["A"]):
                    flag = f"count_mismatch(q={len(sub['Q'])},a={len(sub['A'])})"
                for q in sorted(sub["Q"]):
                    ans_text = re.sub(r"\s+", " ", " ".join(sub["A"].get(q, []))).strip()
                    it = {"q_no": q, "section": f"Additional activities - {sub['kind']}", "mode": "OPEN",
                          "bank": "", "lines": sub["Q"][q], "code": code,
                          # the guide has no real MCQs in these sections; accept one only if the answer is a letter
                          "min_chain": 3 if LETTER_ANS.match(ans_text) else 99}
                    r = finish(it, unit, utitle, meta, False, fname)
                    key, txt, f2 = norm_answer(ans_text, r["item_type"], r["n_options"])
                    r.update(answer_key=key, answer_text=txt, answer_key_source=f"{fname}: additional activities")
                    add_flag(r, f2)
                    add_flag(r, flag)
                    r["_kind"] = "add"
                    rows.append(r)
                    n_add += 1

    key_hits = sum(len(re.findall(r"answer key|marking (?:scheme|guide)|model answers?", l, re.I)) for l in raw_lines)
    found = len(rows) + len(answers)
    rep = {"file": fname, "doc_kind": kind, "subject": meta.get("subject"), "class_level": meta.get("class_level"),
           "pages": pdf_pages(path), "units_found": units, "items": len(rows),
           **{f"n_{t}": sum(r["item_type"] == t for r in rows) for t in ("MCQ", "TF", "FILL", "MATCH", "OPEN")},
           "tg_answers_found": len(answers), "additional_items": n_add, "answer_key_mentions": key_hits,
           "status": "ok" if found else "SKIPPED: nothing parsed (no end-of-unit assessments / answers found)"}
    return rows, answers, rep


def orphan_row(aid, a):
    """An end-of-unit answer with no matching question (no Pupil's Book loaded, or numbering differs)."""
    r = {f: "" for f in FIELDS}
    meta = a["meta"]
    qtype = guess_type(a["text"])
    key, txt, _ = norm_answer(a["text"], qtype)
    r.update(item_id=aid, subject=meta.get("subject", ""), class_level=meta.get("class_level"),
             unit_no=a["unit"], unit_title=a["unit_title"], section=a["section"] or "", q_no=a["q_no"],
             item_type=qtype, stem="", options="[]", option_labels="[]", n_options=0, word_bank="[]",
             n_blanks=0, n_subitems=0, needs_visual=False, needs_computer=False,
             sensitive_topic=bool(re.search(r"human\s+reproduct", a["unit_title"], re.I)), n_words_stem=0,
             sms_text="", sms_chars=0, sms_segments=0, uses_unicode=False,
             answer_key=key, answer_text=txt, answer_key_source=f"{a['source']}: end-of-unit answers",
             stem_source="", review_flag="question_text_missing")
    r["_kind"] = "orphan"
    return r


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) >= 3:
        pdf_dir, out = sys.argv[1], sys.argv[2]
    else:
        pdf_dir, out = os.path.join(root, "data", "raw", "raw_pdfs"), os.path.join(root, "data", "raw", "items.csv")
    files = sorted(p for p in glob.glob(os.path.join(pdf_dir, "*")) if p.lower().endswith((".pdf", ".txt")))
    if not files:
        sys.exit(f"No .pdf or .txt files found in {pdf_dir}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    all_rows, answers, reports = [], {}, []
    for p in files:
        try:
            rows, ans, rep = process_book(p)
        except Exception as e:  # keep going; record the failure
            rows, ans, rep = [], {}, {"file": os.path.basename(p), "items": 0,
                                      "status": f"SKIPPED: {type(e).__name__}: {e}"}
        if rep["status"] != "ok":
            print(f"WARNING: {rep['file']} -> {rep['status']}", file=sys.stderr)
        all_rows += rows
        answers.update(ans)
        reports.append(rep)

    # attach Teacher's Guide answers to the Pupil's Book questions (same subject + class + unit + q_no)
    by_id = {r["item_id"]: r for r in all_rows if r["_kind"] == "end"}
    for aid in list(answers):
        r = by_id.get(aid)
        if r is None:
            continue
        a = answers.pop(aid)
        key, txt, flag = norm_answer(a["text"], r["item_type"], r["n_options"])
        r.update(answer_key=key, answer_text=txt, answer_key_source=f"{a['source']}: end-of-unit answers")
        add_flag(r, flag)
    for r in by_id.values():
        if not r["answer_key"]:
            r["answer_key_source"] = "not found (add the Teacher's Guide to the folder)"
            add_flag(r, "no_answer")
    all_rows += [orphan_row(aid, a) for aid, a in answers.items()]

    all_rows.sort(key=lambda r: (r["subject"], r["class_level"] or 0, r["unit_no"],
                                 CODE_ORDER.get(r["item_id"].split("-")[-1][0], 9), r["q_no"]))
    seen = Counter(r["item_id"] for r in all_rows)
    dups = [k for k, v in seen.items() if v > 1]
    if dups:
        print(f"WARNING: {len(dups)} duplicate item_id values (two books with the same class+subject?): {dups[:5]}", file=sys.stderr)

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    rep_path = os.path.join(os.path.dirname(os.path.abspath(out)), "extraction_report.csv")
    cols = list(dict.fromkeys(k for r in reports for k in r))
    with open(rep_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(reports)

    end = [r for r in all_rows if r["_kind"] == "end"]
    add = [r for r in all_rows if r["_kind"] == "add"]
    print(f"files read: {len(files)}   rows written: {len(all_rows)}   -> {out}")
    for r in reports:
        print(f"  {r['file']} [{r.get('doc_kind', '?')}]: {r.get('items', 0)} items, "
              f"{r.get('tg_answers_found', 0)} end-unit answers, {r.get('units_found', 0)} units  [{r['status']}]")
    print(f"end-of-unit items: {len(end)} ({sum(bool(r['answer_key']) for r in end)} with an answer)   "
          f"additional-activity items: {len(add)}   answer-only rows: {sum(r['_kind'] == 'orphan' for r in all_rows)}")
    print("per type:", dict(Counter(r["item_type"] for r in all_rows)))
    flags = Counter(x.split("(")[0] for r in all_rows for x in r["review_flag"].split(";") if x)
    if flags:
        print("review_flag counts:", dict(flags))
    print(f"report: {rep_path}")


if __name__ == "__main__":
    main()