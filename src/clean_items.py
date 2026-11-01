#!/usr/bin/env python3
"""
clean_items.py -- clean the raw extraction (data/raw/items.csv) and keep only the multiple-choice (MCQ)
and true/false (TF) questions, with the question and the answer in separate tables.

    python src/clean_items.py                       # data/raw/items.csv -> data/processed/
    python src/clean_items.py <in.csv> <out_dir>

Outputs (data/processed/):
    items_clean.csv    every MCQ/TF row after cleaning, with flags and exclude_reason (for the audit)
    questions.csv      the rows usable for modelling: the QUESTION only (no answer anywhere in it)
    answers.csv        the matching ANSWERS, joined to questions.csv by item_id
    cleaning_log.csv   every change: item_id, rule, field, before, after

Every change is a rule you can read and defend. The raw file is never modified.

Rules, in the order they run:
  R1  strip page numbers glued to stems, options and answers; strip section markers (II., III.)
  R2  strip leaked instructions ("State whether ... true or false:", trailing "True or False"); retype the
      statements they introduce as TF
  R3  turn inline options "(a) X (b) Y" into real options and make the item an MCQ
  R4  split a true/false item that bundles several statements into one row per statement
  R5  validate and normalise the answer key (MCQ: a letter inside the options; TF: True/False);
      an untrustworthy key is set to missing and flagged, never guessed
  R0  rows that are not MCQ or TF after R1-R4 are left out of the outputs (logged, counted)
  R6  recompute the length and SMS columns from the cleaned text; rows with no question text get missing, not 0
  R7  unit titles, sensitive-topic flag, unique ids, duplicate-question groups
  R8  mark the rows usable for modelling (exclude_reason says why a row is not)
"""
import hashlib
import json
import os
import re
import sys

import pandas as pd

try:
    from .extract_items import render, sms_stats
except ImportError:  # run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from extract_items import render, sms_stats

KEEP_TYPES = ("MCQ", "TF")
BLANK = re.compile(r"_{2,}|\.{4,}|…{2,}")
PAGE_END = re.compile(r"(?<=[.?:)_!…])\s+\d{1,3}\s*$")
PAGE_MID = re.compile(r"(?<=[.?:!])\s+\d{2,3}\s+(?=[A-Z(•])")
PAGE_WORD = re.compile(r"(?<=[A-Za-z.])\s+\d{2,3}(?:\s+\d{2,3})*\s*$")
SECTION_MARK = re.compile(r"\s+[IVX]{1,4}\.\s*$")
JUNK = re.compile(r"[•●♦�‣]")
NOT_ANSWER = re.compile(r"^(?:refer to|accept|pupils?\b|learners?\b|answers? (?:may|will|vary)|possible answers?|note:|see )", re.I)
TF_INSTR_END = re.compile(r"\s*(?:State whether the following (?:statements )?(?:are )?true or false|"
                          r"Answer by true or false|Write true or false|Read each statement and write true or false)\s*[:.]?\s*$", re.I)
TF_INSTR_START = re.compile(r"^\s*(?:True or False|Answer by true or false|Read each statement and write true or false|"
                            r"Write true or false|State whether[^:.]*)\s*[:.]\s*", re.I)
TF_SUFFIX = re.compile(r"\s*\(?\s*True\s*/\s*False\s*\)?\s*[.?]?\s*$|\s+True or False\s*[.:?]?\s*$", re.I)
SECTION_TAIL = re.compile(r"\s+[ABC]\.\s*(?:Knowledge and understanding|Skills|Attitudes\s*(?:and|&)\s*values)\s*$", re.I)
OTHER_INSTR_END = re.compile(r"\s*(?:Answer the following questions?(?: briefly)?|Choose (?:and circle )?the correct answers?)\s*[:.]?\s*$", re.I)
ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]
SENSITIVE_WORDS = re.compile(r"human reproduct|menstru|circumcis|foreskin|genital|pubert|pregnan|sperm|\bpenis\b|\bvagina|\bscrotum|\buterus\b", re.I)
P6_TITLES = {1: "Mechanics Tools", 2: "Simple Machines", 3: "Air Pollution", 4: "Cows and Goats Management",
             5: "Word Processing and Spreadsheet", 6: "Introduction to Emerging Technologies", 7: "Programming for Children",
             8: "Plants Reproduction", 9: "Sustainable Waste Management", 10: "Human Circulatory System",
             11: "Human Respiratory System", 12: "Human Reproductive System", 13: "Energy Management", 14: "Magnetism"}

QUESTION_COLS = ["item_id", "group_id", "subject", "class_level", "unit_no", "unit_title", "section", "item_type",
                 "question", "options", "n_options", "n_words_question", "sms_text", "sms_chars", "sms_segments",
                 "uses_unicode", "sensitive_topic"]
ANSWER_COLS = ["item_id", "answer_key", "answer_text", "answer_source"]

LOG = []


def log(item_id, rule, field, before, after):
    LOG.append({"item_id": item_id, "rule": rule, "field": field,
                "before": "" if before is None else str(before)[:300], "after": "" if after is None else str(after)[:300]})


def sp(t):
    return re.sub(r"\s+", " ", t).strip() if isinstance(t, str) else t


def nz(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def add_flag(r, f):
    cur = [x for x in str(r.get("clean_flags") or "").split(";") if x]
    if f not in cur:
        cur.append(f)
    r["clean_flags"] = ";".join(cur)


def norm_txt(t):
    """Lower case, no punctuation, no plural 's': lets the key text 'Eyes' match the option 'Eye'."""
    t = re.sub(r"[^a-z0-9 ]+", "", str(t).lower()).strip()
    return re.sub(r"(?<=[a-z]{3})(?:es|s)$", "", t)


# ---------------------------------------------------------------- marker parsing
def chain_positions(text, kind):
    """Consecutive markers in order: 'upper' A. B. | 'lower' a. b. | 'paren' (a) (b) | 'roman' i) ii)."""
    if kind == "roman":
        pats = [re.compile(r"(?:(?<=\s)|^)\(?%s[.)]\s" % re.escape(x)) for x in ROMAN]
    elif kind == "paren":
        pats = [re.compile(r"(?:(?<=\s)|^)\(%s\)\s*" % c) for c in "abcdef"]
    else:
        letters = "ABCDEF" if kind == "upper" else "abcdef"
        pats = [re.compile(r"(?:(?<=\s)|^)\(?%s[.)]\s" % c) for c in letters]
    out, start = [], 0
    for p in pats:
        m = p.search(text, start)
        if not m:
            break
        out.append(m)
        start = m.end()
    return out if len(out) >= 2 else []


def any_chain(text, kinds=("upper", "lower", "paren", "roman")):
    best = []
    for k in kinds:
        c = chain_positions(text, k)
        if len(c) > len(best):
            best = c
    return best


def split_by(text, chain):
    head = text[:chain[0].start()].strip()
    parts = [text[m.end(): (chain[i + 1].start() if i + 1 < len(chain) else len(text))].strip(" ;:") for i, m in enumerate(chain)]
    return head, parts


def tf_key_parts(key, n):
    """'False, True' -> ['False', 'True'] when there are exactly n of them, else None."""
    if not isinstance(key, str):
        return None
    words = re.findall(r"\b(true|false)\b", key, re.I)
    rest = re.sub(r"\b(true|false)\b|[,;&\s]|and", "", key, flags=re.I)
    return [w.capitalize() for w in words] if not rest and len(words) == n else None


# ---------------------------------------------------------------- row-level rules
def r1_page_numbers(r):
    for f in ("stem", "answer_key", "answer_text"):
        v = nz(r.get(f))
        if isinstance(v, str):
            new = PAGE_MID.sub(" ", PAGE_END.sub("", v))
            if len(new.split()) > 3 and (f != "stem" or len(new.split()) > 4):
                new = PAGE_WORD.sub("", new)
            if f == "stem":
                new = SECTION_MARK.sub("", new)
            new = sp(new)
            if new != v:
                log(r["item_id"], "R1_page_number", f, v[-40:], new[-40:])
                r[f] = new
    return r


def clean_options(r, rule):
    """Strip a page number glued to the last option ("Nothing happens 218") and tidy whitespace."""
    opts = json.loads(r.get("options") or "[]")
    if not opts:
        return r
    new = [sp(JUNK.sub(" ", o)) for o in opts]
    last = PAGE_WORD.sub("", new[-1]) if len(new[-1].split()) > 1 else new[-1]
    new[-1] = PAGE_END.sub("", last).strip() or new[-1]
    if new != opts:
        log(r["item_id"], rule, "options", opts[-1][-40:], new[-1][-40:])
        r["options"] = json.dumps(new, ensure_ascii=False)
    return r


def r2_instructions(r):
    """Remove leaked instruction text; return (row, 'TF' when the instruction introduces true/false statements)."""
    mode, s = None, nz(r.get("stem"))
    if not isinstance(s, str):
        return r, mode
    new = s
    if TF_INSTR_END.search(new):
        new, mode = TF_INSTR_END.sub("", new).strip(), "TF"
    elif OTHER_INSTR_END.search(new):
        new = OTHER_INSTR_END.sub("", new).strip()
        add_flag(r, "instruction_only_stem")
    if new != s:
        log(r["item_id"], "R2_leaked_instruction", "stem", s[-60:], new[-60:] or "(empty)")
        r["stem"] = new
    return r, mode


def r3_inline_options(r):
    s = nz(r.get("stem"))
    if not isinstance(s, str) or r["subject"] == "English" or r["options"] not in ("[]", "", None):
        return r
    if r["item_type"] == "MATCH" or re.search(r"true or false", s, re.I) or (r["item_type"] == "TF" and re.match(r"^\s*[A-Fa-f][.)]\s", s)):
        return r
    text = sp(s.replace("•", " ").replace("♦", " "))
    ch = any_chain(text, ("paren", "lower", "upper"))
    if len(ch) < 2:   # tolerate a book typo such as "A. ... A. ... C."
        ch = list(re.finditer(r"(?:(?<=\s)|^)\(?[A-Da-d][.)]\s", text))
        if len(ch) < 3 or not re.search(r"[?:]", text[:ch[0].start() + 1]):
            return r
    head, opts = split_by(text, ch)
    if len(opts) < 2 or any("?" in o for o in opts) or sum(len(o.split()) for o in opts) / len(opts) > 9 or not head:
        return r
    if not (BLANK.search(head) or head.rstrip().endswith(("?", ":")) or ch[0].group().strip().startswith("(")):
        return r
    before = r["stem"]
    r["stem"] = re.sub(r"\.{4,}|…{2,}", "______", head)
    r["options"] = json.dumps(opts, ensure_ascii=False)
    r["option_labels"] = json.dumps(list("abcdef"[:len(opts)]))
    r["n_options"] = len(opts)
    log(r["item_id"], "R3_inline_options", "stem/options/item_type", before[:120], f"{r['item_type']}->MCQ, {len(opts)} options")
    r["item_type"] = "MCQ"
    return r


def r4_split_tf(r):
    """One row per statement for a TF item that bundles several ('A. ... B. ...'); returns a list of rows."""
    s = nz(r.get("stem"))
    if r["item_type"] != "TF" or not isinstance(s, str):
        return [r]
    text = sp(s.replace("•", " "))
    body = TF_INSTR_START.sub("", text)
    ch = any_chain(body, ("upper", "lower", "roman"))
    if len(ch) < 2:
        return [r]
    head, parts = split_by(body, ch)
    if head and not re.search(r"true or false|statements?", head, re.I):
        return [r]
    labels = [re.sub(r"[().\s]", "", m.group()).upper() for m in ch]
    key = nz(r.get("answer_key"))
    kp = tf_key_parts(key, len(parts)) if key is not None else None
    rows = []
    for lab, ptxt, k in zip(labels, parts, kp or [None] * len(parts)):
        p = dict(r)
        p.update(parent_item_id=r["item_id"], item_id=f"{r['item_id']}-{lab}", stem=TF_SUFFIX.sub("", ptxt).strip(),
                 answer_key=k, answer_text=k, options="[]", option_labels="[]", n_options=0)
        if key is not None and kp is None:
            add_flag(p, "key_count_mismatch")
        rows.append(p)
    log(r["item_id"], "R4_split_TF", "item", text[:80], f"{len(rows)} rows, key {'aligned' if kp else 'missing' if key is None else 'mismatch'}")
    return rows


def r5_keys(r):
    key, t = nz(r.get("answer_key")), r["item_type"]
    if not isinstance(key, str) or not key.strip():
        r["answer_key"] = None
        return r
    raw = key
    key = sp(re.sub(r"^[\-–•\s]+", "", JUNK.sub(" ", key)))
    reason = "key_not_an_answer" if NOT_ANSWER.search(key) else None
    if t == "TF":
        m = re.fullmatch(r"(?:\(?[A-Fa-f]\)\s*)?(true|false|t|f)\.?", key, re.I)
        if m:
            key = "True" if m.group(1).lower() in ("true", "t") else "False"
        else:
            reason = reason or "tf_key_invalid"
    else:   # MCQ: the key must be a letter that exists among the options
        opts = json.loads(r["options"] or "[]")
        key = re.sub(r"^(?:answer|ans|key)\s*[:.]\s*", "", key, flags=re.I)
        letter = None
        m = re.fullmatch(r"\(?([a-dA-D])\)?[.)]?(?:\s+(.*))?", key)
        if m and (m.group(2) is None or any(norm_txt(m.group(2)) in norm_txt(o) or norm_txt(o) in norm_txt(m.group(2)) for o in opts)):
            letter = m.group(1).lower()
        elif opts:
            hit = [i for i, o in enumerate(opts) if norm_txt(o) == norm_txt(key)]
            letter = "abcdef"[hit[0]] if hit else None
        if letter and len(opts) >= 2 and "abcdef".index(letter) < len(opts):
            key = letter
            r["answer_text"] = opts["abcdef".index(letter)]
        else:
            reason = reason or "mcq_key_invalid"
    if reason:
        log(r["item_id"], "R5_key_untrusted", "answer_key", raw[:80], f"(missing) {reason}")
        r["answer_key_raw"] = raw
        r["answer_key"], r["answer_text"] = None, None
        add_flag(r, reason)
    else:
        if key != raw:
            log(r["item_id"], "R5_key_normalised", "answer_key", raw[:60], key[:60])
        r["answer_key"] = key
        if t == "TF":
            r["answer_text"] = key
    return r


def run(df):
    """Apply R1-R5 row by row (in file order) and return the MCQ/TF rows as a list of dicts."""
    LOG.clear()
    out, mode, prev_group = [], None, None
    for r in df.to_dict("records"):
        r["answer_key_raw"], r["parent_item_id"], r["clean_flags"] = None, None, None
        r = r1_page_numbers(r)
        r, new_mode = r2_instructions(r)
        group = (r["subject"], r["class_level"], r["unit_no"], r["section"])
        if group != prev_group:
            mode = None
        prev_group = group
        if new_mode:
            mode = new_mode
        elif mode == "TF" and r["item_type"] == "OPEN" and isinstance(r["stem"], str) and "?" not in r["stem"] \
                and len(r["stem"].split()) <= 25 and r["options"] in ("[]", None) and nz(r.get("answer_key")) in (None, "True", "False"):
            log(r["item_id"], "R2_retype", "item_type", "OPEN", "TF")
            r["item_type"] = "TF"
        elif mode == "TF" and r["item_type"] != "TF":
            mode = None
        r = r3_inline_options(r)
        for p in r4_split_tf(r):
            if p["item_type"] not in KEEP_TYPES:
                log(p["item_id"], "R0_out_of_scope", "item_type", p["item_type"], "(left out: only MCQ and TF are kept)")
                continue
            if p["item_type"] == "TF" and isinstance(p.get("stem"), str):   # drop instruction words left in the statement
                new = sp(SECTION_TAIL.sub("", TF_SUFFIX.sub("", SECTION_TAIL.sub("", TF_INSTR_START.sub("", p["stem"])))))
                if new != p["stem"]:
                    log(p["item_id"], "R2_tf_wording", "stem", p["stem"][-40:], new[-40:])
                    p["stem"] = new
            out.append(r5_keys(clean_options(p, "R1_page_number")))
    return out


# ---------------------------------------------------------------- table-level steps
def sms_label(r):
    """'Q6', 'R1', 'E2' ... the label at the start of the SMS; split parts get their letter, e.g. 'Q8a'."""
    parent = r.get("parent_item_id")
    has_parent = isinstance(parent, str) and parent != ""
    parts = (parent if has_parent else str(r["item_id"])).split("-")
    code = parts[3][0] if len(parts) > 3 and parts[3] else "Q"
    label = f"{code}{int(r['q_no'])}"
    return label + str(r["item_id"]).rsplit("-", 1)[-1].lower() if has_parent else label


def finalise(rows):
    d = pd.DataFrame(rows)
    sms, chars, segs, ucs, words = [], [], [], [], []
    for _, r in d.iterrows():
        if not isinstance(r["stem"], str) or not r["stem"]:        # R6: no question text -> missing, not 0
            sms.append(None); chars.append(None); segs.append(None); words.append(None); ucs.append(r["uses_unicode"])
            continue
        txt = render(sms_label(r), r["stem"], json.loads(r["options"] or "[]"), r["item_type"], [])
        c, sg, u = sms_stats(txt)
        sms.append(txt); chars.append(c); segs.append(sg); ucs.append(u); words.append(len(r["stem"].split()))
    d["sms_text"], d["sms_chars"], d["sms_segments"], d["uses_unicode"] = sms, chars, segs, ucs
    d["n_words_question"] = words
    d["n_options"] = d["options"].map(lambda o: len(json.loads(o or "[]")))
    for c in ("sms_chars", "sms_segments", "n_words_question"):
        d[c] = d[c].astype("Int64")

    def title(r):    # R7
        if r["subject"].startswith("Science") and r["class_level"] == 6:
            return P6_TITLES.get(int(r["unit_no"]))
        raw = str(r["unit_title"]).strip()
        if re.fullmatch(r"Unit \d+", raw):
            return None
        t = re.sub(r"[\s.…]+\d*\s*$", "", raw)
        return t.title() if t else None
    d["unit_title"] = d.apply(title, axis=1)
    d["sensitive_topic"] = d.apply(lambda r: bool(SENSITIVE_WORDS.search(f"{r['unit_title']} {r['stem']}")), axis=1)
    dup = d.groupby("item_id").cumcount()
    d["item_id"] = [i if n == 0 else f"{i}-dup{n}" for i, n in zip(d["item_id"], dup)]
    norm = d["stem"].fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    d["group_id"] = [hashlib.md5(f"{c}|{n}".encode()).hexdigest()[:10] if n else f"id-{i}" for c, n, i in zip(d["class_level"], norm, d["item_id"])]
    d["stem_missing"] = d["stem"].isna() | (d["stem"].astype(str).str.len() == 0)
    d["key_missing"] = d["answer_key"].isna()

    def reason(r):   # R8
        if r["stem_missing"]:
            return "question_text_missing"
        if r["item_type"] == "MCQ" and r["n_options"] < 2:
            return "mcq_options_missing"
        if r["key_missing"]:
            f = str(r["clean_flags"] or "")
            return "key_untrusted" if any(x in f for x in ("invalid", "mismatch", "not_an_answer")) else "no_answer_key"
        if r["needs_visual"] or r["needs_computer"]:
            return "needs_picture_or_computer"
        return ""
    d["exclude_reason"] = d.apply(reason, axis=1)
    d["usable_supervised"] = d["exclude_reason"] == ""
    return d


def split_qa(d):
    """Usable rows only: a questions table with no answer in it, and an answers table joined by item_id."""
    k = d[d["usable_supervised"]].copy()
    k["question"] = k["stem"]
    q = k[[c if c != "question" else "question" for c in QUESTION_COLS]].reset_index(drop=True)
    k["answer_source"] = k["answer_key_source"]
    a = k[ANSWER_COLS].reset_index(drop=True)
    return q, a


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "data", "raw", "items.csv")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    raw = pd.read_csv(src)
    d = finalise(run(raw))
    q, a = split_qa(d)
    d.to_csv(os.path.join(out_dir, "items_clean.csv"), index=False, encoding="utf-8")
    q.to_csv(os.path.join(out_dir, "questions.csv"), index=False, encoding="utf-8")
    a.to_csv(os.path.join(out_dir, "answers.csv"), index=False, encoding="utf-8")
    pd.DataFrame(LOG).to_csv(os.path.join(out_dir, "cleaning_log.csv"), index=False, encoding="utf-8")
    print(f"raw rows: {len(raw)}   MCQ/TF rows kept: {len(d)}   log entries: {len(LOG)}")
    print("rule counts:", pd.DataFrame(LOG).groupby("rule").size().to_dict())
    print("usable for modelling:", len(q), q["item_type"].value_counts().to_dict(), " by class:", q["class_level"].value_counts().to_dict())
    print("not usable:", d.exclude_reason[d.exclude_reason != ""].value_counts().to_dict())


if __name__ == "__main__":
    main()