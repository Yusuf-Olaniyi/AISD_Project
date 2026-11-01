#!/usr/bin/env python3
"""
candidates.py -- build the modelling table: one row per (question, candidate answer), label is_correct.

    python src/candidates.py                    # data/processed/qa_merged.csv -> data/processed/candidates.csv

For a multiple-choice question every option is a candidate answer; for a true/false question the candidates are
True and False. The label comes from the answer key: is_correct = 1 for the key, 0 for the others. Nothing here is
generated: language-model replies in free text (shorthand, typos) can be added later as extra candidate rows.

Features use only what the SMS tutor knows when the reply arrives: the question, the options and the candidate.
The answer key is used for the label only.

The split column assigns whole questions to train / val / test (70 / 15 / 15), stratified by item type, so no
question has candidates in two partitions.
"""
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NEG = re.compile(r"\b(not|no|never|cannot|can't|don't|doesn't|isn't|aren't|nothing|none|without)\b", re.I)
ABS = re.compile(r"\b(always|never|only|all|every|none|must|any)\b", re.I)
TOK = re.compile(r"[a-z0-9]+")


def tokens(t):
    return set(TOK.findall(str(t).lower()))


def build_candidates(qa):
    rows = []
    for _, r in qa.iterrows():
        q = r["question"]
        if r["item_type"] == "MCQ":
            opts = json.loads(r["options"])
            lens = np.array([len(o) for o in opts], dtype=float)
            qt = tokens(q)
            for i, o in enumerate(opts):
                ot = tokens(o)
                rows.append(dict(
                    item_id=r["item_id"], cand_label="abcdef"[i], cand_text=o, cand_position=i,
                    cand_n_words=len(o.split()), cand_len_ratio=len(o) / lens.mean(),
                    cand_overlap=len(qt & ot) / max(len(qt | ot), 1), cand_is_longest=int(len(o) == lens.max()),
                    cand_is_true=0, is_correct=int("abcdef"[i] == r["answer_key"])))
        else:
            for i, v in enumerate(("True", "False")):
                rows.append(dict(
                    item_id=r["item_id"], cand_label=v, cand_text=v, cand_position=i, cand_n_words=1,
                    cand_len_ratio=1.0, cand_overlap=0.0, cand_is_longest=0, cand_is_true=int(v == "True"),
                    is_correct=int(v == r["answer_key"])))
    c = pd.DataFrame(rows)
    item_cols = ["item_id", "group_id", "item_type", "class_level", "section", "n_options", "n_words_question",
                 "sms_chars", "sms_segments", "question"]
    c = c.merge(qa[item_cols], on="item_id", validate="many_to_one")
    c["q_has_negation"] = c["question"].map(lambda t: int(bool(NEG.search(t))))
    c["q_has_absolute"] = c["question"].map(lambda t: int(bool(ABS.search(t))))
    c.insert(0, "pair_id", c["item_id"] + "|" + c["cand_label"])
    return c.drop(columns=["question"])


def assign_split(c, seed=42, fractions=(0.70, 0.15, 0.15)):
    """Split by question group (all candidates of a question, and repeated questions, stay together)."""
    g = c.groupby("group_id").agg(item_type=("item_type", "first")).reset_index()
    tr, rest = train_test_split(g, train_size=fractions[0], stratify=g["item_type"], random_state=seed)
    va, te = train_test_split(rest, train_size=fractions[1] / (fractions[1] + fractions[2]), stratify=rest["item_type"], random_state=seed)
    where = {**{k: "train" for k in tr.group_id}, **{k: "val" for k in va.group_id}, **{k: "test" for k in te.group_id}}
    c = c.copy()
    c["split"] = c["group_id"].map(where)
    assert c["split"].notna().all()
    assert c.groupby("group_id")["split"].nunique().max() == 1      # no question in two partitions
    return c


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(root, "data", "processed")
    qa = pd.read_csv(os.path.join(d, "qa_merged.csv"))
    c = assign_split(build_candidates(qa))
    c.to_csv(os.path.join(d, "candidates.csv"), index=False, encoding="utf-8")
    print(f"{len(qa)} questions -> {len(c)} candidate rows, {c.is_correct.mean():.1%} correct")
    print(c.groupby("split").agg(rows=("pair_id", "size"), questions=("item_id", "nunique"), correct=("is_correct", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
