#!/usr/bin/env python3
"""
merge_qa.py -- join data/processed/questions.csv and answers.csv on item_id and prove the match is right.

    python src/merge_qa.py                      # -> data/processed/qa_merged.csv

The join is one-to-one: every question has exactly one answer, no answer is left over, and the checks below
must all pass or the script stops. A wrong match would silently label every reply wrongly, so it is tested
rather than assumed.
"""
import json
import os
import re
import sys

import pandas as pd


def merge_qa(questions, answers):
    q, a = questions.copy(), answers.copy()
    problems = []
    if not q["item_id"].is_unique:
        problems.append("duplicate item_id in questions")
    if not a["item_id"].is_unique:
        problems.append("duplicate item_id in answers")
    only_q, only_a = set(q["item_id"]) - set(a["item_id"]), set(a["item_id"]) - set(q["item_id"])
    if only_q:
        problems.append(f"{len(only_q)} questions without an answer, e.g. {sorted(only_q)[:3]}")
    if only_a:
        problems.append(f"{len(only_a)} answers without a question, e.g. {sorted(only_a)[:3]}")
    if problems:
        raise ValueError("; ".join(problems))
    m = q.merge(a, on="item_id", how="inner", validate="one_to_one")

    # consistency checks between the two tables (a join on id alone cannot catch a wrongly numbered answer)
    bad = []
    for _, r in m.iterrows():
        mt = re.match(r"^[A-Z]+-P(\d)-U(\d+)-[A-Z](\d+)", r["item_id"])
        if mt and (int(mt.group(1)) != r["class_level"] or int(mt.group(2)) != r["unit_no"]):
            bad.append((r["item_id"], "id does not match class/unit columns"))
        src = re.search(r"P(\d)\b", str(r["answer_source"]))
        if src and int(src.group(1)) != r["class_level"]:
            bad.append((r["item_id"], f"answer comes from a P{src.group(1)} guide but the question is P{r['class_level']}"))
        if r["item_type"] == "TF" and r["answer_key"] not in ("True", "False"):
            bad.append((r["item_id"], "TF key is not True/False"))
        if r["item_type"] == "MCQ":
            opts = json.loads(r["options"])
            k = r["answer_key"]
            if not (isinstance(k, str) and len(k) == 1 and "abcdef".find(k) in range(len(opts))):
                bad.append((r["item_id"], "MCQ key is not a letter among the options"))
            elif opts["abcdef".index(k)] != r["answer_text"]:
                bad.append((r["item_id"], "MCQ answer_text is not the option the key points to"))
    if bad:
        raise ValueError(f"{len(bad)} mismatches, e.g. {bad[:3]}")
    assert len(m) == len(q) == len(a)
    return m


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(root, "data", "processed")
    q, a = pd.read_csv(os.path.join(d, "questions.csv")), pd.read_csv(os.path.join(d, "answers.csv"))
    m = merge_qa(q, a)
    m.to_csv(os.path.join(d, "qa_merged.csv"), index=False, encoding="utf-8")
    print(f"questions {len(q)}  answers {len(a)}  merged {len(m)}  (one-to-one, all consistency checks passed)")
    print(m.groupby(["class_level", "item_type"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
