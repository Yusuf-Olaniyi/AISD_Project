"""Every number quoted in the Phase 2 reflection report. Run: python -m src.report_stats  -> reports/phase2_stats.json"""
import json, os, numpy as np, pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from .pipeline_builder import FEATURES, NUMERIC, BINARY, NOMINAL, ORDINAL, TARGET, build_preprocessor
from .transformers import vif


def cramers_v(a, b):
    t = pd.crosstab(a, b); chi2, p, dof, _ = stats.chi2_contingency(t, correction=False)
    n = t.to_numpy().sum(); return float(np.sqrt(chi2 / (n * (min(t.shape) - 1)))), float(chi2), float(p), int(dof)


def little_mcar(df):
    """Little (1988) chi-square test, EM estimates of mean and covariance."""
    X = df.to_numpy(float); n, k = X.shape; M = ~np.isnan(X)
    mu = np.nanmean(X, 0); S = np.diag(np.nanvar(X, 0) + 1e-9)
    for _ in range(300):
        T1 = np.zeros(k); T2 = np.zeros((k, k))
        for i in range(n):
            o, m = M[i], ~M[i]; x = X[i].copy(); C = np.zeros((k, k))
            if m.any():
                B = S[np.ix_(m, o)] @ np.linalg.inv(S[np.ix_(o, o)])
                x[m] = mu[m] + B @ (X[i, o] - mu[o]); C[np.ix_(m, m)] = S[np.ix_(m, m)] - B @ S[np.ix_(o, m)]
            T1 += x; T2 += np.outer(x, x) + C
        new_mu = T1 / n; S = T2 / n - np.outer(new_mu, new_mu)
        if np.abs(new_mu - mu).max() < 1e-8: mu = new_mu; break
        mu = new_mu
    d2 = 0; df_ = 0
    for pat in {tuple(r) for r in M}:
        idx = np.all(M == pat, axis=1); o = np.array(pat)
        if o.all() or not o.any(): 
            if not o.any(): continue
        ybar = X[idx][:, o].mean(0); d = ybar - mu[o]
        d2 += idx.sum() * d @ np.linalg.inv(S[np.ix_(o, o)]) @ d; df_ += o.sum()
    df_ -= k
    return float(d2), int(df_), float(stats.chi2.sf(d2, df_)), len({tuple(r) for r in M})


def main():
    out = {}
    raw = pd.read_csv("data/raw/items.csv"); items = pd.read_csv("data/processed/items_clean.csv")
    c = pd.read_csv("data/processed/candidates.csv"); tr, va, te = (c[c.split == s] for s in ("train", "val", "test"))
    qa = pd.read_csv("data/processed/qa_merged.csv")
    out["shape"] = dict(raw=list(raw.shape), clean=list(items.shape), usable=int(items.usable_supervised.sum()), qa=list(qa.shape),
                        cand=list(c.shape), train=list(tr.shape), val=list(va.shape), test=list(te.shape),
                        q_train=int(tr.group_id.nunique()), q_val=int(va.group_id.nunique()), q_test=int(te.group_id.nunique()),
                        pos_train=float(tr.is_correct.mean()), pos_val=float(va.is_correct.mean()), pos_test=float(te.is_correct.mean()))
    out["exclude"] = items.exclude_reason.fillna("usable").value_counts().to_dict()
    out["raw_types"] = raw.item_type.value_counts().to_dict(); out["clean_types"] = items.item_type.value_counts().to_dict()
    # missingness
    mc = ["stem", "unit_title", "sms_text", "answer_key", "answer_text", "sms_chars"]
    out["miss_pct"] = {k: round(float(items[k].isna().mean() * 100), 2) for k in mc}
    out["miss_pct_raw"] = {k: round(float(raw[k].isna().mean() * 100), 2) for k in ["answer_key", "answer_text", "stem", "sms_text"]}
    out["miss_cells_clean_pct"] = round(float(items[["stem", "answer_key", "answer_text", "sms_text", "sms_chars", "unit_title"]].isna().to_numpy().mean() * 100), 2)
    km = items.answer_key.isna().astype(int); sm = items.stem.isna().astype(int)
    out["cramers_key"] = {col: dict(zip(["V", "chi2", "p", "dof"], cramers_v(items[col], km))) for col in ["class_level", "item_type", "section"]}
    out["cramers_stem"] = {col: dict(zip(["V", "chi2", "p", "dof"], cramers_v(items[col], sm))) for col in ["class_level", "item_type"]}
    out["key_missing_by_class"] = (items.groupby("class_level").answer_key.apply(lambda s: round(s.isna().mean() * 100, 1))).to_dict()
    out["key_missing_by_type"] = (items.groupby("item_type").answer_key.apply(lambda s: round(s.isna().mean() * 100, 1))).to_dict()
    d2, df_, p, npat = little_mcar(items[["n_words_question", "sms_chars", "sms_segments"]])
    out["little"] = dict(chi2=d2, df=df_, p=p, patterns=npat, n=len(items))
    out["model_missing_cells"] = int(c[FEATURES].isna().sum().sum())
    # key position
    mcq = items[(items.item_type == "MCQ") & items.usable_supervised.astype(bool)]
    out["mcq_key_pos"] = mcq.answer_key.value_counts().to_dict()
    tf = items[(items.item_type == "TF") & items.usable_supervised.astype(bool)]
    out["tf_key"] = tf.answer_key.value_counts().to_dict()
    # outliers
    pre = build_preprocessor().fit(tr[FEATURES]); num = pre.named_transformers_["num"]
    w, imp, lgs = num.named_steps["winsor"], num.named_steps["impute"], num.named_steps["log"]
    rows = {}
    capped = w.transform(pd.DataFrame(imp.transform(tr[NUMERIC]), columns=NUMERIC))
    for col in NUMERIC:
        s = tr[col]; q1, q3 = s.quantile(.25), s.quantile(.75); lo, hi = w.lower_[col], w.upper_[col]
        z = (s - s.mean()) / s.std(ddof=0); lg = np.log1p(capped[col].clip(lower=0)) if col in lgs.apply_ else capped[col]
        rows[col] = dict(skew=float(stats.skew(s)), kurt=float(stats.kurtosis(s)), q1=float(q1), q3=float(q3), lo=float(lo), hi=float(hi), rule=w.rule_[col],
                         n_cap=int(((s < lo) | (s > hi)).sum()), pct_cap=float(((s < lo) | (s > hi)).mean() * 100), n_z3=int((z.abs() > 3).sum()),
                         skew_capped=float(stats.skew(capped[col])), log_applied=col in lgs.apply_, skew_log=float(stats.skew(lg)), kurt_log=float(stats.kurtosis(lg)))
    out["outliers"] = rows
    # collinearity
    sp = tr[NUMERIC].corr(method="spearman"); pe = tr[NUMERIC].corr(method="pearson")
    pairs = sorted(((abs(sp.loc[a, b]), a, b) for i, a in enumerate(NUMERIC) for b in NUMERIC[i + 1:]), reverse=True)[:6]
    out["top_pairs"] = [dict(a=a, b=b, spearman=float(sp.loc[a, b]), pearson=float(pe.loc[a, b])) for _, a, b in pairs]
    out["vif_raw"] = vif(tr[NUMERIC].fillna(tr[NUMERIC].median())).round(2).to_dict()
    Zlog = lgs.transform(capped)
    out["vif_log"] = vif(Zlog).round(2).to_dict()
    out["dropped"] = num.named_steps["collinear"].dropped_
    # MI + rank corr
    X = tr[FEATURES].copy(); X["item_type"] = (X.item_type == "TF").astype(int); X["section"] = X.section.astype("category").cat.codes; X = X.fillna(X.median())
    mi = mutual_info_classif(X, tr[TARGET], random_state=0, discrete_features=[k in BINARY + ["item_type", "section", "class_level"] for k in X.columns])
    out["mi"] = dict(zip(X.columns, map(float, mi)))
    out["rho_y"] = {k: float(stats.spearmanr(X[k], tr[TARGET])[0]) if X[k].nunique() > 1 else 0.0 for k in X.columns}
    # scaled stats
    Ztr, Zte = pre.transform(tr[FEATURES]), pre.transform(te[FEATURES]); Zva = pre.transform(va[FEATURES])
    keep = list(num.named_steps["collinear"].keep_)
    out["vif_after_drop"] = vif(Zlog[keep]).round(2).to_dict()
    out["scaled_train"] = dict(mean_abs_max=float(Ztr[keep].mean().abs().max()), std_min=float(Ztr[keep].std(ddof=0).min()), std_max=float(Ztr[keep].std(ddof=0).max()))
    out["scaled_test"] = dict(mean_abs_max=float(Zte[keep].mean().abs().max()), std_min=float(Zte[keep].std(ddof=0).min()), std_max=float(Zte[keep].std(ddof=0).max()))
    out["final_shapes"] = dict(train=list(Ztr.shape), val=list(Zva.shape), test=list(Zte.shape), nan=int(Ztr.isna().sum().sum() + Zva.isna().sum().sum() + Zte.isna().sum().sum()))
    out["n_input_features"] = len(FEATURES); out["feature_cols"] = list(Ztr.columns)
    # PCA
    p = PCA().fit(Ztr); cum = p.explained_variance_ratio_.cumsum()
    out["pca"] = dict(eig=p.explained_variance_.round(3).tolist(), ratio=p.explained_variance_ratio_.round(4).tolist(), k90=int(np.argmax(cum >= .9) + 1), k95=int(np.argmax(cum >= .95) + 1), pc1=float(p.explained_variance_ratio_[0]))
    # constants in scope
    out["constant_in_scope"] = [k for k in ["n_blanks", "n_subitems", "uses_unicode", "needs_visual", "needs_computer", "sensitive_topic", "subject"] if k in items and items[k].nunique() <= 1]
    out["rank_check_full_table_nwords_sms"] = float(raw[["n_words_stem", "sms_chars"]].corr(method="spearman").iloc[0, 1])
    os.makedirs("reports", exist_ok=True); json.dump(out, open("reports/phase2_stats.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
