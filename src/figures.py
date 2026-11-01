"""The eight Phase 2 figures. Run: python -m src.figures   (writes figures/*.png at 300 DPI).
Everything fitted (limits, scaler, PCA) uses the training split only; the missingness figure uses the cleaned item table."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, skew
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from .pipeline_builder import FEATURES, NUMERIC, BINARY, TARGET, build_preprocessor

BLUE, ORANGE, AQUA, GREY, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8f98", "#222831", "#e3e5e8"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GREY, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True,
                     "figure.dpi": 100, "savefig.dpi": 300})
OUT = "figures"


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", facecolor="white"); plt.close(fig)


def load():
    c = pd.read_csv("data/processed/candidates.csv")
    tr = c[c.split == "train"].reset_index(drop=True)
    return pd.read_csv("data/processed/items_clean.csv"), c, tr


def fig1_1(items):
    cols = ["stem", "unit_title", "sms_text", "answer_key", "answer_text", "review_flag", "clean_flags", "parent_item_id"]
    d = items.sort_values(["class_level", "item_type", "item_id"]).reset_index(drop=True)
    M = d[cols].isna().to_numpy().astype(int)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(M, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#f1f3f5", BLUE]), interpolation="nearest")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{c}\n{M[:, i].mean():.0%}" for i, c in enumerate(cols)], rotation=45, ha="right")
    ax.set_ylabel("Item (sorted by class, type)"); ax.grid(False)
    ax.set_title("Fig 1.1  Missing values in the cleaned MCQ/TF table (n = %d items)" % len(d), loc="left", fontsize=10)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color="#f1f3f5", ec=GREY)],
              labels=["missing", "present"], loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    save(fig, "fig1_1_missingness_heatmap")


def fig1_2(c, items):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8))
    t = c.groupby(["item_type", "is_correct"]).size().unstack(fill_value=0)
    x = np.arange(len(t))
    a.bar(x - .18, t[0], .34, color=GREY, label="incorrect candidate"); a.bar(x + .18, t[1], .34, color=BLUE, label="correct candidate")
    for i, (n0, n1) in enumerate(zip(t[0], t[1])):
        a.text(i - .18, n0 + 3, n0, ha="center"); a.text(i + .18, n1 + 3, n1, ha="center")
    a.set_xticks(x); a.set_xticklabels(t.index); a.set_ylabel("Candidate rows"); a.legend(frameon=False)
    a.set_title("Target: is_correct (%.1f%% positive)" % (100 * c.is_correct.mean()), loc="left", fontsize=10)
    m = items[(items.item_type == "MCQ") & items.usable_supervised.astype(bool)].answer_key.value_counts().reindex(list("abcd"), fill_value=0)
    b.bar(m.index, m.values, .55, color=ORANGE)
    for i, v in enumerate(m.values): b.text(i, v + 1, f"{v} ({v / m.sum():.0%})", ha="center")
    b.set_xlabel("Position of the correct option"); b.set_ylabel("MCQ items")
    b.set_title("MCQ answer-key position (shortcut risk)", loc="left", fontsize=10)
    fig.suptitle("Fig 1.2  Class balance of the target", x=.01, ha="left", fontsize=10, y=1.02)
    save(fig, "fig1_2_target_distribution")


def fig2_1(tr):
    pre = build_preprocessor().fit(tr[FEATURES])
    w = pre.named_transformers_["num"].named_steps["winsor"]
    imp = pre.named_transformers_["num"].named_steps["impute"]
    raw = tr[NUMERIC]; cap = w.transform(pd.DataFrame(imp.transform(raw), columns=NUMERIC))
    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    for ax, col in zip(axs.ravel(), NUMERIC):
        bp = ax.boxplot([raw[col].dropna(), cap[col]], widths=.5, patch_artist=True, flierprops=dict(marker="o", ms=3, mfc="none", mec=ORANGE),
                        medianprops=dict(color=INK), tick_labels=["raw", "capped"])
        for p, k in zip(bp["boxes"], (GREY, BLUE)): p.set_facecolor(k); p.set_alpha(.6)
        n_out = int(((raw[col] < w.lower_[col]) | (raw[col] > w.upper_[col])).sum())
        ax.set_title(f"{col}\n{n_out} capped ({w.rule_[col]})", fontsize=8, loc="left")
    fig.suptitle("Fig 2.1  Outliers before and after Winsorizing (fitted on train)", x=.01, ha="left", fontsize=10)
    fig.tight_layout(); save(fig, "fig2_1_outlier_boxplots")


def fig3_1(tr):
    cols = NUMERIC + BINARY
    R = tr[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=60, ha="right"); ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if abs(R.iloc[i, j]) >= .5 and i != j: ax.text(j, i, f"{R.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(R.iloc[i, j]) > .75 else INK)
    fig.colorbar(im, ax=ax, shrink=.7, label="Spearman r")
    ax.set_title("Fig 3.1  Spearman correlation, numeric + binary features (train)", loc="left", fontsize=10)
    save(fig, "fig3_1_correlation_matrix")
    return R


def fig3_2(tr, R):
    cols = NUMERIC
    pairs = sorted(((abs(R.loc[a, b]), a, b) for i, a in enumerate(cols) for b in cols[i + 1:]), reverse=True)[:4]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.2))
    rng = np.random.default_rng(0)
    for ax, (r, a, b) in zip(axs, pairs):
        j = lambda s: s + rng.normal(0, .02 * (s.max() - s.min() + 1e-9), len(s))
        ax.scatter(j(tr[a]), j(tr[b]), s=10, color=BLUE, alpha=.5, lw=0)
        ax.set_xlabel(a); ax.set_ylabel(b); ax.set_title(f"|r| = {r:.2f}" + ("  (>= 0.85)" if r >= .85 else "  (< 0.85, kept)"), fontsize=9, loc="left")
    fig.suptitle("Fig 3.2  Most correlated feature pairs (train, light jitter added)", x=.01, ha="left", fontsize=10, y=1.04)
    fig.tight_layout(); save(fig, "fig3_2_collinear_scatterplots")
    return pairs


def fig3_3(tr):
    X = tr[FEATURES].copy(); y = tr[TARGET]
    X["item_type"] = (X.item_type == "TF").astype(int); X["section"] = X.section.astype("category").cat.codes
    X = X.fillna(X.median())
    mi = pd.Series(mutual_info_classif(X, y, random_state=0, discrete_features=[c in BINARY + ["item_type", "section", "class_level"] for c in X.columns]), index=X.columns)
    rho = X.apply(lambda s: spearmanr(s, y)[0] if s.nunique() > 1 else 0)
    order = mi.sort_values().index
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True)
    a.barh(order, mi[order], color=BLUE, height=.6); a.set_xlabel("Mutual information with is_correct (nats)")
    b.barh(order, rho[order], color=[ORANGE if v < 0 else AQUA for v in rho[order]], height=.6); b.axvline(0, color=GREY, lw=.8)
    b.set_xlabel("Spearman r with is_correct (green +, orange -)")
    fig.suptitle("Fig 3.3  Which features relate to the target (train)", x=.01, ha="left", fontsize=10)
    fig.tight_layout(); save(fig, "fig3_3_target_importance")
    return mi, rho


def fig4_1(tr):
    pre = build_preprocessor().fit(tr[FEATURES])
    Z = pre.transform(tr[FEATURES])
    sk = tr[NUMERIC].apply(lambda s: skew(s.dropna())).abs().sort_values(ascending=False)
    cols = list(sk.index[:3])
    fig, axs = plt.subplots(3, 3, figsize=(9, 7))
    for r, col in enumerate(cols):
        s = tr[col].dropna(); wz = pre.named_transformers_["num"].named_steps["winsor"]
        capped = s.clip(wz.lower_[col], wz.upper_[col])
        for k, (v, lab, colr) in enumerate([(s, f"raw (skew {skew(s):.2f})", GREY), (np.log1p(capped.clip(lower=0)), f"capped + log1p (skew {skew(np.log1p(capped.clip(lower=0))):.2f})", BLUE), (Z[col], f"+ standardised (mean 0, sd 1)", AQUA)]):
            ax = axs[r, k]; ax.hist(v, bins=20, color=colr, alpha=.85); ax.set_title(f"{col}: {lab}", fontsize=8, loc="left")
    fig.suptitle("Fig 4.1  Transformation grid for the three most skewed numeric features (train)", x=.01, ha="left", fontsize=10)
    fig.tight_layout(); save(fig, "fig4_1_transformation_grid")


def fig5_1(tr):
    Z = build_preprocessor().fit_transform(tr[FEATURES])
    p = PCA().fit(Z); ev = p.explained_variance_ratio_; cum = ev.cumsum(); k = int(np.argmax(cum >= .95)) + 1
    fig, ax = plt.subplots(figsize=(7, 4)); x = np.arange(1, len(ev) + 1)
    ax.bar(x, ev, .55, color=BLUE, label="per component"); ax.plot(x, cum, color=ORANGE, marker="o", ms=4, lw=1.5, label="cumulative")
    ax.axhline(.95, color=GREY, ls="--", lw=1); ax.axvline(k, color=GREY, ls=":", lw=1)
    ax.text(len(ev), .955, "95%", ha="right", va="bottom", fontsize=8); ax.text(k + .2, .1, f"{k} components reach 95%", fontsize=8)
    ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance ratio"); ax.set_xticks(x); ax.legend(frameon=False, loc="center right")
    ax.set_title(f"Fig 5.1  PCA scree plot on the {Z.shape[1]} preprocessed features (train)", loc="left", fontsize=10)
    save(fig, "fig5_1_pca_scree_plot")
    return k, ev


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    items, c, tr = load()
    fig1_1(items); fig1_2(c, items); fig2_1(tr); R = fig3_1(tr); fig3_2(tr, R); fig3_3(tr); fig4_1(tr); fig5_1(tr)
    print("wrote 8 figures to", OUT)


if __name__ == "__main__":
    main()
