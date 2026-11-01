"""Custom scikit-learn transformers for the SMS tutor candidate table.

Every transformer follows the fit/transform contract, so it can sit inside a Pipeline and be fitted on X_train only.
All of them return DataFrames with column names so the figures and the decision log can name each feature.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Winsorizer(BaseEstimator, TransformerMixin):
    """Cap each numeric column at limits learned on the training split.

    Rule per column (Phase 1 plan): the 1.5 x IQR fences when the IQR is above zero; when the IQR is zero
    (a count that is almost always the same value) the 99th percentile as the upper limit.
    """

    def __init__(self, k=1.5, upper_q=0.99):
        self.k, self.upper_q = k, upper_q

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1, q3 = X.quantile(0.25), X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = np.where(iqr > 0, q1 - self.k * iqr, X.min())
        self.upper_ = np.where(iqr > 0, q3 + self.k * iqr, X.quantile(self.upper_q))
        self.lower_, self.upper_ = pd.Series(self.lower_, index=X.columns), pd.Series(self.upper_, index=X.columns)
        self.rule_ = pd.Series(np.where(iqr > 0, "IQR fence", "99th percentile"), index=X.columns)
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        return X.clip(self.lower_, self.upper_, axis=1)

    def get_feature_names_out(self, names=None):
        return np.array(self.columns_)


class Log1p(BaseEstimator, TransformerMixin):
    """log(1 + x): shrinks the long right tail of the count features (all have skewness above 1.8)."""

    def fit(self, X, y=None):
        self.columns_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        return np.log1p(pd.DataFrame(X, columns=self.columns_).clip(lower=0))

    def get_feature_names_out(self, names=None):
        return np.array(self.columns_)


def vif(X):
    """Variance inflation factor of each column (1 / (1 - R^2) from regressing it on the others)."""
    Z = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    out = {}
    for c in Z.columns:
        others = Z.drop(columns=c)
        if others.shape[1] == 0:
            out[c] = 1.0
            continue
        A = np.c_[np.ones(len(Z)), others.to_numpy()]
        b = np.linalg.lstsq(A, Z[c].to_numpy(), rcond=None)[0]
        resid = Z[c].to_numpy() - A @ b
        r2 = 1 - (resid ** 2).sum() / max(((Z[c] - Z[c].mean()) ** 2).sum(), 1e-12)
        out[c] = np.inf if r2 >= 0.999999 else 1 / (1 - r2)
    return pd.Series(out)


class CollinearityDropper(BaseEstimator, TransformerMixin):
    """Phase 1 protocol, learned on the training split only.

    Step 1: for every pair with Spearman |r| >= r_max, drop the feature with the higher mean |r| to the others
            (a tie drops the later column).
    Step 2: while any VIF > vif_max, drop the feature with the highest VIF.
    The dropped features and the reason are kept in dropped_ for the decision log.
    """

    def __init__(self, r_max=0.85, vif_max=10.0):
        self.r_max, self.vif_max = r_max, vif_max

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.dropped_ = {}
        keep = list(X.columns)
        corr = X.corr(method="spearman").abs().fillna(0)
        while True:
            sub = corr.loc[keep, keep].to_numpy().copy()
            np.fill_diagonal(sub, 0)
            if sub.max() < self.r_max:
                break
            i, j = np.unravel_index(sub.argmax(), sub.shape)
            mean_abs = sub.sum(axis=1) / max(len(keep) - 1, 1)
            drop = keep[i] if mean_abs[i] > mean_abs[j] else keep[j]
            self.dropped_[drop] = f"|r| = {sub[i, j]:.2f} with {keep[j] if drop == keep[i] else keep[i]}"
            keep.remove(drop)
        while len(keep) > 1:
            v = vif(X[keep])
            if v.max() <= self.vif_max:
                break
            drop = v.idxmax()
            self.dropped_[drop] = f"VIF = {v.max():.1f}"
            keep.remove(drop)
        self.keep_ = keep
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns_)[self.keep_]

    def get_feature_names_out(self, names=None):
        return np.array(self.keep_)
