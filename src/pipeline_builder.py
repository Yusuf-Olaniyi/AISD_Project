"""ColumnTransformer / Pipeline for the candidate table (data/processed/candidates.csv).

    X_train -> fit -> transform(X_train, X_val, X_test)

Nothing is learned outside the training split: imputer values, winsorizing limits, the collinearity decision,
scaler means and category lists are all fitted inside this one Pipeline.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .transformers import CollinearityDropper, Log1p, Winsorizer

NUMERIC = ["n_words_question", "sms_chars", "sms_segments", "n_options", "cand_position",
           "cand_n_words", "cand_len_ratio", "cand_overlap"]
NOMINAL = ["item_type", "section"]
ORDINAL = ["class_level"]                     # 4 < 5 < 6
BINARY = ["q_has_negation", "q_has_absolute", "cand_is_longest", "cand_is_true"]
FEATURES = NUMERIC + NOMINAL + ORDINAL + BINARY
TARGET = "is_correct"


def build_preprocessor(r_max=0.85, vif_max=10.0):
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("winsor", Winsorizer()),
                    ("log", Log1p()),
                    ("collinear", CollinearityDropper(r_max=r_max, vif_max=vif_max)),
                    ("scale", StandardScaler())])
    nom = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    ordi = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                     ("ordinal", OrdinalEncoder(categories=[[4, 5, 6]], handle_unknown="use_encoded_value", unknown_value=-1))])
    pre = ColumnTransformer([("num", num, NUMERIC), ("nom", nom, NOMINAL), ("ord", ordi, ORDINAL),
                             ("bin", "passthrough", BINARY)], verbose_feature_names_out=False)
    pre.set_output(transform="pandas")
    return pre
