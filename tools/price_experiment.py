"""Price experiment (scratch / reproducible check of final.ipynb section 7).

Same 793 test listings, same build_groups, training twins of test groups removed, StratifiedGroupKFold(5)
stratified on type, seed 42. Prints the CV and test table for every model.

Run: .venv/bin/python tools/price_experiment.py [--lightgbm-name]
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from furniture.features import add_features, build_groups, material_columns  # noqa: E402
from furniture.price import (SIZE_NUM, CATEGORICAL, e5_embed, price_metrics, spec_prep, specs_embedding_prep,  # noqa: E402
                             specs_text_prep, text_column)

from lightgbm import LGBMRegressor  # noqa: E402

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
LGB_GRID = {"model__learning_rate": [0.03, 0.06], "model__num_leaves": [15, 31], "model__min_child_samples": [10, 30],
            "model__colsample_bytree": [0.3, 0.7]}


def load():
    df = add_features(pd.read_csv("data/processed/furniture.csv"))
    mats = material_columns(df)
    is_test = df["url"].isin(set(pd.read_csv("data/processed/test_urls.csv")["url"]))
    df["group"] = build_groups(df)
    twin = (~is_test) & df["group"].isin(set(df.loc[is_test, "group"]))
    df["text"] = text_column(df, masked=False)
    df["text_masked"] = text_column(df, masked=True)
    return df, mats, is_test, twin


def main():
    df, mats, is_test, twin = load()
    train, test = df[~is_test & ~twin].copy(), df[is_test].copy()
    y_tr, y_te = np.log(train["price_kzt"].values), np.log(test["price_kzt"].values)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(train, train["furniture_type"], train["group"]))

    t0 = time.time()
    for col in ["text", "text_masked"]:
        emb = e5_embed(df[col].tolist(), cache_name=f"name_e5_{col}")
        cols = [f"{col}_e{i}" for i in range(emb.shape[1])]
        e = pd.DataFrame(emb, columns=cols, index=df.index)
        train[cols], test[cols] = e.loc[train.index], e.loc[test.index]
    print(f"embeddings {time.time() - t0:.0f}s")
    E = lambda col: [c for c in train.columns if c.startswith(col + "_e")]

    specs = SIZE_NUM + mats + CATEGORICAL
    alphas = {"model__alpha": [0.3, 1, 3, 10, 30]}
    models = {
        "ridge, specs": (Pipeline([("prep", spec_prep(mats)), ("model", Ridge())]), alphas),
        "random forest, specs": (Pipeline([("prep", spec_prep(mats)), ("model", RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))]),
            {"model__min_samples_leaf": [1, 3, 10], "model__max_features": [0.3, 0.6]}),
        "boosting, specs": (Pipeline([("prep", spec_prep(mats)), ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE))]),
                            {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31],
                             "model__min_samples_leaf": [10, 30]}),
        "LightGBM, specs": (Pipeline([("prep", spec_prep(mats)), ("model", LGBMRegressor(
            n_estimators=600, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1))]), LGB_GRID),
        "ridge, specs + TF-IDF name": (Pipeline([("prep", specs_text_prep(mats, "text")), ("model", Ridge())]), alphas),
        "ridge, specs + TF-IDF masked name": (Pipeline([("prep", specs_text_prep(mats, "text_masked")), ("model", Ridge())]), alphas),
        "ridge, specs + e5 name": (Pipeline([("prep", specs_embedding_prep(mats, E("text"))), ("model", Ridge())]),
                                   {"model__alpha": [3, 10, 30, 100, 300]}),
        "ridge, specs + e5 masked name": (Pipeline([("prep", specs_embedding_prep(mats, E("text_masked"))), ("model", Ridge())]),
                                          {"model__alpha": [3, 10, 30, 100, 300]}),
    }
    if "--lightgbm-name" in sys.argv:   # about 2 hours on CPU (thousands of sparse TF-IDF columns); CV MAE 62,167, test 56,391
        models["LightGBM, specs + TF-IDF name"] = (Pipeline([("prep", specs_text_prep(mats, "text")), ("model", LGBMRegressor(
            n_estimators=600, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1))]), LGB_GRID)
    rows = []
    med_type = train.groupby("furniture_type")["price_kzt"].median()
    oof = np.log(train["furniture_type"].map(
        pd.concat([train.iloc[va].assign(m=train.iloc[tr].groupby("furniture_type")["price_kzt"].median()
                                         .reindex(train.iloc[va]["furniture_type"]).values)["m"] for tr, va in splits])))
    rows.append({"model": "median of the type", **{f"CV {k}": v for k, v in price_metrics(y_tr, np.log(
        pd.concat([pd.Series(train.iloc[tr].groupby("furniture_type")["price_kzt"].median().reindex(train.iloc[va]["furniture_type"]).values,
                             index=train.index[va]) for tr, va in splits]).reindex(train.index).values)).items()},
                 **{f"test {k}": v for k, v in price_metrics(y_te, np.log(test["furniture_type"].map(med_type).values)).items()}})
    del oof
    for name, (pipe, grid) in models.items():
        t0 = time.time()
        s = GridSearchCV(pipe, grid, cv=splits, scoring="neg_mean_absolute_error", n_jobs=1 if "forest" in name else -1)
        s.fit(train, y_tr)
        best = s.best_estimator_
        oof = np.zeros(len(train))
        for tr, va in splits:
            oof[va] = clone(best).fit(train.iloc[tr], y_tr[tr]).predict(train.iloc[va])
        pred = best.predict(test)
        rows.append({"model": name, "params": s.best_params_,
                     **{f"CV {k}": v for k, v in price_metrics(y_tr, oof).items()},
                     **{f"test {k}": v for k, v in price_metrics(y_te, pred).items()}})
        print(f"{name}: {s.best_params_} {time.time() - t0:.0f}s  CV MAE {rows[-1]['CV MAE (KZT)']:,.0f}  "
              f"test MAE {rows[-1]['test MAE (KZT)']:,.0f}", flush=True)
    out = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.width", 250)
    print(out.drop(columns="params").round(3).to_string())


if __name__ == "__main__":
    main()
