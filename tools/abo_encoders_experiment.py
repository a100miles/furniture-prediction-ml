"""Stronger photo models on ABO (Final, after the survey), on the same listings and outer folds as abo_experiment.py.

Adds: CLIP ViT-B/32 + logistic regression, DINOv2 ViT-S/14 + logistic regression (frozen encoders, settings
chosen by inner grouped 3-fold CV), specs + each photo encoder, and a fine-tuned ResNet-18 (blocks 3-4 and the
output layer trained on each outer training fold; main photo + one more view, 160 px, 3 epochs).
Then an outside test: every photo model, trained on ABO only, predicts the labelled IKEA catalogue photos.

Writes data/cache/abo_oof_v2.npz (read by final.ipynb section 5.6). The fine-tuned network (about 1 hour on CPU)
is reused from that file when present; pass --retrain-finetune to train it again.
Run: .venv/bin/python tools/abo_encoders_experiment.py   (needs tools/abo_experiment.py's data/cache/abo_oof.npz)
"""
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from furniture.abo import MATERIALS, SIZE, WEIGHT, generic_photos, load_abo, spec_prep  # noqa: E402
from furniture.features import TYPES  # noqa: E402
from furniture.photos import (cached_features, load_images, photo_views, predict_proba,  # noqa: E402
                              train_finetune)

RANDOM_STATE = 42
FT_SIZE, FT_EPOCHS, FT_VIEWS = 160, 3, 2
OLD = ROOT / "data" / "cache" / "abo_oof.npz"
OUT = ROOT / "data" / "cache" / "abo_oof_v2.npz"
IKEA_DIR = ROOT / "data" / "external" / "ikea_images"
torch.set_num_threads(8)


def listing_mean(feats, rows, n):
    emb = np.zeros((n, feats.shape[1]))
    np.add.at(emb, rows, feats)
    return emb / np.bincount(rows, minlength=n)[:, None]


def main():
    d = load_abo()
    v = photo_views(d, generic_photos(d))
    has_photo = np.zeros(len(d), bool)
    has_photo[v["row"].unique()] = True
    d = d[has_photo].reset_index(drop=True)
    v = photo_views(d, generic_photos(load_abo()))
    y = d["furniture_type"].map({t: i for i, t in enumerate(TYPES)}).values
    groups = d["group"].values
    old = np.load(OLD, allow_pickle=True)
    assert list(old["item_id"]) == list(d["item_id"]), "ABO listings changed: rerun tools/abo_experiment.py first"

    # IKEA outside test photos (labelled ones only)
    ik = pd.read_csv(ROOT / "data" / "processed" / "ikea_images.csv")
    ik = ik[ik["ok"]].reset_index(drop=True)
    y_ik = ik["furniture_type"].map({t: i for i, t in enumerate(TYPES)}).values

    X, X_ik, cols = d.copy(), pd.DataFrame(index=ik.index), {}
    for enc in ["resnet18", "clip", "dinov2"]:
        f = cached_features(enc, v["image_id"], v["path"])
        f_ik = cached_features(enc, ik["path"], ik["path"], cache=ROOT / "data" / "cache" / f"ikea_{enc}.npz", img_dir=IKEA_DIR)
        cols[enc] = [f"{enc}_{i}" for i in range(f.shape[1])]
        X = pd.concat([X, pd.DataFrame(listing_mean(f, v["row"].values, len(d)), columns=cols[enc])], axis=1)
        X_ik = pd.concat([X_ik, pd.DataFrame(f_ik, columns=cols[enc])], axis=1)
        print(enc, f.shape, "IKEA", f_ik.shape, flush=True)

    SPEC_NUM = SIZE + WEIGHT
    photo = lambda enc: Pipeline([("scale", ColumnTransformer([("e", StandardScaler(), cols[enc])])),
                                  ("model", LogisticRegression(class_weight="balanced", max_iter=3000))])
    both = lambda enc: Pipeline([("prep", ColumnTransformer([("spec", spec_prep(SPEC_NUM), SPEC_NUM + MATERIALS + ["main_color"]),
                                                             ("e", StandardScaler(), cols[enc])])),
                                 ("model", LogisticRegression(class_weight="balanced", max_iter=3000))])
    grid = {"model__C": [0.001, 0.003, 0.01, 0.1]}
    MODELS = {"photo (CLIP + logistic regression)": photo("clip"),
              "photo (DINOv2 + logistic regression)": photo("dinov2"),
              "specs + photo (CLIP, logistic regression)": both("clip"),
              "specs + photo (DINOv2, logistic regression)": both("dinov2")}
    PHOTO_ONLY = {"photo (ResNet-18 + logistic regression)": photo("resnet18"), **{k: m for k, m in MODELS.items() if k.startswith("photo")}}

    folds = old["folds"]
    FT = "photo (fine-tuned ResNet-18)"
    oof = {name: np.zeros((len(d), len(TYPES))) for name in [*MODELS, FT]}
    chosen = {name: [] for name in MODELS}
    reuse = None
    if OUT.exists() and "--retrain-finetune" not in sys.argv:
        prev = np.load(OUT, allow_pickle=True)
        if list(prev["item_id"]) == list(d["item_id"]) and (prev["folds"] == folds).all():
            reuse = prev
    ik_ft = np.zeros((len(ik), len(TYPES))) if reuse is None else reuse["ikea_probs"][list(reuse["ikea_names"]).index(FT)]
    if reuse is None:
        v2 = v.groupby("row").head(FT_VIEWS)                 # main photo + next view (photo_views sorts by view)
        images = load_images(v2["path"].tolist(), FT_SIZE)
        images_all = load_images(v["path"].tolist(), FT_SIZE)
        images_ik = load_images(ik["path"].tolist(), FT_SIZE, IKEA_DIR)
        pos = {i: k for k, i in enumerate(v2.index)}
    for k in range(5):
        tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=RANDOM_STATE)
        for name, pipe in MODELS.items():
            t = time.time()
            s = GridSearchCV(clone(pipe), grid, cv=inner, scoring="f1_macro", n_jobs=1)
            s.fit(X.iloc[tr], y[tr], groups=groups[tr])
            oof[name][te] = s.predict_proba(X.iloc[te])
            chosen[name].append(s.best_params_["model__C"])
            print(f"fold {k} {name}: C={s.best_params_['model__C']} ({time.time() - t:.0f}s)", flush=True)
        if reuse is not None:
            oof[FT][te] = reuse["probs"][list(reuse["names"]).index(FT)][te]
            continue
        t = time.time()
        vtr = v2[v2["row"].isin(tr)]
        model = train_finetune(images[[pos[i] for i in vtr.index]], y[vtr["row"].values], len(TYPES),
                               epochs=FT_EPOCHS, log=lambda r: print("  ", k, r, flush=True))
        vte = v[v["row"].isin(te)]                          # predict from every photo of a test listing, then average
        p = predict_proba(model, images_all[v.index.get_indexer(vte.index)])
        agg = pd.DataFrame(p).groupby(vte["row"].values).mean()
        oof[FT][agg.index.values] = agg.values
        ik_ft += predict_proba(model, images_ik) / 5        # IKEA: average of the five fold models
        print(f"fold {k} fine-tuned ResNet-18 ({time.time() - t:.0f}s)", flush=True)

    # outside test on IKEA photos: frozen-encoder models refitted on all ABO listings with the most chosen C
    ikea = {}
    for name, pipe in PHOTO_ONLY.items():
        c = Counter(chosen.get(name, [])).most_common(1)[0][0] if name in chosen else 0.01
        ikea[name] = clone(pipe).set_params(model__C=c).fit(X, y).predict_proba(X_ik)
    ikea[FT] = ik_ft
    np.savez_compressed(OUT, folds=folds, y=y, item_id=d["item_id"].values, names=np.array(list(oof)),
                        probs=np.stack(list(oof.values())), chosen=str(chosen), ikea_names=np.array(list(ikea)),
                        ikea_probs=np.stack(list(ikea.values())), ikea_y=y_ik, ikea_path=ik["path"].values)
    for name, p in oof.items():
        print(f"{name}: ABO CV macro-F1 {f1_score(y, p.argmax(1), average='macro'):.3f}")
    present = sorted(set(y_ik))
    for name, p in ikea.items():
        print(f"{name}: IKEA accuracy {(p.argmax(1) == y_ik).mean():.3f}, "
              f"macro-F1 over its {len(present)} types {f1_score(y_ik, p.argmax(1), labels=present, average='macro'):.3f}")


if __name__ == "__main__":
    main()
