"""ABO experiment: specs vs photo vs combined, grouped 5-fold CV with settings chosen inside each fold.

Writes data/cache/abo_oof.npz (cross-validated probabilities of every model), which final.ipynb section 5 reads.
The small CNN (fixed settings, about 20 minutes) is reused from that file when present; pass --retrain-cnn
(or delete the file) to train it again. Run:
    .venv/bin/python tools/abo_experiment.py [--retrain-cnn]
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from furniture.abo import MATERIALS, SIZE, WEIGHT, generic_photos, load_abo, spec_prep  # noqa: E402
from furniture.features import TYPES  # noqa: E402
from furniture.photos import (cached_resnet18_features, load_images, photo_views, predict_proba,  # noqa: E402
                              train_small_cnn)

RANDOM_STATE = 42
CNN_SIZE, CNN_EPOCHS = 64, 15
OUT = Path(__file__).resolve().parents[1] / "data" / "cache" / "abo_oof.npz"
torch.set_num_threads(4)


def main():
    d = load_abo()
    v = photo_views(d, generic_photos(d))
    has_photo = np.zeros(len(d), bool)
    has_photo[v["row"].unique()] = True
    d = d[has_photo].reset_index(drop=True)          # every model is scored on the same listings
    v = photo_views(d, generic_photos(load_abo()))
    y = d["furniture_type"].map({t: i for i, t in enumerate(TYPES)}).values
    groups = d["group"].values
    print(f"{len(d)} listings with a photo, {len(v)} photos, {d['group'].nunique()} groups")

    # photo features: mean ResNet-18 vector over a listing's photos
    feats = cached_resnet18_features(v["image_id"], v["path"])
    emb = np.zeros((len(d), feats.shape[1]))
    np.add.at(emb, v["row"].values, feats)
    emb /= np.bincount(v["row"].values, minlength=len(d))[:, None]
    EMB = [f"r{i}" for i in range(emb.shape[1])]
    X = pd.concat([d, pd.DataFrame(emb, columns=EMB)], axis=1)

    SPEC_NUM = SIZE + WEIGHT
    spec = Pipeline([("prep", spec_prep(SPEC_NUM)), ("model", SVC(class_weight="balanced", probability=True,
                                                                     random_state=RANDOM_STATE))])
    photo = Pipeline([("scale", ColumnTransformer([("r", StandardScaler(), EMB)])),
                      ("model", LogisticRegression(class_weight="balanced", max_iter=3000))])
    both = Pipeline([("prep", ColumnTransformer([("spec", spec_prep(SPEC_NUM), SPEC_NUM + MATERIALS + ["main_color"]),
                                                 ("r", StandardScaler(), EMB)])),
                     ("model", LogisticRegression(class_weight="balanced", max_iter=3000))])
    MODELS = {
        "specs (SVM)": (spec, {"model__C": [1, 3, 10, 30], "model__gamma": ["scale", 0.1]}),
        "photo (ResNet-18 + logistic regression)": (photo, {"model__C": [0.001, 0.003, 0.01, 0.1]}),
        "specs + photo (logistic regression)": (both, {"model__C": [0.001, 0.003, 0.01, 0.1]}),
    }

    outer = StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE)
    folds = np.full(len(d), -1)
    oof = {name: np.zeros((len(d), len(TYPES))) for name in [*MODELS, "photo (small CNN)"]}
    chosen = {name: [] for name in MODELS}
    reuse = None
    if OUT.exists() and "--retrain-cnn" not in sys.argv:
        old = np.load(OUT, allow_pickle=True)
        if list(old["item_id"]) == list(d["item_id"]) and list(old["folds"]) != []:
            reuse = old
    images = None if reuse is not None else load_images(v["path"].tolist(), CNN_SIZE)
    history = pd.read_json(io.StringIO(reuse["history"].item())) if reuse is not None else None
    for k, (tr, te) in enumerate(outer.split(X, y, groups)):
        folds[te] = k
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=RANDOM_STATE)
        for name, (pipe, grid) in MODELS.items():
            t = time.time()
            s = GridSearchCV(clone(pipe), grid, cv=inner, scoring="f1_macro", n_jobs=1)
            s.fit(X.iloc[tr], y[tr], groups=groups[tr])
            oof[name][te] = s.predict_proba(X.iloc[te])
            chosen[name].append(s.best_params_)
            print(f"fold {k} {name}: {s.best_params_} ({time.time() - t:.0f}s)", flush=True)
        # small CNN on the training listings' photos; a listing's prediction = mean over its photos
        if reuse is not None:
            assert (reuse["folds"][te] == k).all(), "folds changed: rerun with --retrain-cnn"
            oof["photo (small CNN)"][te] = reuse["probs"][list(reuse["names"]).index("photo (small CNN)")][te]
            continue
        t = time.time()
        vtr, vte = v[v["row"].isin(tr)], v[v["row"].isin(te)]
        if k == 0:
            # learning curve: a grouped validation split inside fold 0's training listings (the test fold is not used)
            from sklearn.model_selection import GroupShuffleSplit
            a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=RANDOM_STATE).split(tr, groups=groups[tr]))
            va, vb = vtr[vtr["row"].isin(tr[a])], vtr[vtr["row"].isin(tr[b])]
            _, history = train_small_cnn(images[va.index.values], y[va["row"].values], len(TYPES), epochs=CNN_EPOCHS,
                                         x_val=images[vb.index.values], y_val=y[vb["row"].values],
                                         log=lambda r: print("  curve", {k_: round(v_, 3) for k_, v_ in r.items()}, flush=True))
        model, _ = train_small_cnn(images[vtr.index.values], y[vtr["row"].values], len(TYPES), epochs=CNN_EPOCHS)
        p = predict_proba(model, images[vte.index.values])
        agg = pd.DataFrame(p).groupby(vte["row"].values).mean()
        oof["photo (small CNN)"][agg.index.values] = agg.values
        print(f"fold {k} small CNN ({time.time() - t:.0f}s)", flush=True)

    np.savez_compressed(OUT, folds=folds, y=y, item_id=d["item_id"].values, domain=d["domain"].values,
                        names=np.array(list(oof)), probs=np.stack(list(oof.values())),
                        history=history.to_json(), chosen=str(chosen))
    from sklearn.metrics import f1_score
    for name, p in oof.items():
        print(f"{name}: CV macro-F1 {f1_score(y, p.argmax(1), average='macro'):.3f}")


if __name__ == "__main__":
    main()
