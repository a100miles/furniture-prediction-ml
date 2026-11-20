# Endterm guide - Furniture type from specifications (MLA)

Course: Machine Learning Algorithms (Kenzhe Damir), team Sultanbek Zhumagali + Tamerlan Yessimov, SE-2420.
Built from: Moodle (Endterm Submission Box, Project Guide, Endterm Rubric, Final Rubric, Course Project Registration page) and your midterm zip.

How to use this guide: every section has **Learn** (the idea in plain words), **Do** (tested code), **Show** (what goes in the notebook and slides) and **Be ready for** (defense questions). The code was run on your real data with your package versions (scikit-learn 1.9.1, pandas 3.0.6). The numbers I got are written down so you can check your own run against them. Your numbers should come out the same if nothing else changes.

---

## 0. The facts

| What | Detail |
|---|---|
| Applicable submission deadline | **Wednesday 11 November 2026, 23:59**, the day before the **Thursday 12 November 2026** defense. The team confirmed this defense date and deadline rule on 11 November. Both partners upload the same archive. |
| File name | `Endterm_Zhumagali_Sultanbek_SE-2420.zip` (both partners upload the same archive) |
| Defense | 12 minutes: **7 min talk + 5 min questions**. Both of you speak. Either of you can be asked about any part. |
| Grade | Team points (criteria 1-7, out of 85) + your own answers (criterion 8, out of 15). It is 50% of attestation 2. |
| What is new | Tuned ensembles, unsupervised analysis, an MLP neural network, one fair table of all models, a Final plan. Text/image models are **optional** now (they are required at the Final). |
| Midterm | 84/100. Moodle has **no written comments**, only the grade. Write down what the teacher said at your midterm defense: criterion 1 needs a change log that links feedback to fixes. |

Rubric points (memorise this, it is your to-do list):

| # | Criterion | Points |
|---|---|---|
| 1 | Continuity and fixes since the Midterm | 10 |
| 2 | Tuned ensembles (forest + boosting, CV on training data) | 15 |
| 3 | Unsupervised analysis (clustering or PCA), interpreted | 10 |
| 4 | Neural network (MLP) with a learning check, fair comparison | 15 |
| 5 | Fair comparison + error analysis across all models | 15 |
| 6 | Conclusions, limitations, plan for the Final | 10 |
| 7 | Presentation, 7 minutes | 10 |
| 8 | Your individual answers | 15 |

The rubric says it twice: **"A high metric earns nothing by itself."** Points go to *fair comparisons, justified choices and explained errors*. Your midterm was already strong on this, so keep the same style.

Learning material on Moodle (MLA course page): **Lecture 5 to Lecture 8 PDFs**, and the **Student Zip Packs** for Week 6 (Activity 1-2), Week 7 (Activity 1-2, Demo 3) and Week 8 (Activity 1-2). Per the syllabus table in your midterm plan: Week 5 = ensembles + tuning, Week 6 = PCA + clustering, Week 7 = neural networks, Week 8 = CNN / transfer learning. Read the lecture first, then the activity notebook, then do the step below.

---

## 1. Confirmed plan: defense on Thursday 12 November

**Current plan:** submit by **Wednesday 11 November, 23:59** and defend on **Thursday 12 November 2026**. The ownership table and timed speaking roles are in `CONTRIBUTIONS.md`. The original preparation tracks below are historical planning notes; the defense date is now settled.

You upload by 23:59 **the day before** you defend:
- Tuesday defense: upload by **Monday 9 Nov, 23:59**. That leaves 2.5 days.
- Thursday defense: upload by **Wednesday 11 Nov, 23:59**. That leaves 4.5 days.

**Be honest with yourselves about time.** You are starting ML from zero. In 2.5 days you cannot both learn ML *and* write every line by hand. What works: AI helps write the code (allowed, as long as you say so in `CONTRIBUTIONS.md`), and **your** time goes into understanding every cell. Criterion 8 (15 points each) is the one that checks your understanding, and it is the one AI can't do for you.

**Decision rule:** if the endterm notebook runs from top to bottom by **Sunday night**, and you can answer the section 11 questions with it, go Tuesday. If not, go Thursday. A calm Thursday beats a panicked Tuesday.

### Tuesday track

| When | Do | Learn |
|---|---|---|
| Sat 7, evening | Read **Part A (ML from zero)** below, slowly. Open the midterm notebook and find each idea from Part A in it. | Part A |
| Sun 8, morning | Steps 2-4: setup, fixes, ensembles | Part A again + "Learn" boxes in steps 3-4 |
| Sun 8, afternoon | Steps 5-6: PCA/clustering, MLP | "Learn" boxes in steps 5-6 |
| Sun 8, evening | **Checkpoint:** does the notebook run top to bottom? Yes: stay on Tuesday. No: switch to Thursday. | - |
| Mon 9, day | Steps 7-8: big table, errors, conclusions. Slides (section 9). README, `CHANGES.md`, `CONTRIBUTIONS.md` | - |
| Mon 9, before 23:59 | Zip, **both** upload, open the zip once to check it | - |
| Mon 9, night / Tue morning | Rehearse the 7-minute talk twice with a timer. Quiz each other with section 11 (answers are there) | Section 11 |

### Thursday track

| When | Do |
|---|---|
| Sat 7 | Part A + midterm notebook walk-through |
| Sun 8 | Steps 2-4 |
| Mon 9 | Steps 5-6 + Lecture 5-7 PDFs (just the slides that match each step) |
| Tue 10 | Steps 7-8 + slides |
| Wed 11 | Docs, Restart & Run All, zip, upload by 23:59. Evening: rehearse + quiz |

**Agreed responsibility split.** Sultanbek leads fixes, ensembles, MLP and execution/model checks. Tamerlan leads PCA/clustering, comparison and error analysis, conclusions/Final plan and consistency checks across documents and slides. Each reviews the other's sections. Both prepare the slides, rehearse, inspect the archive and upload it. See `CONTRIBUTIONS.md` for the detailed assignments and seven-minute speaking schedule.

---

## Part A. ML from zero, using your own project

Read this first. Every idea is explained with your furniture data, so it sticks.

**1. What the project does.** You have a table: 3,961 rows, one row = one furniture listing from mebel.kz. Each row has numbers (width, depth, height, price), materials (chipboard, fabric...) and a colour. Each row also has the **answer**: its type (sofa, bed, wardrobe... 9 types). The goal: a program that looks at the numbers and guesses the type. Guessing a category like this is called **classification**.
- The inputs (width, height, price...) are called **features**.
- The answer (furniture type) is called the **label** or **target**.
- A **model** is the program that learns the link between features and label from examples.

**2. Why it is useful.** A marketplace could check if a seller put a listing in the wrong category: "you said chest of drawers, but these sizes look like a TV stand".

**3. Training and testing.** You show the model most rows with their answers (the **training set**, 3,168 rows at the midterm). Then you check it on rows it has **never seen** (the **test set**, 793 rows). It is like an exam: if you test a student on the exact questions they practised, you learn nothing. The test set is opened **once**, at the very end. The list of test rows is saved in `data/processed/test_urls.csv` and must stay the same at the endterm and the Final.

**4. Overfitting = memorising.** A model can memorise the training rows perfectly (train score 1.00) but fail on new rows. In the midterm, the decision tree got 0.993 on training rows and only 0.899 on new ones. That gap is overfitting.

**5. Cross-validation (CV).** You don't want to touch the test set while choosing between models. So you split the **training** rows into 5 parts (**folds**). Train on 4, check on the 5th, repeat 5 times so each part is checked once. You get 5 scores, and you report their **mean** (how good on average) and **standard deviation** (how much it jumps around; small = stable). All choices (which model, which settings) are made with CV, never with the test set.

**6. Grouping by series (your project's special trick).** mebel.kz sells the same model in many colours, e.g. "Комод Леванте" in white, grey, beige. Same size, same price. If white is in training and grey is in checking, the model just "recognises" it, which is cheating. So all colours of one series always stay on the same side. That is `StratifiedGroupKFold` with `groups=series`. Your midterm showed that without grouping, scores look about 0.05 too good.

**7. Baseline.** The dumbest possible model, e.g. "always say wardrobe" (the biggest class). It scores 0.033. Any real model must beat it by a lot, or it learned nothing.

**8. Metric: why macro-F1 and not accuracy.** Accuracy = share of correct guesses. But classes are unequal (wardrobes 18%, coffee tables 4%). A model could ignore coffee tables and still have high accuracy. **F1** for one class mixes two questions:
- **precision:** when the model says "coffee table", how often is it right?
- **recall:** of all real coffee tables, how many did it find?

**Macro-F1** = the F1 of each of the 9 types, averaged with equal weight. So small classes count as much as big ones.

**9. Leakage.** An input that secretly contains the answer. The product name "Кровать Нордик 160" contains the word "bed", so the name is **not** used as an input. The midterm also showed that *which spec fields a page has* gives away the type (only beds have "Основание кровати"), so those aren't used either. Leakage makes a model look great on paper and useless in real life.

**10. Scaling and the pipeline.** Width is around 100, price is around 300,000. Some models (KNN, SVM, neural networks) measure distances, so without scaling, price would drown everything. `StandardScaler` puts every number on the same scale (mean 0, spread 1). The **pipeline** bundles "scale, then model" together, so the scaler only learns from training rows in each fold. Otherwise information from checking rows leaks in.

**11. The three midterm models.**
- **Decision tree:** a chain of yes/no questions. "Height over 150 cm? Then wardrobe. Otherwise: depth under 50 cm? ..." Easy to read, but unstable: small data changes give a different tree.
- **KNN (k nearest neighbours):** to classify a listing, find the 15 most similar listings in training and take their majority type. "You look like these, so you are what they are."
- **SVM:** draws the widest possible border between types; with the RBF kernel the border can curve. It was your best midterm model (CV 0.941, test 0.978).

**12. Hyperparameters.** Settings you choose *before* training, like the tree's maximum depth or KNN's k. You pick them with CV: try several values, keep the one with the best CV score. That is **tuning**.

**13. Confusion matrix.** A 9x9 table: rows = true type, columns = predicted type. The diagonal = correct. Off the diagonal = mistakes. Yours showed the main mistake: **chest of drawers vs TV stand** (similar depth, similar materials, the store itself labels some low cabinets inconsistently).

**14. What the endterm adds** (each explained in its step below):
- **Ensembles** (step 4): many trees voting together, more stable than one tree.
- **PCA and clustering** (step 5): looking at the data *without* the answers, to see its natural shape.
- **Neural network / MLP** (step 6): a more flexible model made of layers of simple units.
- **One fair comparison** (step 7): every model, same rows, same folds, same metric.

When Part A feels clear, open `furniture_classification.ipynb` (the midterm notebook) and find each of these 14 ideas in it. That is your best preparation, because the endterm questions build on the midterm.

---

## 2. Setup

**Do**

1. Done already: this folder (`~/projects/mla-furniture-endterm`) has all the midterm files (data, scraper, data card, contributions, midterm notebook and slides). When you zip at the end, name the archive `Endterm_Zhumagali_Sultanbek_SE-2420.zip`.
2. Add `joblib` to `requirements.txt` (it comes with scikit-learn, but list it because you will save models).
3. Create `endterm.ipynb`. Copy from the midterm notebook: imports, `RANDOM_STATE = 42`, `TYPES`, the cleaning cell, `add_features`, `NUMERIC / MATERIALS / COLOUR / FEATURES`, and `make_pipeline`.
4. **Most important rule:** do NOT recompute the split. Load it from the saved file, so the test rows are guaranteed identical:

```python
test_urls = set(pd.read_csv("data/processed/test_urls.csv")["url"])
is_test = df["url"].isin(test_urls)
assert is_test.sum() == 793, "test set changed!"
```

**Be ready for:** "How do you know the test set is the same as at the Midterm?" - We load the 793 saved test URLs and assert the count; we never re-split.

---

## 3. Criterion 1 - Fixes and change log (10 pts)

**Learn.** Your midterm promised two fixes (section 5, "Fixed first"):
- 97 listings had no series, so they were grouped by their SKU, and their colour twins could sit in both train and validation folds. That leaks (the model has "seen" the same item in another colour).
- Some training rows are colour twins of test rows. That makes the test score slightly too optimistic.

Fix both. The test set itself does not change.

**Do**

```python
# 1. Better groups: listings without a series are grouped by their exact specs
#    (colour variants share width, depth, height, price and materials)
no_series = df["series"].astype(str) == df["sku"].astype(str)
spec_key = df[["width_cm", "depth_cm", "height_cm", "price_kzt", "materials"]].astype(str).agg("|".join, axis=1)
df["group"] = np.where(no_series, "spec:" + spec_key, df["series"].astype(str))

# 2. Remove colour twins of test rows from training
test_groups = set(df.loc[is_test, "group"])
twin = (~is_test) & df["group"].isin(test_groups)
print("rows without series:", no_series.sum(), "| training twins removed:", twin.sum())

train, test = df[~is_test & ~twin], df[is_test]
X_train, y_train, g_train = train[FEATURES], train["furniture_type"], train["group"]
X_test, y_test = test[FEATURES], test["furniture_type"]
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
```

What I got: 97 rows without series, **41 training twins removed**, train 3127 rows, test 793 rows.

Then re-run the midterm best model (SVM, C=3) with the new groups to get the "after" number. I got **CV macro-F1 0.946 ± 0.018** (midterm: 0.941 ± 0.024) and test **0.975** (midterm: 0.978). The test score dropped a little, which is what you expect once the twins are gone, and that is a good thing to say out loud.

**Show:** a `CHANGES.md` file (and the same table in the notebook):

| Feedback or weakness | Fix | Before | After |
|---|---|---|---|
| (what the teacher said at the midterm defense) | ... | ... | ... |
| 97 listings without series grouped by SKU | grouped by exact specs | CV 0.941 ± 0.024 | CV 0.946 ± 0.018 |
| colour twins of test rows in training | 41 rows removed from training | test 0.978 | test 0.975 |
| single tree unstable (± 0.053) | ensembles, step 4 | ... | ... |

Also add a line "Remaining issues" (for example: label noise at chest / TV stand, no weight feature). The rubric wants issues named honestly.

**Be ready for:** "Why did the test score go down after the fix?" - The removed rows were near copies of test items, so the old score was a bit optimistic. The new one is more honest.

---

## 4. Criterion 2 - Tuned ensembles (15 pts)

**Learn**
- **Random forest (bagging):** many deep trees, each trained on a random sample of rows and a random subset of features at each split, then they vote. Averaging cancels out the random mistakes of single trees, which lowers **variance**. This fixes your single tree's instability.
- **Gradient boosting:** small trees built **one after another**, and each new tree fixes the errors left by the previous ones. This lowers **bias**. It needs a small learning rate and limits on tree size, or it overfits.
- **Tuning with CV on training data only:** `GridSearchCV` (tries every combination) or `RandomizedSearchCV` (tries N random combinations, faster). Pass `groups=` so the folds stay grouped by series. Because the scaler is inside the pipeline, it is re-fitted inside each training fold, so preprocessing is "fitted within training folds" (the rubric asks for exactly that).

What each setting means:

| Model | Setting | Meaning |
|---|---|---|
| Forest | `n_estimators` | number of trees (more is more stable, just slower) |
| Forest | `max_depth` | how deep each tree may grow |
| Forest | `max_features` | how many features each split may look at (`sqrt` = more randomness, trees less alike) |
| Forest | `min_samples_leaf` | smallest leaf; bigger = smoother, less memorising |
| Boosting | `learning_rate` | how big each correction step is |
| Boosting | `max_leaf_nodes` | size of each small tree |
| Boosting | `min_samples_leaf`, `l2_regularization` | brakes against overfitting |

**Do**

```python
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def search_report(s, name):
    i = s.best_index_
    print(f"{name}: best {s.best_params_}")
    print(f"   CV macro-F1 {s.cv_results_['mean_test_score'][i]:.3f} ± {s.cv_results_['std_test_score'][i]:.3f}")

rf_search = GridSearchCV(
    make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    {"model__n_estimators": [300],
     "model__max_depth": [None, 10, 20],
     "model__max_features": ["sqrt", 0.5],
     "model__min_samples_leaf": [1, 3]},
    cv=cv, scoring="f1_macro").fit(X_train, y_train, groups=g_train)
search_report(rf_search, "Random forest")

hgb_search = RandomizedSearchCV(
    make_pipeline(HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
    {"model__learning_rate": [0.03, 0.1, 0.2],
     "model__max_leaf_nodes": [15, 31, 63],
     "model__min_samples_leaf": [10, 20, 40],
     "model__l2_regularization": [0, 0.1, 1.0]},
    n_iter=12, cv=cv, scoring="f1_macro", random_state=RANDOM_STATE).fit(X_train, y_train, groups=g_train)
search_report(hgb_search, "Gradient boosting")
```

Each search takes about 1.5 minutes. What I got:

| Model | Best settings | CV macro-F1 | Test macro-F1 |
|---|---|---|---|
| SVM (midterm best, new groups) | C=3 | 0.946 ± 0.018 | 0.975 |
| Random forest | depth None, max_features sqrt, leaf 1, 300 trees | **0.958 ± 0.006** | 0.969 |
| Gradient boosting | lr 0.1, 15 leaves, leaf 20, l2 0 | 0.952 ± 0.008 | 0.938 |

Also do what you promised in the midterm plan: tune the SVM's `C` and `gamma` **together** (step 7 uses `svm_search`):

```python
svm_search = GridSearchCV(
    make_pipeline(SVC()),
    {"model__C": [1, 3, 10, 30], "model__gamma": ["scale", 0.03, 0.1, 0.3]},
    cv=cv, scoring="f1_macro").fit(X_train, y_train, groups=g_train)
search_report(svm_search, "SVM")
```

**The big lesson here.** The forest wins in CV, but the SVM wins on test. **You pick by CV, not by test.** The test set is one grouped split of 103 series, and the CV mean covers 5 splits. Look at the spread too: the forest's ± 0.006 is much smaller than the SVM's ± 0.018, so the forest is also the more **stable** model, which was exactly the weakness you planned to fix. Say this in the slides. It is the kind of "justified decision" the rubric rewards.

**Show:** search space table, chosen settings, CV mean ± sd, and a sentence on why the spread matters.

**Be ready for:** bagging vs boosting; why `max_features` makes trees different; why the scaler doesn't matter for trees but you keep it in the pipeline anyway (same pipeline for every model = fair comparison); "why not tune on the test set?" (you would be choosing the model that is lucky on those 793 rows, so the test score stops being an honest estimate).

---

## 5. Criterion 3 - Unsupervised analysis (10 pts)

**Learn**
- **PCA** finds new axes (components) that are mixes of your features, ordered by how much variation they explain. With 9 numeric features that are all built from 3 sizes + price, many are redundant.
- **k-means** puts rows into k groups by distance, **without** labels. You must scale features first (otherwise price in tenge would dominate cm).
- **Silhouette score**: how well separated the clusters are (closer to 1 is better).
- **Stability (ARI)**: run k-means with different seeds and compare the labelings. ARI = 1 means identical clusters. Unstable clusters are not "real groups".
- The rubric warns: **a 2D map is not proof of real groups.** Always check stability and explain clusters with real features and real example listings.

**Do** (fit on **training rows only**)

```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

# PCA on the 9 scaled numeric features
Z = StandardScaler().fit_transform(X_train[NUMERIC])
pca = PCA().fit(Z)
print("cumulative explained variance:", pca.explained_variance_ratio_.cumsum().round(3))
# plot: components 1-2 coloured by furniture_type; print pca.components_ as a table to name each axis

# k-means on scaled height, width, depth, log price
CL = ["height_cm", "width_cm", "depth_cm", "log_price"]
Zc = StandardScaler().fit_transform(X_train[CL])
for k in range(3, 11):
    print(k, round(silhouette_score(Zc, KMeans(k, n_init=10, random_state=RANDOM_STATE).fit_predict(Zc)), 3))

k = 6
runs = [KMeans(k, n_init=10, random_state=s).fit_predict(Zc) for s in range(5)]
print("ARI vs seed 0:", [round(adjusted_rand_score(runs[0], r), 3) for r in runs[1:]])
pd.crosstab(runs[0], y_train.values)   # which types land in which cluster
```

What I got:
- PCA: 2 components explain 72%, **3 explain 90%**, 4 explain 97%. Your 9 numeric inputs really carry about 3-4 independent directions (size, height/shape, price). That supports what you suspected in the midterm plan.
- Silhouette is best at **k = 3** (0.49). That matches your midterm finding: height splits the catalogue into three levels (tall wardrobes, middle, low tables/stands).
- At k = 6 the ARI across seeds is only 0.65-0.87, so the six clusters are **not stable**. Say this honestly. Two wardrobe clusters come out clean, but one cluster mixes chests, TV stands, sofas and beds.

**Connect it to the models** (the rubric gives 4/4 only for this): pick one of these.
- (a) Error rate per cluster: get out-of-fold predictions from your best model with `cross_val_predict(..., groups=g_train, cv=cv)`, then `pd.crosstab(cluster, oof != y_train, normalize="index")`. Show that the errors pile up in the "low, wide case furniture" cluster where chests and TV stands live.
- (b) PCA inside the pipeline: `Pipeline([("prep", ...), ("pca", PCA(5)), ("model", KNN)])`. I got KNN 0.888 without PCA, 0.882 with 3 components, 0.894 with 5. The differences are smaller than the spread, so "PCA does not really help KNN". That is a fine, honest result.

**Show:** explained variance plot, 2D PCA map, silhouette by k, the crosstab, 3-4 real listing names per interesting cluster (names only for explanation, never as input), the stability numbers.

**Be ready for:** why scale before k-means; how you chose k; what PC1 means (look at its loadings); why clusters don't prove categories.

---

## 6. Criterion 4 - Neural network, MLP (15 pts)

**Learn**
- An **MLP** = layers of neurons. Each neuron computes a weighted sum, then a non-linear function (ReLU). Training = repeat over the data (**epochs**), and adjust weights a little each time to lower the **loss** (backpropagation + Adam optimizer).
- Settings: `hidden_layer_sizes` (e.g. (64, 32) = two layers), `alpha` (L2 penalty, fights overfitting), `learning_rate_init`, `batch_size`.
- **Learning curve check:** plot train and validation score per epoch. Train keeps rising and validation flattens = the **plateau** (more epochs won't help). Validation going down while train goes up = **overfitting**.
- MLPs need scaled inputs. Your pipeline already does that.

**Trap to avoid (good thing to mention at the defense):** `MLPClassifier(early_stopping=True)` takes a **random** 10% validation split. That ignores your series groups, so colour twins land in both sides and the curve looks too good. Make your own **grouped** validation split instead:

```python
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit

tr_i, va_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
                  .split(X_train, y_train, g_train))
prep = make_prep().fit(X_train.iloc[tr_i])   # fitted on the training part only
A, B = prep.transform(X_train.iloc[tr_i]), prep.transform(X_train.iloc[va_i])
ya, yb = y_train.iloc[tr_i], y_train.iloc[va_i]

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=64, random_state=RANDOM_STATE)
classes = np.unique(y_train)
curve = []
for epoch in range(1, 151):
    mlp.partial_fit(A, ya, classes=classes)          # one pass over the data
    if epoch % 5 == 0:
        curve.append((epoch, f1_score(ya, mlp.predict(A), average="macro"),
                             f1_score(yb, mlp.predict(B), average="macro")))
curve = pd.DataFrame(curve, columns=["epoch", "train", "validation"]).set_index("epoch")
curve.plot(title="MLP learning curve (grouped validation)", ylabel="macro-F1")
```

For this, split your midterm `make_pipeline` into two functions in step 2, so you can get the preprocessor alone:

```python
def make_prep():
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("materials", "passthrough", MATERIALS),
        ("colour", OneHotEncoder(handle_unknown="ignore", min_frequency=10), COLOUR),
    ])

def make_pipeline(model):
    return Pipeline([("prep", make_prep()), ("model", model)])
```

What I got: train reaches 1.000 by epoch 50, validation climbs to about 0.975 and stays flat from epoch 50 on. That is a plateau, with mild overfitting (train is perfect).

Then tune a little with the same grouped CV and compare fairly:

```python
for hs in [(32,), (64, 32), (128, 64)]:
    for a in [1e-4, 1e-2]:
        r = cross_validate(make_pipeline(MLPClassifier(hs, alpha=a, max_iter=300, random_state=RANDOM_STATE)),
                           X_train, y_train, groups=g_train, cv=cv, scoring="f1_macro")
        print(hs, a, f"{r['test_score'].mean():.3f} ± {r['test_score'].std():.3f}")
```

What I got: best (128, 64), alpha 0.01: **0.939 ± 0.011**, so **below the forest (0.958) and the SVM (0.946)**. That is a fine result. Explain why: only about 3,100 rows, from about 400 series, with mostly simple threshold rules (height!). Trees and SVMs are built for that, and an MLP needs more data to shine. Your midterm plan predicted this: "a similar score would show the limit is the labels, not the model."

**Be ready for:** what an epoch is; what alpha does; how you know it converged; why the MLP doesn't beat the forest; why you didn't use `early_stopping=True` as is.

---

## 7. Criterion 5 - One fair table + error analysis (15 pts)

**Learn.** "Fair" = same training rows, same grouped folds, same metric, same test rows, and the choice made on CV only. Keep **macro-F1** as the main metric (classes are imbalanced: coffee tables are 4%, wardrobes 18%) and say why again.

**Do**

```python
best_models = {
    "dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
    "dummy (stratified)": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
    "decision tree (depth 10)": DecisionTreeClassifier(max_depth=10, min_samples_leaf=3, random_state=RANDOM_STATE),
    "KNN (k=15)": KNeighborsClassifier(15, weights="distance"),
    "SVM (tuned)": svm_search.best_estimator_.named_steps["model"],
    "random forest (tuned)": rf_search.best_estimator_.named_steps["model"],
    "boosting (tuned)": hgb_search.best_estimator_.named_steps["model"],
    "MLP (best)": MLPClassifier((128, 64), alpha=1e-2, max_iter=300, random_state=RANDOM_STATE),
}
rows, test_preds = [], {}
for name, model in best_models.items():
    r = cross_validate(make_pipeline(model), X_train, y_train, groups=g_train, cv=cv, scoring="f1_macro")
    fitted = make_pipeline(model).fit(X_train, y_train)
    test_preds[name] = fitted.predict(X_test)
    rows.append({"model": name, "CV macro-F1": r["test_score"].mean(), "CV sd": r["test_score"].std(),
                 "test macro-F1": f1_score(y_test, test_preds[name], average="macro")})
summary = pd.DataFrame(rows).set_index("model").round(3)
summary
```

Then the error analysis, across models:
1. **Confusion matrix** for the chosen model (and for the SVM next to it).
2. **Per-class F1 for every model** in one table (`classification_report(..., output_dict=True)`): which classes improve with ensembles? I expect chest / TV stand.
3. **Errors all models share vs. errors only some make:**
   ```python
   wrong = pd.DataFrame({n: p != y_test.values for n, p in test_preds.items() if not n.startswith("dummy")})
   test_view = test.assign(n_models_wrong=wrong.sum(axis=1).values)
   test_view.sort_values("n_models_wrong", ascending=False)[["name", "furniture_type", "width_cm", "depth_cm", "height_cm", "price_kzt", "n_models_wrong"]].head(12)
   ```
   Listings that every model gets wrong are most likely **label noise** (the "Тумба ... 150x64" filed under "Комоды", the "Письменный стол Пенни" with a 190 cm tower). Errors that only the MLP or only the boosting makes are **model weaknesses**. That is a strong and simple story.
4. End with: **what changed since the midterm and why** (one short paragraph).

**Be ready for:** why macro-F1 and not accuracy; why CV and test disagree; what a confusion matrix row means; one concrete wrong listing and why it is wrong.

---

## 8. Criterion 6 - Conclusions, limitations, Final plan (10 pts)

**Conclusions:** answer the question again, in one sentence, with the new best model and its CV score. Say what the ensembles, PCA/clusters and MLP each taught you.

**Limitations:** tie them to **your** observed errors, not generic ones (chest / TV stand label noise, no weight, one store, form-template materials, about 400 series in training).

**Final plan** (the Final rubric checks that you test **this exact hypothesis** later, so write it carefully):

- **Source: product photos** from the same mebel.kz product pages. Why not text: the product **name** contains the type word in 93% of rows (you proved this in midterm 2.3), and the descriptions start with the name. Text would leak the answer. Photos don't.
- **Lawful access:** product pages are allowed by robots.txt; check that the image paths are allowed too, and keep the same polite settings (2 connections, 0.7 s, named User-Agent). Your raw JSONL did **not** save image URLs, so this is a new collection run. The JSON-LD `Product` block on each page normally has an `image` field.
- **Linking:** by `url`, so every photo belongs to an existing row and the **same 793 test listings** are used. Report how many rows have no photo and how you handle them (fall back to the structured model).
- **Hypothesis:** photos show what the numbers can't (drawers vs open shelves, a TV-stand shape), so they should fix the **chest of drawers vs TV stand** errors. **Test:** F1 on those two classes and the count of chest/TV-stand confusions on the same test set, structured model vs photo model vs combined.
- **Method (the Final rubric asks for both):** a pretrained CNN (e.g. ResNet/EfficientNet features + logistic regression, or fine-tuning) **and** a simple model you train yourselves (a small CNN) to compare against.
- **Also needed at the Final:** a local app (Streamlit is easiest) that loads a saved model. Start now, it costs one line:
  ```python
  import joblib, os
  os.makedirs("models", exist_ok=True)
  joblib.dump(chosen_pipeline, "models/structured_best.joblib")
  ```

---

## 9. Criterion 7 - Slides, 7 minutes (10 pts)

About 8 slides. Figures and tables, **no code**. Label every axis. Both speak (for example, switch at slide 4).

1. Question + data recap (1 line each, the numbers: 3,961 listings, 9 types, 512 series)
2. What we fixed since the midterm (the change log table, before / after)
3. Tuned ensembles: search space, chosen settings, CV mean ± sd
4. Unsupervised: PCA variance + map, clusters k=3 vs k=6, stability, where errors live
5. MLP: learning curve, plateau, score vs others
6. The one big table (all models, CV mean ± sd, test)
7. Errors: confusion matrix + 3 real listings everyone gets wrong
8. Conclusions, limitations, Final plan (photos, hypothesis, how we'll test it)

Rehearse with a timer. 7 minutes is short, so cut, don't speed up.

---

## 10. Submission checklist

The rubric says a missing element caps that criterion, so tick every line:

- [ ] `endterm.ipynb`, run with **Kernel > Restart & Run All**, saved **with outputs** (numbers must match the slides)
- [ ] `data/` (raw + processed) and `data/processed/test_urls.csv` (unchanged)
- [ ] `CHANGES.md` (change log with before / after)
- [ ] Tuning, PCA/clustering, MLP curves, big table, error examples, all inside the notebook
- [ ] Conclusions, limitations, Final source plan
- [ ] Slides PDF
- [ ] `CONTRIBUTIONS.md` updated for the endterm: who did what + **AI tools used and for what**
- [ ] `README.md` with run instructions (venv, `pip install -r requirements.txt`, run the notebook)
- [ ] `models/structured_best.joblib`
- [ ] Zip named `Endterm_Zhumagali_Sultanbek_SE-2420.zip`, **both** of you upload it

About AI use: the rubric allows AI but you must name it. This guide and its code checks were made with Claude Code, so add a line saying so, then make sure each of you can explain every cell yourself. That is criterion 8.

---

## 11. Quiz yourselves (criterion 8, 15 points each)

Answer out loud without notes. If you can't, re-read the step.

1. What is the difference between bagging and boosting? Which one lowers variance?
2. Why does `max_features="sqrt"` help a random forest?
3. Why did you pass `groups=` to `GridSearchCV.fit`? What would go wrong without it? (Your midterm 3.4 showed about +0.05 inflation.)
4. The forest wins in CV, the SVM wins on test. Which do you choose, and why?
5. What does the CV standard deviation tell you?
6. Why scale before k-means and PCA?
7. How did you pick k? What does ARI = 0.65 mean?
8. What does PC1 represent in your data?
9. What is an epoch? What does `alpha` do in an MLP?
10. How do you know your MLP converged? Is it overfitting?
11. Why didn't the MLP beat the forest?
12. Why macro-F1?
13. Show one listing that every model gets wrong and explain why.
14. What did you change since the midterm, and how do you know it helped?
15. Why photos and not text for the Final? How will you test whether they help?
16. What did *you* do, what did your partner do, and what did AI do?

### Answer key

Cover these, try to answer, then check. Say them in **your own words**; a memorised sentence falls apart at the first follow-up question. Where an answer has a number, use the number from **your** notebook run.

1. **Bagging** (random forest): many trees, each trained on a random sample of rows, then they vote. It lowers **variance** (less jumping around). **Boosting**: small trees built one after another, each fixing the previous ones' mistakes. It lowers **bias** (fewer systematic mistakes).
2. Each split only looks at a random subset of features, so the trees end up different from each other. Different trees make different mistakes, and voting cancels them out. If all trees were the same, voting would add nothing.
3. So all colours of one series stay in the same fold. Without it, the model gets checked on a colour copy of something it trained on, and the score looks about 0.05 better than it really is (midterm section 3.4).
4. The one with the best **CV** score (the forest, in my run), because CV is 5 checks and the test set is one. Choosing by the test set would mean picking the model that got lucky on those 793 rows. The forest also has a smaller spread (± 0.006), so it is more stable.
5. How much the score changes from fold to fold. Small = the model behaves the same on different data, so we can trust the mean more. If two models differ by less than their spread, we can't really say one is better.
6. k-means and PCA work with distances and variance. Price in tenge (hundreds of thousands) would drown sizes in cm. Scaling puts all features on the same footing.
7. Silhouette score for k = 2 to 10 (k = 3 was best), plus common sense: 3 matches the three height levels. ARI compares two clusterings, where 1 means identical. 0.65 means the clusters change quite a lot when you only change the random start, so the 6 groups are not solid.
8. Look at its loadings (the weights in `pca.components_[0]`). In this data PC1 is "overall size": width, depth, footprint and volume all have weights of about 0.4 in the same direction. PC2 is "how tall": height and height/width have the biggest weights (0.62 each). So the data mainly varies by "how big" and "how tall", which matches the depth-3 tree from the midterm.
9. One epoch = the model has seen every training row once. `alpha` is a penalty on big weights; a bigger alpha makes the network simpler and less likely to memorise.
10. The learning curve: validation score stops going up (the plateau, about epoch 50). Train reaches 1.00 while validation stays at about 0.975, so there is mild overfitting, but validation doesn't fall, so it isn't serious.
11. Only about 3,100 rows from about 400 series, and the rules are simple thresholds (height above X = wardrobe). Trees are made for that kind of rule. Neural networks shine with much more data or with images and text.
12. The classes are unequal (coffee tables 4%, wardrobes 18%). Macro-F1 gives each type equal weight, so ignoring a small type is punished.
13. Pick the listings that every model gets wrong: "Письменный стол Пенни 132x55" in three colours. It is a desk, but its listed height is 190 cm because of a tall shelving tower, so every model calls it a wardrobe. The sizes alone point the wrong way; a photo would show the desk top. A second example of a fuzzy label is "Тумба Воямполь 150x64": the store files it under chests of drawers, but its shape is a TV stand, and the forest predicts TV stand.
14. Better grouping for the 97 listings with no series, and 41 colour twins of test rows removed from training. Before/after: SVM CV 0.941 to 0.946, test 0.978 to 0.975. The test score went slightly down because the old one was a bit too optimistic. Then the ensembles fixed the single tree's instability.
15. Text leaks the answer: the product name contains the type word in 93% of rows. Photos show what the numbers can't (drawers vs open shelves). Test: on the same 793 test rows, compare the structured model and the photo model on chest vs TV stand errors specifically.
16. Read your `CONTRIBUTIONS.md` and say it plainly. Name the AI tool (Claude Code) and what it did (helped write code, drafted the guide, checked code ran). Then show you understand the parts it helped with. That last bit is the point of the question.
