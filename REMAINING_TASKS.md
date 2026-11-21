# Remaining project tasks

> **Superseded.** This review is from the Endterm (11 November). For the Final, see `HANDOFF.md` (status on 21 November) and the checklist in `CONTRIBUTIONS.md`.

Reviewed on 11 November 2026 against the files in this folder. The team subsequently confirmed a Thursday 12 November defense and submission by Wednesday 11 November at 23:59. Moodle, uploads, completed personal work and rehearsal have not been independently verified.

## Already present

- Both notebooks have saved execution counts and outputs: 26 Endterm code cells and 33 Midterm code cells, with no saved error outputs. A fresh run was not performed during this review.
- Endterm sections cover grouping fixes, tuned forest/boosting/SVM, PCA and k-means with stability checks, an MLP learning curve and search, model comparison, error analysis, conclusions and a Final plan.
- The data checks reproduce 3,961 unique listings, 793 unique saved test URLs, 97 listings without a series, and 3,127 training listings after excluding 41 test-group twins. The 359 training groups and 100 test groups do not overlap.
- The saved comparison reports forest CV macro-F1 0.958 +/- 0.006 and test 0.969. The selected model file exists; notebook section 5.4 records a reload/prediction equality check.
- Both slide PDFs, the change log, data card, requirements and run instructions are present. The Endterm deck has 11 pages, including the settings appendix.

## Endterm: finish before submission

| Priority | Task | Evidence / completion condition |
|---|---|---|
| Assigned | Carry out the contribution and review responsibilities. | `CONTRIBUTIONS.md` now assigns leads/reviewers, validation tasks and timed speaking roles, and discloses Codex assistance. Notebook section 7 agrees. Fresh execution, personal review and rehearsal remain unchecked until performed. |
| Confirmed | Submit Wednesday 11 November 2026 by 23:59; defend Thursday 12 November. | Confirmed by the team in this conversation using the day-before-defense rule; the guide has been updated. No further deadline clarification is pending. |
| High | Run the Endterm notebook from a fresh kernel in the intended Python environment; save the outputs and compare the final numbers with the slides. | Saved outputs show a completed earlier run, not a fresh reproducibility check. Confirm installation of `requirements.txt`, successful completion of all 26 code cells, the unchanged test URLs, and the saved model reload check. |
| Medium | Correct the boosting interpretation in the notebook and slides. | Notebook cell 22 and Endterm slide 5 call leaf size 20 the "strongest brake" tried, but the search also includes 40. Say that all five displayed best configurations use 20; the table does not establish the claimed causal explanation about fitting individual listings. |
| Medium | Align the practice guide with the completed notebook. | Section 11 answer 13 uses "Тумба Воямполь" as an example wrong for every model; notebook section 5.3 identifies three colours of the "Пенни" desk. Answer 7 says k = 3 to 10, while the code searches 2 to 10. Treat the guide as preparation material, not the authoritative result record. |
| Medium | Check claims about uncertainty and what the features can prove. | Notebook sections 5-6 infer test-score noise from designs and make absolute claims that no specification-based model can fix certain listings. No group-bootstrap interval or impossibility proof is supplied. Qualify these as explanations/hypotheses, or add a suitable group-level uncertainty analysis if retaining quantitative uncertainty claims. |
| High | Rehearse the seven-minute talk and the five-minute questions, with both partners speaking. | Assign slide ownership and time the talk. Each partner should explain grouping, tuning, macro-F1, model selection by CV, the MLP curve, the errors and their own contribution. Rehearsal cannot be verified from the files. |
| High | Build and inspect the submission archive, then both partners upload it. | No ZIP exists in this folder. Use `Endterm_Zhumagali_Sultanbek_SE-2420.zip`; include the notebook with outputs, slides, required docs, data, saved test URLs and model. Exclude temporary tooling and virtual environments. Open/extract the archive to verify it. Upload status is unknown. |

## Final: implementation still to do

These are planned in `endterm.ipynb`, section 6, and are not missing Endterm model families. No photo dataset, image-training code or app source is present in this folder.

1. Implement photo collection: obtain image URLs from the existing product pages, check current source/CDN access rules, download at the documented rate, and create a manifest keyed by the existing listing URL. The current scraper does not save the Product `image` field. Record missing/unusable images.
2. Preserve the existing listing split and grouping for photos. Check repeated/shared images across groups and fit learned preprocessing only on training data.
3. Train and evaluate a small CNN and a pretrained image model; save their models, preprocessing and results.
4. Build a combined photo + specification model, including structured-only fallback when an image is missing. Compare structured, photo-only and combined results on the same test listings and on the photo-available subset. Report macro-F1, per-type F1 and desk/chest/TV-stand errors, including the named difficult designs.
5. Build the local Streamlit app with validated dimensions, price, material and colour inputs, photo upload, prediction and probability output. The saved structured pipeline expects engineered columns such as `footprint_m2`, `log_price` and material flags; move/reuse `add_features` in shared inference code and preserve the saved feature order. Loading the pipeline alone does not turn raw form inputs into those columns.
6. Add inference/image/app dependencies and reproducible run instructions, then prepare the Final notebook/report, slides and contributions according to the actual Final rubric. The original rubric is not included locally, so confirm its complete deliverable list before submission.

## Continuing limitations

Store-label ambiguity, missing weight, one store/city and one price snapshot remain documented limitations. A manual audit of borderline labels would improve the evidence, but relabelling must be documented and must not silently alter the frozen evaluation protocol.

## Review limits

No live website or Moodle checks, model retraining, model loading, submission upload or rehearsal was performed. The saved artifact's runtime compatibility still needs the fresh-run check above. Sultanbek's surname was corrected to **Zhumagali** in the Markdown documents, both notebooks and both slide decks as part of this review.
