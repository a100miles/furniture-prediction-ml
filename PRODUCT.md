# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React, TypeScript, and Tailwind CSS frontend with a local Python API that loads the existing joblib and PyTorch models.

## Users

Primary users are the two student presenters operating the app during a timed university project defense. The secondary audience is the instructor evaluating whether saved models can predict on genuinely new input.

## Product Purpose

Provide a reliable, clear local demonstration of furniture-type prediction from specifications, a product photo, or photo plus specifications. Success means a presenter can enter or upload new input, receive an interpretable prediction, and explain the model and data source without exposing implementation complexity.

## Positioning

The interface exposes two evidence paths from the submitted project: a mebel.kz structured model and lawful Amazon Berkeley Objects photo models, with their provenance and validation context visible at prediction time.

## Operating Context

The app runs locally during a defense with approximately ten minutes shared between the presentation and live demo. It must start from submitted code, load submitted saved models and preprocessing, and remain understandable to an instructor seeing it for the first time.

## Capabilities and Constraints

- Preserve the existing structured, photo-only, and combined prediction behavior.
- Keep inference in Python because the submitted artifacts are joblib and PyTorch models.
- Accept validated dimensions, price, material, colour, weight, and JPG/PNG photo inputs as appropriate.
- Explain uncertainty and the different training domains without implying that the ABO model was trained on mebel.kz.
- Run locally and avoid a hosted-service dependency.
- Price prediction is not implemented yet and must not be implied.

## Evidence on Hand

- `app.py` contains the current working Streamlit interaction and inference path.
- `models/structured_best.joblib`, `models/abo_photo.joblib`, and `models/abo_specs_photo.joblib` are the submitted saved models.
- `final.ipynb`, `DATA_CARD.md`, and `CHANGES.md` contain the supporting metrics, source rights, and limitations.
- No logo, product photography for decoration, or independent brand system is established.

## Product Principles

- Make the prediction path obvious before showing methodological detail.
- State model provenance and limitations beside the result.
- Prefer a controlled, rehearsable demo over feature breadth.
- Keep every displayed claim traceable to submitted evidence.
- Make invalid, missing, and uncertain states calm and recoverable.

## Accessibility & Inclusion

Use semantic controls, visible labels, keyboard focus, sufficient contrast, reduced-motion support, and responsive layouts suitable for a projected laptop screen.
