"""Local HTTP API for the React Final-project demo."""

from functools import lru_cache
from pathlib import Path
import io

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from furniture.abo import size_features
from furniture.features import TYPES, add_features
from furniture.photos import resnet18_features


ROOT = Path(__file__).resolve().parent
DIST = ROOT / "frontend" / "dist"


@lru_cache(maxsize=1)
def load_models():
    return (
        joblib.load(ROOT / "models" / "structured_best.joblib"),
        joblib.load(ROOT / "models" / "abo_photo.joblib"),
        joblib.load(ROOT / "models" / "abo_specs_photo.joblib"),
    )


def probability_payload(model, probabilities, source):
    values = pd.Series(probabilities, index=model.classes_).reindex(TYPES).fillna(0).sort_values(ascending=False)
    return {
        "prediction": values.index[0],
        "confidence": float(values.iloc[0]),
        "probabilities": [{"type": key, "probability": float(value)} for key, value in values.items()],
        "source": source,
    }


def validate_number(data, name, low, high):
    try:
        value = float(data[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low:g} and {high:g}.")
    return value


def specification_row(data, include_price=True, include_weight=False):
    row = {
        "width_cm": validate_number(data, "width_cm", 10, 500),
        "depth_cm": validate_number(data, "depth_cm", 10, 500),
        "height_cm": validate_number(data, "height_cm", 10, 500),
        "materials": "|".join(data.get("materials", [])),
        "main_color": str(data.get("main_color", "other")),
    }
    if include_price:
        row["price_kzt"] = validate_number(data, "price_kzt", 5_000, 10_000_000)
    if include_weight:
        row["weight_kg"] = validate_number(data, "weight_kg", 0.5, 500)
    return row


async def health(_request):
    try:
        load_models()
        return JSONResponse({"status": "ready", "models": 3})
    except FileNotFoundError:
        return JSONResponse({"status": "missing_models"}, status_code=503)


async def predict_specs(request: Request):
    try:
        data = await request.json()
        row = pd.DataFrame([specification_row(data)])
        structured, _, _ = load_models()
        inputs = add_features(row, materials=structured["materials"])[structured["features"]]
        pipe = structured["pipeline"]
        payload = probability_payload(pipe, pipe.predict_proba(inputs)[0], "mebel.kz specifications")
        payload["model"] = structured["name"]
        payload["cv_macro_f1"] = float(structured["cv_macro_f1"])
        return JSONResponse(payload)
    except (ValueError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    except Exception as exc:
        return JSONResponse({"error": f"Prediction failed: {exc}"}, status_code=500)


async def predict_photo(request: Request):
    try:
        form = await request.form()
        upload = form.get("photo")
        if upload is None or not getattr(upload, "filename", ""):
            raise ValueError("Choose a JPG or PNG product photo.")
        if upload.content_type not in {"image/jpeg", "image/png"}:
            raise ValueError("The photo must be a JPG or PNG file.")
        raw = await upload.read()
        if len(raw) > 10 * 1024 * 1024:
            raise ValueError("The photo must be smaller than 10 MB.")

        image = Image.open(io.BytesIO(raw)).convert("RGB")
        side = max(image.size)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        tensor = torch.from_numpy(np.array(square.resize((224, 224)))).permute(2, 0, 1).unsqueeze(0)

        _, photo_model, combined = load_models()
        embedding = pd.DataFrame(
            resnet18_features(tensor), columns=photo_model["embedding_columns"]
        )
        use_specs = str(form.get("use_specs", "false")).lower() == "true"
        if not use_specs:
            pipe = photo_model["pipeline"]
            payload = probability_payload(pipe, pipe.predict_proba(embedding)[0], "ABO photo")
            payload["model"] = "ResNet-18 features + logistic regression"
            payload["cv_macro_f1"] = float(photo_model["cv_macro_f1"])
            return JSONResponse(payload)

        data = {
            "width_cm": form.get("width_cm"),
            "depth_cm": form.get("depth_cm"),
            "height_cm": form.get("height_cm"),
            "weight_kg": form.get("weight_kg"),
            "materials": form.getlist("materials"),
            "main_color": form.get("main_color", "other"),
        }
        specs = size_features(pd.DataFrame([specification_row(data, include_price=False, include_weight=True)]))
        specs["log_weight_kg"] = np.log(specs["weight_kg"])
        specs["log_kg_per_m3"] = np.log(specs["weight_kg"] / specs["volume_m3"])
        inputs = pd.concat([specs.reset_index(drop=True), embedding], axis=1)
        pipe = combined["pipeline"]
        payload = probability_payload(pipe, pipe.predict_proba(inputs)[0], "ABO photo + specifications")
        payload["model"] = "ResNet-18 features + structured features"
        payload["cv_macro_f1"] = float(combined["cv_macro_f1"])
        return JSONResponse(payload)
    except (ValueError, KeyError, Image.UnidentifiedImageError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    except Exception as exc:
        return JSONResponse({"error": f"Prediction failed: {exc}"}, status_code=500)


async def frontend(request: Request):
    # Serve top-level files copied from frontend/public (e.g. the hero photo); anything else gets the app.
    requested = (DIST / request.path_params.get("path", "")).resolve()
    if requested.is_file() and requested.is_relative_to(DIST.resolve()):
        return FileResponse(requested)
    if not (DIST / "index.html").exists():
        return JSONResponse(
            {"error": "Frontend is not built. Run `cd frontend && npm install && npm run build`."},
            status_code=503,
        )
    return FileResponse(DIST / "index.html")


routes = [
    Route("/api/health", health),
    Route("/api/predict/specs", predict_specs, methods=["POST"]),
    Route("/api/predict/photo", predict_photo, methods=["POST"]),
]
if (DIST / "assets").exists():
    routes.append(Mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets"))
routes.append(Route("/{path:path}", frontend))

app = Starlette(
    debug=False,
    routes=routes,
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
