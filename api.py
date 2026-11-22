"""Local HTTP API for the React Final-project demo."""

from functools import lru_cache
from pathlib import Path
import io
import os

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
from furniture.photos import encoder_features
from furniture.price import app_inputs, check_name

MATERIAL_NAMES = ["chipboard", "mdf", "fibreboard", "solid_wood", "plywood", "veneer", "metal", "plastic",
                  "glass", "mirror", "fabric", "leather", "foam", "spring", "rattan", "stone"]


ROOT = Path(__file__).resolve().parent
DIST = ROOT / "frontend" / "dist"


# DINOv2 replaced ResNet-18 after the survey (final.ipynb 5.6: CV macro-F1 0.844 vs 0.766 photo only).
# PHOTO_ENCODER=resnet18 keeps the earlier ResNet-18 models: the free Render server (512 MB) runs out of
# memory with DINOv2, so the online demo uses them (Render sets RENDER=true); the local app uses DINOv2.
PHOTO_ENCODER = os.environ.get("PHOTO_ENCODER") or ("resnet18" if os.environ.get("RENDER") else "dinov2")
PHOTO_FILES = {"dinov2": ("abo_photo_dinov2", "abo_specs_photo_dinov2"), "resnet18": ("abo_photo", "abo_specs_photo")}
PHOTO_LABEL = {"dinov2": "DINOv2", "resnet18": "ResNet-18"}[PHOTO_ENCODER]


@lru_cache(maxsize=1)
def load_models():
    photo, combined = PHOTO_FILES[PHOTO_ENCODER]
    return (
        joblib.load(ROOT / "models" / "structured_best.joblib"),
        joblib.load(ROOT / "models" / f"{photo}.joblib"),
        joblib.load(ROOT / "models" / f"{combined}.joblib"),
        joblib.load(ROOT / "models" / "structured_noprice.joblib"),
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
        "materials": "|".join(validate_materials(data, MATERIAL_NAMES)),
        "main_color": str(data.get("main_color", "other")),
    }
    if include_price:
        row["price_kzt"] = validate_number(data, "price_kzt", 5_000, 10_000_000)
    else:
        row["price_kzt"] = np.nan        # not an input of the model used without a price
    if include_weight:
        row["weight_kg"] = validate_number(data, "weight_kg", 0.5, 500)
    return row


@lru_cache(maxsize=1)
def load_price_model():
    return joblib.load(ROOT / "models" / "price_best.joblib")


def validate_choice(data, name, allowed, default=None):
    value = data.get(name, default)
    if value not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(allowed)}.")
    return value


def validate_materials(data, allowed):
    materials = data.get("materials", [])
    if not isinstance(materials, list):
        raise ValueError("materials must be a list.")
    unknown = sorted(set(materials) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown material(s): {', '.join(unknown)}.")
    return materials


async def health(_request):
    try:
        load_models()
        load_price_model()
        return JSONResponse({"status": "ready", "models": 5, "photo_encoder": PHOTO_ENCODER})
    except FileNotFoundError:
        return JSONResponse({"status": "missing_models"}, status_code=503)


def price_estimate(entry, inputs):
    log_price = float(entry["pipeline"].predict(inputs)[0])
    low, high = entry["residual_q10"], entry["residual_q90"]
    return {
        "price_kzt": round(float(np.exp(log_price)), -2),
        "low_kzt": round(float(np.exp(log_price + low)), -2),
        "high_kzt": round(float(np.exp(log_price + high)), -2),
        "model": entry["name"],
        "cv_mae_kzt": float(entry["cv_mae_kzt"]),
        "cv_mape": float(entry["cv_mape"]),
    }


async def predict_price(request: Request):
    try:
        data = await request.json()
        saved = load_price_model()
        name = str(data.get("name", "") or "").strip()
        if len(name) > 200:
            raise ValueError("name must be at most 200 characters.")
        row = {
            "width_cm": validate_number(data, "width_cm", 10, 500),
            "depth_cm": validate_number(data, "depth_cm", 10, 500),
            "height_cm": validate_number(data, "height_cm", 10, 500),
            "furniture_type": validate_choice(data, "furniture_type", TYPES),
            "main_color": validate_choice(data, "main_color", saved["colours"], "other"),
            "materials": "|".join(validate_materials(data, saved["material_names"])),
            "name": name,
        }
        specs_entry, name_entry = saved["specs"], saved["with_name"]
        name_ok, name_note = check_name(name, name_entry["pipeline"], row["furniture_type"]) if name else (False, "")
        inputs = app_inputs(pd.DataFrame([row]), saved["materials"], name_entry.get("e5_columns") if name_ok else None)
        payload = {"specs_only": price_estimate(specs_entry, inputs), "with_name": None, "name_note": name_note}
        if name_ok:
            payload["with_name"] = price_estimate(name_entry, inputs)
        return JSONResponse(payload)
    except (ValueError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    except Exception as exc:
        return JSONResponse({"error": f"Prediction failed: {exc}"}, status_code=500)


async def predict_specs(request: Request):
    try:
        data = await request.json()
        has_price = data.get("price_kzt") not in (None, "")
        row = pd.DataFrame([specification_row(data, include_price=has_price)])
        structured, _, _, noprice = load_models()
        model = structured if has_price else noprice
        inputs = add_features(row, materials=model["materials"])[model["features"]]
        pipe = model["pipeline"]
        source = "mebel.kz specifications" if has_price else "mebel.kz specifications without price"
        payload = probability_payload(pipe, pipe.predict_proba(inputs)[0], source)
        payload["model"] = model["name"]
        payload["cv_macro_f1"] = float(model["cv_macro_f1"])
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

        _, photo_model, combined, _ = load_models()
        embedding = pd.DataFrame(
            encoder_features(photo_model.get("encoder", "resnet18"), tensor), columns=photo_model["embedding_columns"]
        )
        use_specs = str(form.get("use_specs", "false")).lower() == "true"
        if not use_specs:
            pipe = photo_model["pipeline"]
            payload = probability_payload(pipe, pipe.predict_proba(embedding)[0], "ABO photo")
            payload["model"] = f"{PHOTO_LABEL} features + logistic regression"
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
        payload["model"] = f"{PHOTO_LABEL} features + specifications, logistic regression"
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
    Route("/api/predict/price", predict_price, methods=["POST"]),
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
