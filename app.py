from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import cv2
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
PIXELATE_DIR = ROOT / "pixelate"
MODEL_PATH = ROOT / "ai_recommender" / "recommender_v4.joblib.gz"
sys.path.insert(0, str(PIXELATE_DIR))
sys.path.insert(0, str(ROOT / "ai_recommender"))

from main import choose_block_size, read_image, render_mode  # noqa: E402
from extract_features import extract_features  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
ALLOWED = {"png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"}


def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


def args_for(mode: str, detail: str, palette_size: int, block_size: int | None) -> SimpleNamespace:
    values = {
        "mode": mode, "block_size": block_size, "grid_candidates": 12,
        "preserve_size": True,
        "detail_mode": detail, "palette_size": palette_size, "random_state": 42,
        "edge_threshold": 80, "min_region_size": 24, "segmentation_blur_ksize": 5,
        "edge_dilate_iter": 1, "area_cap": 2048, "weight_mode": "uniform",
        "exclude_outline": False, "outline_color_bin_size": 16,
        "outline_minimum_edge_pixels": 16, "outline_minimum_support_fraction": .03,
        "outline_color_tolerance": 12., "outline_edge_coverage_threshold": .05,
        "outline_fraction_threshold": .60, "outline_neighbor_radius": 2,
        "outline_spatial_penalty": .35, "overcluster_factor": 2.,
        "gradient_tolerance": 1., "solid_max_distance": 4., "minimum_dither_span": 6.,
        "minimum_linearity": .90, "maximum_reconstruction_rmse": 3.,
        "maximum_point_error": 6., "minimum_endpoint_support_fraction": .05,
        "endpoint_quantile": .05, "forced_solid_penalty": 4., "dither_penalty": 0.,
        "dither_method": "ordered", "matrix_size": 8, "extreme_ratio_threshold": .08,
        "disable_palette_usage_enforcement": False,
    }
    return SimpleNamespace(**values)


def resize(path: Path, max_side: int) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file could not be read as an image.")
    height, width = image.shape[:2]
    if max(height, width) <= max_side:
        return
    scale = max_side / max(height, width)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), resized)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/recommend")
def recommend():
    image_file = request.files.get("image")
    if not image_file or not image_file.filename:
        return jsonify(error="Please upload an image before requesting recommendations."), 400
    data = np.frombuffer(image_file.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify(error="The uploaded file could not be read as an image."), 400
    bundle = joblib.load(MODEL_PATH)
    block, palette = bundle["model"].predict(extract_features(image).reshape(1, -1))[0]
    return jsonify(block_size=int(np.clip(round(block), 1, 128)), palette_size=int(np.clip(round(palette), 6, 64)))


@app.post("/process")
def process():
    image_file = request.files.get("image")
    if not image_file or not image_file.filename or not allowed(image_file.filename):
        return render_template("index.html", error="Upload a supported image file (PNG, JPG, JPEG, BMP, WebP, TIF, or TIFF)."), 400
    try:
        mode = request.form.get("mode", "compressed")
        detail = request.form.get("detail", "balanced")
        palette = int(request.form.get("palette_size", 32))
        block_value = request.form.get("block_size", "")
        block = int(block_value) if block_value else None
        max_side = int(request.form.get("max_side", 1800))
        if mode not in {"compressed", "region", "raw"} or detail not in {"simplify", "balanced", "preserve"}:
            raise ValueError("Invalid processing setting.")
        if not 6 <= palette <= 64 or block is not None and not 1 <= block <= 128 or not 256 <= max_side <= 4096:
            raise ValueError("One or more settings are outside the allowed range.")
        run_id = uuid.uuid4().hex
        upload_dir = ROOT / "static" / "uploads" / run_id
        output_dir = ROOT / "static" / "outputs" / run_id
        upload_dir.mkdir(parents=True); output_dir.mkdir(parents=True)
        input_name = secure_filename(image_file.filename)
        input_path = upload_dir / input_name
        image_file.save(input_path)
        resize(input_path, max_side)
        image = read_image(input_path)
        selected_block = choose_block_size(image, args_for(mode, detail, palette, block))
        started = time.perf_counter()
        result, info = render_mode(image, selected_block, mode, args_for(mode, detail, palette, selected_block))
        output_name = "pixelated.png"
        cv2.imwrite(str(output_dir / output_name), result)
        return render_template("index.html", result=True, original=url_for("static", filename=f"uploads/{run_id}/{input_name}"), output=url_for("static", filename=f"outputs/{run_id}/{output_name}"), download=url_for("static", filename=f"outputs/{run_id}/{output_name}"), meta={"mode": mode, "detail": detail, "palette": palette, "block": selected_block, "runtime": round(time.perf_counter() - started, 2), "dither_groups": info.get("dither_groups", 0)})
    except Exception as error:
        return render_template("index.html", error=str(error)), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055, debug=False)
