# Pixelate Everything

A team project that converts uploaded images into pixel art through a local Flask interface. The team demo includes raw, region-based, and palette-compressed rendering, along with an image-based recommendation for block size and palette size.

## Stack

Python, Flask, OpenCV, NumPy, scikit-learn, and Joblib.

## Run locally

Use Python 3.12 and install the versions in the requirements file to match the saved recommendation model.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m flask --app app run --port 5055
```

Open http://127.0.0.1:5055, upload an image, and choose a rendering mode. The local demo writes uploads and results to ignored directories under `static/`.

## Team and portfolio scope

The image-processing pipeline and recommendation model are team deliverables. This portfolio edition packages the team's existing Web demo for local reproduction. Individual algorithm ownership is not attributed here.
