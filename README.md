# SAM2 Interactive Segmentation

Click-to-segment: upload an image, click on an object, and [SAM2](https://github.com/facebookresearch/sam2) (Meta's Segment Anything Model 2) segments it in real time — no manual coordinates, no bounding boxes, just a click.

Built with [Streamlit](https://streamlit.io/) and [`streamlit-image-coordinates`](https://github.com/blackary/streamlit-image-coordinates).

## Features

- **Click-to-segment** — click anywhere on the image and SAM2 returns a mask for the object under your cursor
- **Multi-point mode** — accumulate several clicks as positive prompts to refine a tricky mask
- **Adjustable overlay** — control mask opacity and color from the sidebar
- **Confidence score** — see SAM2's confidence for the returned mask
- **Download result** — export the segmented image as a PNG

## Demo

1. Upload an image (PNG, JPG, JPEG, BMP, or WebP)
2. Click on the object you want to segment
3. SAM2 returns a mask, overlaid on the original image with a colored fill and outline
4. Optionally add more points (multi-point mode) or re-run segmentation
5. Download the result

## Requirements

- Python 3.10+
- A GPU is recommended but not required — the app falls back to CPU automatically (inference will be slower)

## Installation

```bash
git clone https://github.com/usf132/sam2-interactive-segmentation.git
cd sam2-interactive-segmentation
pip install -r requirements.txt
```

The SAM2 checkpoint (`sam2.1_hiera_small.pt`, ~185 MB) downloads automatically on first run and is cached in `models/`.

## Usage

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Project structure

```
.
├── app.py               # Streamlit app: UI, click handling, SAM2 inference
├── requirements.txt      # Python dependencies
├── test_click.py         # Minimal standalone test of the click component
└── README.md
```

## How it works

The app reuses SAM2's standard image-prediction API:

```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint_path, device=device))
predictor.set_image(image)
masks, scores, _ = predictor.predict(
    point_coords=points,
    point_labels=labels,   # every click is a positive point
    multimask_output=True,
)
```

The only custom part is the coordinate source: instead of a hardcoded array of points, clicks are captured in the browser via `streamlit-image-coordinates` and mapped back from the displayed (resized) image to the original image's pixel space before being passed to SAM2.

## Deploying to Streamlit Community Cloud

This app runs on [Streamlit Community Cloud](https://streamlit.io/cloud) out of the box. A couple of notes:

- The free tier has limited RAM; SAM2 + PyTorch can be tight. If the app stalls or crashes while "Loading SAM2 model...", that's a resource limit rather than a bug — consider the `sam2.1_hiera_tiny` checkpoint for lower memory use, or deploy somewhere with more headroom.
- After editing `requirements.txt` or `app.py`, use **Manage app → Reboot app** to force a clean redeploy.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Error installing requirements` | `requirements.txt` has an invalid or unreachable entry — check the deploy logs for the specific package |
| App hangs on "Loading SAM2 model..." | Downloading the checkpoint on a slow connection, or the host is low on memory |
| `SAM2 inference failed` shown in the UI | Usually a corrupt/partial checkpoint download — delete the `models/` folder and restart to re-download |
| Clicks don't register | Make sure you're clicking directly on the image preview, not the surrounding panel |

## License

See [LICENSE](LICENSE).
