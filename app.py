"""
Interactive SAM2 segmentation app.

Workflow: upload an image -> click a point on it -> SAM2 segments the
clicked object -> mask is overlaid on the original image.

Reuses the same SAM2 initialization / inference calls (build_sam2,
SAM2ImagePredictor, predictor.predict(point_coords=..., point_labels=...))
as the original notebook-based pipeline; only the coordinate source changed
from a hardcoded array to a click captured via the `streamlit-image-coordinates`
component.
"""

import os
import urllib.request
from io import BytesIO

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw

from streamlit_image_coordinates import streamlit_image_coordinates

SAM2_CHECKPOINT = "models/sam2.1_hiera_small.pt"
SAM2_CHECKPOINT_URL = (
    "https://huggingface.co/facebook/sam2.1-hiera-small/resolve/main/"
    "sam2.1_hiera_small.pt"
)

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
DISPLAY_WIDTH = 700  # fixed width (px) the image is shown at in the UI

POINT_PALETTE = ["#7C9BFF", "#FBBF24", "#34D399", "#F472B6", "#C084FC", "#F87171"]
DEFAULT_MASK_COLOR = "#7C9BFF"

st.set_page_config(page_title="SAM2 Interactive Segmentation", page_icon="🎯", layout="wide")


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(1200px 600px at 10% -10%, #1b1f3b 0%, rgba(27,31,59,0) 60%),
                        radial-gradient(1000px 500px at 110% 10%, #3a1530 0%, rgba(58,21,48,0) 55%),
                        #0e1117;
            color: #e5e7eb;
        }
        .stApp, .stApp p, .stApp span, .stApp label, .stApp li,
        .stMarkdown, .stCaption, [data-testid="stMarkdownContainer"] {
            color: #e5e7eb;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #f3f4f6 !important;
        }
        .app-title {
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0.1rem;
            background: linear-gradient(90deg, #7C9BFF, #B794F6 60%, #F472B6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .app-subtitle {
            color: #9ca3af;
            font-size: 0.98rem;
            margin-bottom: 1.4rem;
        }
        .panel {
            background: #171b2b;
            border: 1px solid #262b3d;
            border-radius: 16px;
            padding: 1.1rem 1.2rem 1.3rem 1.2rem;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
        }
        .panel-title {
            font-weight: 700;
            font-size: 1.02rem;
            color: #f3f4f6;
            display: flex;
            align-items: center;
            gap: 0.45rem;
            margin-bottom: 0.6rem;
        }
        .step-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 22px;
            height: 22px;
            border-radius: 999px;
            background: linear-gradient(135deg, #7C9BFF, #B794F6);
            color: #0e1117;
            font-size: 0.75rem;
            font-weight: 700;
        }
        div[data-testid="stFileUploaderDropzone"] {
            border-radius: 14px !important;
            background: #12141f !important;
            border: 1px dashed #33384f !important;
        }
        section[data-testid="stFileUploader"] label,
        section[data-testid="stFileUploader"] small,
        section[data-testid="stFileUploader"] span {
            color: #d1d5db !important;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 10px;
            font-weight: 600;
            background: #1c2033;
            color: #e5e7eb;
            border: 1px solid #303650;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: #7C9BFF;
            color: #7C9BFF;
        }
        section[data-testid="stSidebar"] {
            background: #12141f;
            border-right: 1px solid #262b3d;
        }
        section[data-testid="stSidebar"] * {
            color: #e5e7eb !important;
        }
        [data-testid="stMetric"] {
            background: #12141f;
            border: 1px solid #262b3d;
            border-radius: 12px;
            padding: 0.6rem 0.8rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.4rem;
            color: #f3f4f6;
        }
        [data-testid="stMetricLabel"] {
            color: #9ca3af;
        }
        .stAlert {
            background: #12141f !important;
            border: 1px solid #262b3d;
            color: #e5e7eb !important;
        }
        div[data-baseweb="slider"] div[role="slider"] {
            background-color: #7C9BFF !important;
        }
        input[type="color"] {
            border-radius: 8px;
            border: 1px solid #303650 !important;
        }
        hr, [data-testid="stDivider"] {
            border-color: #262b3d !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def panel_start(title: str, badge: str):
    st.markdown(
        f'<div class="panel"><div class="panel-title">'
        f'<span class="step-badge">{badge}</span>{title}</div>',
        unsafe_allow_html=True,
    )


def panel_end():
    st.markdown("</div>", unsafe_allow_html=True)


def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# Model loading (cached so SAM2 is loaded once, not on every click/rerun)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading SAM2 model...")
def load_model():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not os.path.exists(SAM2_CHECKPOINT):
        os.makedirs(os.path.dirname(SAM2_CHECKPOINT), exist_ok=True)
        urllib.request.urlretrieve(SAM2_CHECKPOINT_URL, SAM2_CHECKPOINT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor, device


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def load_image(uploaded_file) -> Image.Image:
    """Load an uploaded file into a PIL RGB image."""
    image = Image.open(uploaded_file)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def get_display_size(image: Image.Image, target_width: int = DISPLAY_WIDTH):
    """(width, height) the image is shown at in the browser, preserving aspect ratio."""
    width, height = image.size
    if width <= target_width:
        return width, height
    scale = target_width / width
    return target_width, max(1, round(height * scale))


def to_original_coords(x_disp, y_disp, image: Image.Image, display_size):
    """Map a click from displayed-image pixel space back to original image pixels."""
    disp_w, disp_h = display_size
    orig_w, orig_h = image.size
    x = int(round(x_disp * orig_w / disp_w))
    y = int(round(y_disp * orig_h / disp_h))
    # clamp in case of rounding at the image edge
    x = min(max(x, 0), orig_w - 1)
    y = min(max(y, 0), orig_h - 1)
    return x, y


def draw_points(image: Image.Image, points, display_size) -> Image.Image:
    """Return a resized copy of `image` with a numbered marker at each selected point."""
    preview = image.resize(display_size)
    draw = ImageDraw.Draw(preview, "RGBA")
    orig_w, orig_h = image.size
    disp_w, disp_h = display_size
    for i, (ox, oy) in enumerate(points):
        dx, dy = ox * disp_w / orig_w, oy * disp_h / orig_h
        color = POINT_PALETTE[i % len(POINT_PALETTE)]
        rgb = hex_to_rgb(color)
        r_outer = 11
        r_inner = 5
        # soft outer glow
        draw.ellipse(
            (dx - r_outer - 4, dy - r_outer - 4, dx + r_outer + 4, dy + r_outer + 4),
            fill=rgb + (45,),
        )
        # white halo for contrast against any background
        draw.ellipse((dx - r_outer, dy - r_outer, dx + r_outer, dy + r_outer), fill=(255, 255, 255, 235))
        # colored core
        draw.ellipse((dx - r_inner, dy - r_inner, dx + r_inner, dy + r_inner), fill=rgb + (255,))
    return preview


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def run_segmentation(predictor, image: Image.Image, points):
    """Run SAM2 with accumulated positive-point prompts, return the best mask + score."""
    image_np = np.array(image)
    point_coords = np.array(points)
    point_labels = np.ones(len(points), dtype=int)  # every click is a positive point

    predictor.set_image(image_np)
    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
    best_idx = int(np.argmax(scores))
    return masks[best_idx], float(scores[best_idx])


def mask_edge(mask: np.ndarray) -> np.ndarray:
    """Return a thin boolean edge/outline around the True region of `mask`."""
    m = mask
    edge = np.zeros_like(m, dtype=bool)
    edge[1:, :] |= m[1:, :] & ~m[:-1, :]
    edge[:-1, :] |= m[:-1, :] & ~m[1:, :]
    edge[:, 1:] |= m[:, 1:] & ~m[:, :-1]
    edge[:, :-1] |= m[:, :-1] & ~m[:, 1:]
    return edge


def create_mask_overlay(image: Image.Image, mask: np.ndarray, alpha: float, color=(124, 155, 255)):
    """Blend a mask over the image: soft color fill + a crisp outline for a cleaner look."""
    mask = mask.astype(bool)
    base = np.array(image).astype(np.float32)
    overlay = base.copy()

    color_arr = np.array(color, dtype=np.float32)
    overlay[mask] = (1 - alpha) * base[mask] + alpha * color_arr

    # crisp outline around the mask boundary, drawn at near-full opacity
    edge = mask_edge(mask)
    # thicken the outline slightly by also marking a 1px erosion ring
    outline_color = np.clip(color_arr * 0.55, 0, 255)
    overlay[edge] = 0.15 * overlay[edge] + 0.85 * outline_color

    return Image.fromarray(overlay.astype(np.uint8))


def st_image_compat(img, **kwargs):
    """st.image wrapper that works across Streamlit versions old and new."""
    try:
        st.image(img, use_container_width=True, **kwargs)
    except TypeError:
        st.image(img, use_column_width=True, **kwargs)


def st_wide(widget_fn, *args, **kwargs):
    """Call a Streamlit widget with use_container_width=True, falling back for old versions."""
    try:
        return widget_fn(*args, use_container_width=True, **kwargs)
    except TypeError:
        return widget_fn(*args, **kwargs)


def display_results(image: Image.Image, mask, score, alpha: float, n_points: int, mask_color_hex: str):
    result_img = create_mask_overlay(image, mask, alpha, hex_to_rgb(mask_color_hex))
    st_image_compat(result_img)

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Confidence", f"{score:.1%}")
    with m2:
        st.metric("Points used", n_points)

    buf = BytesIO()
    result_img.save(buf, format="PNG")
    st_wide(
        st.download_button,
        "⬇ Download result",
        data=buf.getvalue(),
        file_name="segmentation_result.png",
        mime="image/png",
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "points": [],          # list of (x, y) in ORIGINAL image coordinates
        "last_click_time": None,
        "mask": None,
        "score": None,
        "widget_key": 0,       # bumped to force-remount the click widget on reset
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def main():
    inject_css()

    st.markdown('<div class="app-title">🎯 SAM2 Interactive Segmentation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Upload an image, click on an object, get its mask — '
        "no coordinates typed by hand.</div>",
        unsafe_allow_html=True,
    )

    init_state()

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"])
    if uploaded_file is None:
        st.info("Upload an image to get started.")
        return

    try:
        image = load_image(uploaded_file)
    except Exception as e:
        st.error(f"Could not read this image: {e}")
        return

    try:
        predictor, device = load_model()
    except Exception as e:
        st.error(f"Could not load SAM2: {e}")
        return

    st.sidebar.header("⚙️ Options")
    multi_point = st.sidebar.checkbox("Multi-point mode (accumulate clicks)", value=False)
    alpha = st.sidebar.slider("Mask overlay opacity", 0.0, 1.0, 0.5, 0.05)
    mask_color_hex = st.sidebar.color_picker("Mask color", DEFAULT_MASK_COLOR)
    st.sidebar.divider()
    st.sidebar.caption(f"Running on: **{device.upper()}**")

    display_size = get_display_size(image)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        panel_start("Click an object", "1")
        preview = draw_points(image, st.session_state.points, display_size)
        click = streamlit_image_coordinates(
            preview,
            width=display_size[0],
            key=f"click_{st.session_state.widget_key}",
        )

        clear_col, run_col = st.columns(2)
        with clear_col:
            if st_wide(st.button, "🗑 Clear point(s)"):
                st.session_state.points = []
                st.session_state.mask = None
                st.session_state.score = None
                st.session_state.last_click_time = None
                st.session_state.widget_key += 1
                st.rerun()
        with run_col:
            manual_run = st_wide(st.button, "↻ Re-run segmentation")
        panel_end()

    # A click widget replays its last value on every rerun; only treat it as a
    # *new* click if its timestamp differs from the last one we processed.
    new_click = (
        click is not None
        and click.get("x") is not None
        and click.get("unix_time") != st.session_state.last_click_time
    )

    if new_click:
        st.session_state.last_click_time = click["unix_time"]
        try:
            orig_point = to_original_coords(click["x"], click["y"], image, display_size)
        except Exception as e:
            st.error(f"Invalid click coordinates: {e}")
            orig_point = None
        if orig_point:
            if multi_point:
                st.session_state.points.append(orig_point)
            else:
                st.session_state.points = [orig_point]

    if not st.session_state.points:
        with col2:
            panel_start("Segmentation result", "2")
            st.info("No point selected yet — click on the image to select an object.")
            panel_end()
        return

    if new_click or manual_run or st.session_state.mask is None:
        try:
            with st.spinner("Segmenting..."):
                mask, score = run_segmentation(predictor, image, st.session_state.points)
            st.session_state.mask = mask
            st.session_state.score = score
        except Exception as e:
            st.error(f"SAM2 inference failed: {e}")
            return

    with col2:
        panel_start("Segmentation result", "2")
        display_results(
            image,
            st.session_state.mask,
            st.session_state.score,
            alpha,
            len(st.session_state.points),
            mask_color_hex,
        )
        panel_end()


if __name__ == "__main__":
    main()
