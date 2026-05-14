
"""
Visual Product Search Engine — Streamlit App
CPU-safe version: uses BLIP base captioning instead of BLIP-2 (MX570A compatible)
"""

import os
import json
import torch
import faiss
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
)

# ==========================================
# 1. CONFIGURATION & STATE
# ==========================================
ALPHA        = 0.5
INDEX_DIR    = "offline_index"
DATASET_ROOT = ""

# Force CPU — stable on MX570A (BLIP-2 would crash with limited VRAM)
device = "cpu"

st.set_page_config(page_title="Visual Product Search", layout="wide")

if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# ==========================================
# 2. MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Loading Clothing Detector …")
def load_yolo():
    return YOLO("best.pt")


@st.cache_resource(show_spinner="Loading BLIP captioner …")
def load_blip():
    """
    BLIP base captioning model — much lighter than BLIP-2.
    ~900 MB vs ~6 GB. Works fine on CPU for demo.
    """
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    return processor, model


@st.cache_resource(show_spinner="Loading CLIP …")
def load_clip():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    return processor, model


@st.cache_resource(show_spinner="Loading FAISS index …")
def load_index(index_dir):
    index = faiss.read_index(os.path.join(index_dir, "gallery_index.faiss"))
    with open(os.path.join(index_dir, "gallery_metadata.json"), "r") as f:
        metadata = json.load(f)
    return index, metadata

# ==========================================
# 3. PIPELINE HELPERS
# ==========================================

# Lowercase — matches kesimeg/yolov8n-clothing-detection class names
ALLOWED_CLASSES = [
    "clothing",
    "accessories",
    "shoes",
    "bags",
]


def get_region_type(y1, y2, img_height):
    """Map bounding box centroid to Upper / Lower / Full Body."""
    center_y = (y1 + y2) / 2
    if center_y < img_height * 0.45:
        return "Upper Body"
    elif center_y > img_height * 0.60:
        return "Lower Body"
    else:
        return "Full Body"


def detect_clothing_regions(yolo_model, img_rgb):
    """
    Detect ALL clothing regions using the fashion-aware YOLO model.
    Filters to ALLOWED_CLASSES only and tags each detection with a
    body-region label (Upper Body / Lower Body / Full Body).
    """
    results = yolo_model(img_rgb, verbose=False)[0]
    height  = img_rgb.shape[0]
    detections = []

    for box in results.boxes:
        conf   = float(box.conf[0])
        if conf < 0.4:
            continue

        cls_id = int(box.cls[0])
        label  = yolo_model.names[cls_id].lower()   # lowercase for safe matching

        if label not in ALLOWED_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        region_type = get_region_type(y1, y2, height)
        detections.append({
            "label"      : label.capitalize(),
            "confidence" : conf,
            "bbox"       : (x1, y1, x2, y2),
            "crop"       : crop,
            "region_type": region_type,
        })

    return detections


def run_retrieval(pil_img, alpha, top_k):
    """Caption → fused CLIP embedding → FAISS search."""
    blip_proc, blip_mdl   = load_blip()
    clip_proc, clip_mdl   = load_clip()
    faiss_index, metadata = load_index(INDEX_DIR)

    # 1. Caption (BLIP base — CPU friendly)
    inputs = blip_proc(pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = blip_mdl.generate(**inputs, max_new_tokens=20)
    caption = blip_proc.decode(ids[0], skip_special_tokens=True).strip()

    # 2. CLIP fused embedding
    inputs = clip_proc(
        text=[caption], images=pil_img, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        outputs = clip_mdl(**inputs)

    img_emb = outputs.image_embeds[0].cpu().numpy()
    txt_emb = outputs.text_embeds[0].cpu().numpy()
    img_emb /= np.linalg.norm(img_emb)
    txt_emb /= np.linalg.norm(txt_emb)

    fused_emb  = alpha * img_emb + (1 - alpha) * txt_emb
    fused_emb /= np.linalg.norm(fused_emb)

    # 3. FAISS retrieval
    qv = fused_emb.reshape(1, -1).astype("float32")
    distances, indices = faiss_index.search(qv, top_k)

    # 4. Format results
    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        meta = metadata[idx]
        results.append({
            "image_path" : meta.get("image_path", meta.get("path", "")),
            "caption"    : meta.get("caption", "No caption"),
            "item_id"    : meta.get("item_id", "Unknown"),
            "region_type": meta.get("region_type", ""),
            "label"      : meta.get("label", ""),
            "score"      : float(distances[0][rank]),
        })

    return caption, results

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.title("🔍 Visual Product Search Engine")
st.caption(
    "Pipeline: YOLOv8 (clothing) → BLIP caption → "
    "CLIP fused embedding (α=0.5) → FAISS retrieval"
)

with st.sidebar:
    st.header("⚙️ Settings")
    top_k_slider = st.slider("Top-K results", 1, 20, 5)
    st.info(
        "Running in **CPU mode** for stability.\n\n"
        "Captioning may take 10–20 s per query."
    )

# --- STEP 1: Upload ---
uploaded = st.file_uploader(
    "Upload a query image", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded is not None:
    if st.session_state.last_uploaded != uploaded.name:
        st.session_state.last_uploaded = uploaded.name

    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- STEP 2: YOLO detection ---
    yolo       = load_yolo()
    detections = detect_clothing_regions(yolo, img_rgb)

    # --- STEP 3: Annotated image ---
    st.markdown("### Detected Clothing Regions")
    disp = img_rgb.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            disp,
            f'{det["label"]} | {det["region_type"]} {det["confidence"]:.2f}',
            (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    st.image(disp, caption="Detected Clothing Regions", use_container_width=True)

    # --- STEP 4: User selects which crop to search ---
    st.markdown("### Select Search Target")

    if len(detections) == 0:
        st.warning("No clothing items detected. Using full image for search.")
        selected_crop = img_rgb
    else:
        options = [
            f"{i + 1}. {det['label']} — {det['region_type']} "
            f"(conf={det['confidence']:.2f})"
            for i, det in enumerate(detections)
        ]
        options.append(f"{len(detections) + 1}. Full Image (no crop)")

        selected_option = st.radio("Choose clothing item to search:", options)
        selected_idx    = options.index(selected_option)

        if selected_idx < len(detections):
            selected_crop = detections[selected_idx]["crop"]
            st.image(
                selected_crop,
                caption=(
                    f"Selected: {detections[selected_idx]['label']} "
                    f"— {detections[selected_idx]['region_type']}"
                ),
                use_container_width=True,
            )
        else:
            selected_crop = img_rgb
            st.image(
                selected_crop,
                caption="Selected: Full Image",
                use_container_width=True,
            )

    # --- STEP 5: Search ---
    st.markdown("---")

    if st.button("🔍 Search Similar Products", use_container_width=True):
        pil_target = Image.fromarray(selected_crop)

        with st.spinner("Captioning image and searching … (CPU — ~15 s)"):
            try:
                gen_caption, top_k_results = run_retrieval(
                    pil_target, ALPHA, top_k_slider
                )

                st.success(f"**Query Caption:** {gen_caption}")
                st.subheader(f"Top-{top_k_slider} Retrieved Products")

                cols = st.columns(min(top_k_slider, 5))
                for rank, res in enumerate(top_k_results):
                    col = cols[rank % len(cols)]
                    with col:
                        img_path = res["image_path"]
                        if DATASET_ROOT and not img_path.startswith(DATASET_ROOT):
                            img_path = os.path.join(DATASET_ROOT, img_path)

                        if os.path.exists(img_path):
                            res_img = cv2.cvtColor(
                                cv2.imread(img_path), cv2.COLOR_BGR2RGB
                            )
                            st.image(res_img, use_container_width=True)
                        else:
                            st.warning("Image not found on disk")

                        st.markdown(f"**Rank {rank + 1}**")
                        st.markdown(f"**Score:** `{res['score']:.4f}`")
                        st.markdown(f"**ID:** `{res['item_id']}`")
                        if res["region_type"]:
                            st.markdown(f"**Region:** {res['region_type']}")
                        st.caption(f"_{res['caption']}_")

            except Exception as e:
                st.error(f"Pipeline Error: {e}")
                st.info(
                    "Check that `offline_index/` contains "
                    "`gallery_index.faiss` and `gallery_metadata.json`."
                )

