"""
Visual Product Search Engine — Streamlit App
Fulfills final project requirements with explicit User Confirmation step.
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
    Blip2Processor,
    Blip2ForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
    BitsAndBytesConfig,
)

# ==========================================
# 1. CONFIGURATION & STATE
# ==========================================
ALPHA = 0.5                       
INDEX_DIR  = "offline_index"      
DATASET_ROOT = ""                 

device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Visual Product Search", layout="wide")

# Initialize Session State
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# ==========================================
# 2. MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Loading Clothing Detector …")
def load_yolo():
    return YOLO("best.pt")

@st.cache_resource(show_spinner="Loading BLIP-2 (CPU Safe) …")
def load_blip2():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b"
    ).to(device)
    return processor, model

@st.cache_resource(show_spinner="Loading CLIP …")
def load_clip():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
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

# Classes produced by kesimeg/yolov8n-clothing-detection
ALLOWED_CLASSES = [
    "Clothing",
    "Accessories",
    "Shoes",
    "Bags",
]

def get_region_type(y1, y2, img_height):
    """Map a bounding box to Upper Body / Lower Body / Full Body."""
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
    Returns a list of detection dicts.
    """
    results = yolo_model(img_rgb, verbose=False)[0]
    height = img_rgb.shape[0]
    detections = []

    for box in results.boxes:
        conf = float(box.conf[0])

        if conf < 0.4:
            continue

        cls_id = int(box.cls[0])
        label = yolo_model.names[cls_id]

        # Only keep fashion-relevant classes
        if label not in ALLOWED_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img_rgb[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        region_type = get_region_type(y1, y2, height)

        detections.append({
            "label": label,
            "confidence": conf,
            "bbox": (x1, y1, x2, y2),
            "crop": crop,
            "region_type": region_type,
        })

    return detections


def run_retrieval(pil_img, alpha, top_k):
    """Core retrieval pipeline: Caption -> Embed -> FAISS Search"""
    blip_proc, blip_mdl = load_blip2()
    clip_proc, clip_mdl = load_clip()
    faiss_index, metadata = load_index(INDEX_DIR)
    
    # 1. Caption Generation
    inputs = blip_proc(pil_img, return_tensors="pt").to(device, torch.float16)
    ids = blip_mdl.generate(**inputs, max_new_tokens=20)
    caption = blip_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    
    # 2. CLIP Fused Embedding
    inputs = clip_proc(text=[caption], images=pil_img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_mdl(**inputs)
        
    img_emb = outputs.image_embeds[0].cpu().numpy()
    txt_emb = outputs.text_embeds[0].cpu().numpy()
    img_emb /= np.linalg.norm(img_emb)
    txt_emb /= np.linalg.norm(txt_emb)
    
    fused_emb = alpha * img_emb + (1 - alpha) * txt_emb
    fused_emb /= np.linalg.norm(fused_emb)
    
    # 3. FAISS Retrieval
    qv = fused_emb.reshape(1, -1).astype("float32")
    distances, indices = faiss_index.search(qv, top_k)
    
    # 4. Format Results
    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1: 
            continue
        meta = metadata[idx]
        results.append({
            "image_path": meta.get("image_path", meta.get("path", "")),
            "caption": meta.get("caption", "No caption"),
            "item_id": meta.get("item_id", "Unknown"),
            "score": float(distances[0][rank])
        })
        
    return caption, results

# ==========================================
# 4. STREAMLIT UI FLOW
# ==========================================
st.title("🔍 Visual Product Search Engine")
st.caption("Ablation Config: YOLOv8 → BLIP-2 Caption → CLIP Fused Embedding (α=0.5) → FAISS Retrieval")

with st.sidebar:
    st.header("⚙️ Settings")
    top_k_slider = st.slider("Top-K results", 1, 20, 5)

# --- STEP 1: Upload Image ---
uploaded = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    # Reset state if a new image is uploaded
    if st.session_state.last_uploaded != uploaded.name:
        st.session_state.last_uploaded = uploaded.name

    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- STEP 2: Run YOLO Detection ---
    yolo = load_yolo()
    print(yolo.names)  # prints {0: 'Clothing', 1: 'Shoes', ...}
    
    detections = detect_clothing_regions(yolo, img_rgb)

    # --- STEP 3: Show All Detected Regions ---
    st.markdown("### Detected Clothing Regions")

    disp = img_rgb.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]

        cv2.rectangle(
            disp,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            disp,
            f'{det["label"]} | {det["region_type"]} {det["confidence"]:.2f}',
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    st.image(
        disp,
        caption="Detected Clothing Regions",
        use_container_width=True
    )

    # --- STEP 4: Multi-Choice Selection UI ---
    st.markdown("### Select Search Target")

    if len(detections) == 0:
        st.warning("No clothing items detected. Falling back to full image for search.")
        selected_crop = img_rgb

    else:
        options = []

        for i, det in enumerate(detections):
            option_text = (
                f'{i + 1}. '
                f'{det["label"]} — {det["region_type"]} '
                f'(conf={det["confidence"]:.2f})'
            )
            options.append(option_text)

        # Add full image as fallback option
        options.append(f"{len(detections) + 1}. Full Image (no crop)")

        selected_option = st.radio(
            "Choose clothing item to search:",
            options
        )

        selected_idx = options.index(selected_option)

        if selected_idx < len(detections):
            selected_crop = detections[selected_idx]["crop"]
            st.image(
                selected_crop,
                caption=(
                    f"Selected: {detections[selected_idx]['label']} "
                    f"— {detections[selected_idx]['region_type']}"
                ),
                use_container_width=True
            )
        else:
            selected_crop = img_rgb
            st.image(
                selected_crop,
                caption="Selected Search Region: Full Image",
                use_container_width=True
            )

    # --- STEP 5: Run Retrieval Pipeline & Display Results ---
    st.markdown("---")

    if st.button("🔍 Search Similar Products", use_container_width=True):
        pil_target = Image.fromarray(selected_crop)

        with st.spinner("Processing semantics and searching vector database..."):
            try:
                gen_caption, top_k_results = run_retrieval(pil_target, ALPHA, top_k_slider)
                
                st.success(f"**Query Caption Generated:** {gen_caption}")
                st.subheader(f"Top-{top_k_slider} Retrieved Products")
                
                # Display Results in a dynamic grid
                cols = st.columns(min(top_k_slider, 5))
                for rank, res in enumerate(top_k_results):
                    col = cols[rank % len(cols)]
                    with col:
                        img_path = res["image_path"]
                        if DATASET_ROOT and not img_path.startswith(DATASET_ROOT):
                            img_path = os.path.join(DATASET_ROOT, img_path)
                            
                        if os.path.exists(img_path):
                            res_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                            st.image(res_img, use_container_width=True)
                        else:
                            st.warning("Image not found on disk")
                            
                        st.markdown(f"**Rank {rank+1}**")
                        st.markdown(f"**Score:** `{res['score']:.4f}`")
                        st.markdown(f"**ID:** `{res['item_id']}`")
                        st.caption(f"_{res['caption']}_")
                        
            except Exception as e:
                st.error(f"Pipeline Error: {e}")
                st.info("Ensure offline_index/ contains both gallery_index.faiss and gallery_metadata.json")
