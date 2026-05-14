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

# Initialize Session State for User Confirmation
if "retrieval_target" not in st.session_state:
    st.session_state.retrieval_target = None
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# ==========================================
# 2. MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Loading YOLOv8 …")
def load_yolo():
    return YOLO("yolov8n.pt")

# @st.cache_resource(show_spinner="Loading BLIP-2 (8-bit) …")
# def load_blip2():
#     processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
#     quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
#     model = Blip2ForConditionalGeneration.from_pretrained(
#         "Salesforce/blip2-opt-2.7b",
#         quantization_config=quant_cfg,
#         device_map="auto",
#     )
#     return processor, model
@st.cache_resource(show_spinner="Loading BLIP-2 (CPU Safe) …")
def load_blip2():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    # Removed bitsandbytes and device_map for CPU compatibility
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b"
    ).to(device) # This forces it onto your CPU if no CUDA is found
    
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
def detect_and_crop(yolo_model, img_rgb):
    """Run YOLOv8; return the most-confident crop and original image."""
    results = yolo_model(img_rgb, verbose=False)[0]
    best_crop, best_conf = None, 0.0
    best_box = None
    
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf > best_conf and conf > 0.5:
            best_conf = conf
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            best_crop = img_rgb[y1:y2, x1:x2]
            best_box = (x1, y1, x2, y2)
            
    if best_crop is None or best_crop.size == 0:
        best_crop = img_rgb
        best_box = None
        
    return best_crop, best_box, best_conf

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
            "image_path": meta.get("image_path", meta.get("path", "")), # Fallback if metadata keys vary slightly
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
        st.session_state.retrieval_target = None
        st.session_state.last_uploaded = uploaded.name

    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- STEP 2: Run YOLO Detection ---
    yolo = load_yolo()
    crop_rgb, bbox, conf = detect_and_crop(yolo, img_rgb)

    st.markdown("### Product Localization")
    col_orig, col_crop = st.columns(2)
    
    with col_orig:
        disp = img_rgb.copy()
        if bbox:
            cv2.rectangle(disp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
        st.image(disp, caption="Original Image (YOLO Bounding Box)", use_container_width=True)
        
    with col_crop:
        st.image(crop_rgb, caption=f"YOLO Cropped Product (Confidence: {conf:.2f})" if bbox else "No product detected. Falling back to full image.", use_container_width=True)

    # --- STEP 3: User Confirmation ---
    if st.session_state.retrieval_target is None:
        st.markdown("### Confirm Search Target")
        st.info("Do you want to search using the cropped product, or the full original image?")
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("✅ Confirm Crop", use_container_width=True):
                st.session_state.retrieval_target = "crop"
                st.rerun()
        with btn_col2:
            if st.button("🖼️ Use Full Image", use_container_width=True):
                st.session_state.retrieval_target = "full"
                st.rerun()

    # --- STEP 4 & 5: Run Retrieval Pipeline & Display Results ---
    if st.session_state.retrieval_target is not None:
        st.markdown("---")
        
        # Determine which image the user selected
        final_target_rgb = crop_rgb if st.session_state.retrieval_target == "crop" else img_rgb
        pil_target = Image.fromarray(final_target_rgb)
        
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
                        # Attempt to load the image from disk
                        img_path = res["image_path"]
                        if DATASET_ROOT and not img_path.startswith(DATASET_ROOT):
                            img_path = os.path.join(DATASET_ROOT, img_path)
                            
                        if os.path.exists(img_path):
                            res_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                            st.image(res_img, use_container_width=True)
                        else:
                            st.warning("Image not found on disk")
                            
                        # Display requested metadata
                        st.markdown(f"**Rank {rank+1}**")
                        st.markdown(f"**Score:** `{res['score']:.4f}`")
                        st.markdown(f"**ID:** `{res['item_id']}`")
                        st.caption(f"_{res['caption']}_")
                        
            except Exception as e:
                st.error(f"Pipeline Error: {e}")
                st.info("Ensure offline_index/ contains both gallery_index.faiss and gallery_metadata.json")
