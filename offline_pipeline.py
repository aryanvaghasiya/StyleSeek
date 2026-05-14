!pip install -q ultralytics transformers faiss-cpu accelerate bitsandbytes

!wget -q https://huggingface.co/kesimeg/yolov8n-clothing-detection/resolve/main/best.pt \
     -O /kaggle/working/best.pt
print("Downloaded best.pt")

import os
import warnings
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

import transformers
transformers.logging.set_verbosity_error()

import cv2
import json
import torch
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
    BitsAndBytesConfig,
)

ALPHA = 0.5
CONF_THRESHOLD = 0.4

ALLOWED_CLASSES = ["clothing", "accessories", "shoes", "bags"]

MAX_IMAGES_TO_PROCESS = None
CHECKPOINT_INTERVAL = 500
BATCH_SIZE = 8

OUTPUT_DIR = "/kaggle/working/offline_index"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

possible_partition_paths = [
    "/kaggle/input/datasets/aryanvaghasiya/deepfashion-inshop/list_eval_partition.txt",
    "/kaggle/input/datasets/aryanvaghasiya/deepfashion-inshop/img/list_eval_partition.txt",
    "/kaggle/input/deepfashion-inshop/list_eval_partition.txt",
    "/kaggle/input/deepfashion-inshop/img/list_eval_partition.txt",
]

PARTITION_FILE = None
for p in possible_partition_paths:
    if os.path.exists(p):
        PARTITION_FILE = p
        break

if not PARTITION_FILE:
    raise FileNotFoundError("Could not find list_eval_partition.txt")

DATASET_ROOT = os.path.dirname(PARTITION_FILE)
print(f"Partition file : {PARTITION_FILE}")
print(f"Dataset root   : {DATASET_ROOT}")

def get_gallery_items(partition_file, dataset_root, max_images=None):
    print("Parsing dataset split...")
    gallery_items = []

    with open(partition_file, "r") as f:
        lines = f.readlines()

    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        img_name, item_id, status = parts[0], parts[1], parts[2]

        if status != "gallery":
            continue

        candidates = [
            os.path.join(dataset_root, img_name),
            os.path.join(dataset_root, "img", img_name),
            os.path.join(dataset_root, img_name.replace("img/", "")),
            os.path.join(dataset_root, "img", img_name.replace("img/", "")),
            os.path.join(dataset_root, "img/img", img_name.replace("img/", "")),
        ]

        valid_path = next((p for p in candidates if os.path.exists(p)), None)
        if valid_path:
            gallery_items.append({
                "path": valid_path,
                "img_name": img_name,
                "item_id": item_id,
            })

        if max_images and len(gallery_items) >= max_images:
            break

    if not gallery_items:
        raise RuntimeError("No gallery images found. Check dataset paths.")

    print(f"Found {len(gallery_items)} gallery images.")
    return gallery_items

print("Loading clothing detector...")
yolo_model = YOLO("/kaggle/working/best.pt")
print(f"Classes: {yolo_model.names}")

print("Loading BLIP-2...")
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map={"": 0},
)

print("Loading CLIP...")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

print("All models loaded.\n")

def get_region_type(y1, y2, img_height):
    center_y = (y1 + y2) / 2
    if center_y < img_height * 0.45:
        return "Upper Body"
    elif center_y > img_height * 0.60:
        return "Lower Body"
    else:
        return "Full Body"


def generate_captions_batch(pil_images):
    inputs = blip_processor(
        images=pil_images, return_tensors="pt", padding=True
    ).to(device, torch.float16)
    with torch.no_grad():
        ids = blip_model.generate(**inputs, max_new_tokens=15)
    return [c.strip() for c in blip_processor.batch_decode(ids, skip_special_tokens=True)]


def generate_fused_embeddings_batch(pil_images, captions, alpha):
    inputs = clip_processor(
        text=captions, images=pil_images, return_tensors="pt", padding=True
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)

    img_embs = outputs.image_embeds.cpu().numpy()
    text_embs = outputs.text_embeds.cpu().numpy()

    img_embs /= np.linalg.norm(img_embs, axis=1, keepdims=True)
    text_embs /= np.linalg.norm(text_embs, axis=1, keepdims=True)

    fused = alpha * img_embs + (1 - alpha) * text_embs
    fused /= np.linalg.norm(fused, axis=1, keepdims=True)
    return fused

checkpoint_meta_file  = os.path.join(OUTPUT_DIR, "metadata_checkpoint.json")
checkpoint_embs_file  = os.path.join(OUTPUT_DIR, "embeddings_checkpoint.npy")
processed_paths_file  = os.path.join(OUTPUT_DIR, "processed_paths.json")

metadata_store   = []
embeddings_list  = []
processed_paths  = set()
global_object_id = 0

if (os.path.exists(checkpoint_meta_file)
        and os.path.exists(checkpoint_embs_file)
        and os.path.exists(processed_paths_file)):

    print("Resuming from checkpoint...")
    with open(checkpoint_meta_file, "r") as f:
        metadata_store = json.load(f)

    embeddings_list = list(np.load(checkpoint_embs_file))

    with open(processed_paths_file, "r") as f:
        processed_paths = set(json.load(f))

    global_object_id = len(metadata_store)
    print(f"Resumed with {global_object_id} objects already indexed.")
else:
    print("Starting fresh index build.")

gallery_data = get_gallery_items(PARTITION_FILE, DATASET_ROOT, max_images=MAX_IMAGES_TO_PROCESS)

batch_crops    = []
batch_meta_tmp = []

def flush_batch(batch_crops, batch_meta_tmp):
    global global_object_id

    if not batch_crops:
        return

    captions   = generate_captions_batch(batch_crops)
    fused_vecs = generate_fused_embeddings_batch(batch_crops, captions, ALPHA)

    for pil_img, caption, fused, meta in zip(batch_crops, captions, fused_vecs, batch_meta_tmp):
        meta["caption"] = caption
        meta["object_id"] = global_object_id
        embeddings_list.append(fused)
        metadata_store.append(meta)
        global_object_id += 1

for item in tqdm(gallery_data, desc="Indexing gallery"):
    img_path = item["path"]

    if img_path in processed_paths:
        continue

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    height  = img_rgb.shape[0]

    results   = yolo_model(img_rgb, verbose=False)[0]
    found_any = False

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        cls_id = int(box.cls[0])
        label  = yolo_model.names[cls_id]

        if label not in ALLOWED_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        batch_crops.append(Image.fromarray(crop))
        batch_meta_tmp.append({
            "image_path" : img_path,
            "image_name" : item["img_name"],
            "item_id"    : item["item_id"],
            "bbox"       : [x1, y1, x2, y2],
            "label"      : label,
            "region_type": get_region_type(y1, y2, height),
        })
        found_any = True

    if not found_any:
        batch_crops.append(Image.fromarray(img_rgb))
        batch_meta_tmp.append({
            "image_path" : img_path,
            "image_name" : item["img_name"],
            "item_id"    : item["item_id"],
            "bbox"       : [],
            "label"      : "Full Image",
            "region_type": "Full Body",
        })

    processed_paths.add(img_path)

    if len(batch_crops) >= BATCH_SIZE:
        flush_batch(batch_crops, batch_meta_tmp)
        batch_crops.clear()
        batch_meta_tmp.clear()

    if global_object_id > 0 and global_object_id % CHECKPOINT_INTERVAL == 0:
        np.save(checkpoint_embs_file, np.array(embeddings_list, dtype="float32"))
        with open(checkpoint_meta_file, "w") as f:
            json.dump(metadata_store, f)
        with open(processed_paths_file, "w") as f:
            json.dump(list(processed_paths), f)
        tqdm.write(f"Checkpoint saved at {global_object_id} objects.")

flush_batch(batch_crops, batch_meta_tmp)
print(f"\nDone. Total indexed objects: {global_object_id}")

print("Building FAISS index...")

all_embeddings = np.array(embeddings_list, dtype="float32")
dim = all_embeddings.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(all_embeddings)
print(f"Index size: {index.ntotal} vectors")

faiss.write_index(index, os.path.join(OUTPUT_DIR, "gallery_index.faiss"))

with open(os.path.join(OUTPUT_DIR, "gallery_metadata.json"), "w") as f:
    json.dump(metadata_store, f, indent=2)

for f in [checkpoint_meta_file, checkpoint_embs_file, processed_paths_file]:
    if os.path.exists(f):
        os.remove(f)

print("Index and metadata saved. Checkpoints cleaned up.")

print(f"Vectors in index : {index.ntotal}")
print(f"Metadata entries : {len(metadata_store)}")

region_counts = {}
for m in metadata_store:
    r = m["region_type"]
    region_counts[r] = region_counts.get(r, 0) + 1
print(f"Region breakdown : {region_counts}")

label_counts = {}
for m in metadata_store:
    l = m["label"]
    label_counts[l] = label_counts.get(l, 0) + 1
print(f"Label breakdown  : {label_counts}")

print(f"\nOutput folder: {OUTPUT_DIR}")