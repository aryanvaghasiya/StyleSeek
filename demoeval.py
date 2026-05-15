import os
import warnings
import logging
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

import transformers
transformers.logging.set_verbosity_error()

import cv2
import json
import torch
import faiss
import copy
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

# ==========================================
# 1. ARGUMENT PARSING
# ==========================================
parser = argparse.ArgumentParser(description="Evaluate End-to-End Retrieval Pipeline")
parser.add_argument(
    "--dataset_root",
    type=str,
    required=True,
    help="Path to the DeepFashion_InShop dataset root folder"
)
parser.add_argument(
    "--partition_file",
    type=str,
    required=True,
    help="Path to the list_eval_partition.txt file"
)
args = parser.parse_args()

DATASET_ROOT = args.dataset_root
PARTITION_FILE = args.partition_file

if not os.path.exists(PARTITION_FILE):
    raise FileNotFoundError(f"Could not find partition file at: {PARTITION_FILE}")

print(f"Dataset root   : {DATASET_ROOT}")
print(f"Partition file : {PARTITION_FILE}")

# ==========================================
# 2. CONFIGURATION
# ==========================================
ALPHA          = 0.5
CONF_THRESHOLD = 0.4

ALLOWED_CLASSES = [
    "clothing",
    "accessories",
    "shoes",
    "bags",
]

K_VALUES             = [5, 10, 15]
CHECKPOINT_INTERVAL  = 500
MAX_QUERIES_TO_TEST  = 200  # Set to None for final full-dataset evaluation

INDEX_DIR  = "offline_index"
OUTPUT_DIR = "eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==========================================
# 3. LOAD MODELS
# ==========================================
print("\nLoading Clothing Detector (best.pt)...")
yolo_model = YOLO("best.pt")
print(f"Classes: {yolo_model.names}")

print("Loading BLIP-2 (8-bit)...")

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

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def normalize_id(item_id):

    return str(item_id).replace("id_", "").strip()


def get_region_type(y1, y2, img_height):
    center_y = (y1 + y2) / 2
    if center_y < img_height * 0.45:
        return "Upper Body"
    elif center_y > img_height * 0.60:
        return "Lower Body"
    else:
        return "Full Body"


def generate_caption(pil_image):
    inputs = blip_processor(pil_image, return_tensors="pt").to(device, torch.float16)
    with torch.no_grad():
        ids = blip_model.generate(**inputs, max_new_tokens=15)
    return blip_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def generate_fused_embedding(pil_image, caption, alpha):
    inputs = clip_processor(
        text=[caption], images=pil_image, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)

    img_emb  = outputs.image_embeds[0].cpu().numpy()
    text_emb = outputs.text_embeds[0].cpu().numpy()
    img_emb  /= np.linalg.norm(img_emb)
    text_emb /= np.linalg.norm(text_emb)

    fused = alpha * img_emb + (1 - alpha) * text_emb
    return fused / np.linalg.norm(fused)


def process_query_and_search(query_path, top_k=15):
    img = cv2.imread(query_path)
    if img is None:
        return []
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height  = img_rgb.shape[0]

    results    = yolo_model(img_rgb, verbose=False)[0]
    best_crop  = None
    best_conf  = 0.0

    for box in results.boxes:
        conf   = float(box.conf[0])
        cls_id = int(box.cls[0])
        label  = yolo_model.names[cls_id]

        if conf < CONF_THRESHOLD or label not in ALLOWED_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img_rgb[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        if conf > best_conf:
            best_conf = conf
            best_crop = crop

    # Fall back to full image if no clothing region was detected
    pil_crop = Image.fromarray(best_crop if best_crop is not None else img_rgb)

    caption      = generate_caption(pil_crop)
    query_vector = generate_fused_embedding(pil_crop, caption, ALPHA)
    query_vector = query_vector.reshape(1, -1).astype("float32")

 
    pool_size  = top_k * 3
    distances, indices = index.search(query_vector, pool_size)

    retrieved = []
    for i in indices[0]:
        if i == -1:
            continue
        retrieved.append(copy.deepcopy(metadata_store[i]))

    return retrieved[:top_k]

# ==========================================
# 5. LOAD FAISS INDEX & BUILD QUERY LIST
# ==========================================
print("Loading FAISS index and metadata...")
index = faiss.read_index(os.path.join(INDEX_DIR, "gallery_index.faiss"))
with open(os.path.join(INDEX_DIR, "gallery_metadata.json"), "r") as f:
    metadata_store = json.load(f)

print(f"Index size     : {index.ntotal} vectors")
print(f"Metadata size  : {len(metadata_store)} entries")

gallery_item_counts = {}
for item in metadata_store:
    iid = normalize_id(item["item_id"])
    gallery_item_counts[iid] = gallery_item_counts.get(iid, 0) + 1


def get_all_queries(partition_file, dataset_root):
    queries = []
    with open(partition_file, "r") as f:
        lines = f.readlines()

    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < 3 or parts[2] != "query":
            continue

        img_name, item_id = parts[0], parts[1]
        candidates = [
            os.path.join(dataset_root, img_name),
            os.path.join(dataset_root, "img", img_name),
            os.path.join(dataset_root, img_name.replace("img/", "")),
            os.path.join(dataset_root, "img", img_name.replace("img/", "")),
            os.path.join(dataset_root, "img/img", img_name.replace("img/", "")),
        ]
        for path in candidates:
            if os.path.exists(path):
                queries.append({"path": path, "item_id": item_id})
                break

    return queries

# ==========================================
# 6. METRIC FUNCTIONS
# ==========================================
def calculate_metrics(retrieved_item_ids, target_id, k, total_rel):
    if total_rel == 0:
        return 0.0, 0.0, 0.0

    rels = [1 if i == target_id else 0 for i in retrieved_item_ids[:k]]

    recall = 1.0 if sum(rels) > 0 else 0.0

    ap, hits = 0.0, 0
    for i, rel in enumerate(rels):
        if rel == 1:
            hits += 1
            ap += hits / (i + 1.0)
    ap /= min(k, total_rel)

    dcg  = sum(rel / np.log2(i + 2) for i, rel in enumerate(rels))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, total_rel)))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return recall, ap, ndcg

# ==========================================
# 7. EVALUATION LOOP (WITH CHECKPOINTING)
# ==========================================
queries = get_all_queries(PARTITION_FILE, DATASET_ROOT)
if MAX_QUERIES_TO_TEST:
    queries = queries[:MAX_QUERIES_TO_TEST]
print(f"Total query images found : {len(queries)}")

eval_state_file = os.path.join(OUTPUT_DIR, "eval_state.json")
processed_qfile = os.path.join(OUTPUT_DIR, "eval_processed.json")

results           = {k: {"recall": 0.0, "ap": 0.0, "ndcg": 0.0} for k in K_VALUES}
processed_queries = set()

if os.path.exists(eval_state_file) and os.path.exists(processed_qfile):
    print("Resuming from checkpoint...")
    with open(eval_state_file, "r") as f:
        saved = json.load(f)
        results = {int(k): v for k, v in saved.items()}
    with open(processed_qfile, "r") as f:
        processed_queries = set(json.load(f))
    print(f"Already evaluated : {len(processed_queries)} queries")
else:
    print("Starting fresh evaluation.")

queries_to_run = [q for q in queries if q["path"] not in processed_queries]
print(f"Queries remaining : {len(queries_to_run)}\n")

for query in tqdm(queries_to_run, desc="Evaluating"):
    try:
        top_results = process_query_and_search(query["path"], top_k=max(K_VALUES))

        target_id      = normalize_id(query["item_id"])
        retrieved_ids  = [normalize_id(r["item_id"]) for r in top_results]
        total_relevant = gallery_item_counts.get(target_id, 0)

        for k in K_VALUES:
            recall, ap, ndcg = calculate_metrics(retrieved_ids, target_id, k, total_relevant)
            results[k]["recall"] += recall
            results[k]["ap"]     += ap
            results[k]["ndcg"]   += ndcg

        processed_queries.add(query["path"])

        if len(processed_queries) % CHECKPOINT_INTERVAL == 0:
            with open(eval_state_file, "w") as f:
                json.dump(results, f)
            with open(processed_qfile, "w") as f:
                json.dump(list(processed_queries), f)
            tqdm.write(f"Checkpoint saved at {len(processed_queries)} queries.")

    except Exception as e:
        tqdm.write(f"Error on {query['path']}: {e}")
        processed_queries.add(query["path"])

# ==========================================
# 8. FINAL REPORT
# ==========================================
total_evaluated = len(processed_queries)

print("\n" + "=" * 45)
print(f"  FINAL EVALUATION REPORT  (Alpha = {ALPHA})")
print(f"  Total Queries Evaluated : {total_evaluated}")
print(f"  YOLO model              : best.pt (fashion-aware)")
print(f"  FAISS metric            : Inner Product (cosine)")
print("=" * 45)

if total_evaluated > 0:
    for k in K_VALUES:
        r = results[k]["recall"] / total_evaluated
        m = results[k]["ap"]     / total_evaluated
        n = results[k]["ndcg"]   / total_evaluated
        print(f"Metrics @ K={k}:")
        print(f"  Recall@{k:<2} : {r:.4f}")
        print(f"  mAP@{k:<5} : {m:.4f}")
        print(f"  NDCG@{k:<5} : {n:.4f}")
        print("-" * 25)

final_report = {
    "alpha": ALPHA,
    "total_evaluated": total_evaluated,
    "metrics": {
        str(k): {
            "recall": results[k]["recall"] / total_evaluated,
            "mAP":    results[k]["ap"]     / total_evaluated,
            "NDCG":   results[k]["ndcg"]   / total_evaluated,
        }
        for k in K_VALUES
    }
}
with open(os.path.join(OUTPUT_DIR, "final_report.json"), "w") as f:
    json.dump(final_report, f, indent=2)
print(f"\nReport saved to {OUTPUT_DIR}/final_report.json")
