# StyleSeek - Visual Product Search Engine

Visual Recognition Course Project

This project implements an end-to-end visual product search system using:

* Clothing-aware YOLOv8 detection
* BLIP/BLIP-2 semantic captioning
* CLIP multimodal embeddings
* FAISS vector similarity search
* Streamlit interactive frontend

The system allows users to upload an image, select a detected clothing region, and retrieve visually and semantically similar fashion products.

---

# Project Structure

```bash
.
├── app.py
├── batchEval.py
├── best.pt
├── offline_index
│   ├── gallery_index.faiss
│   └── gallery_metadata.json
├── offline_pipeline.py
├── requirements.txt
└── VR-Final-Project.pdf
```

---

# Requirements

* Python 3.10+
* Conda (recommended)
* NVIDIA GPU recommended (optional)
* Linux tested

---

# Environment Setup

## 1. Create Conda Environment

```bash
conda create -n vr-opencv python=3.10 -y
```

## 2. Activate Environment

```bash
conda activate vr-opencv
```

---

# Install Dependencies

Run this only once:

```bash
pip install -r requirements.txt
```

---

# Running the Streamlit Application

From the project root directory:

```bash
streamlit run app.py
```

The app will automatically open in your browser.

If it does not open automatically, visit:

```bash
http://localhost:8501
```

---

# Streamlit Application Workflow

1. Upload a query image
2. Clothing-aware YOLO detects clothing regions
3. User selects the clothing item/region to search
4. Caption + embedding generation is performed
5. FAISS retrieves similar products
6. Top-K results are displayed

---


# Notes

* The project uses a clothing-aware YOLOv8 detector (`best.pt`) instead of generic COCO YOLO.
* The same alpha value (`α = 0.5`) is used consistently across offline indexing, retrieval, and evaluation.
* FAISS inner-product similarity is used on normalized embeddings.
* The offline indexing pipeline uses batch processing (`batch_size = 8`) to accelerate BLIP/CLIP inference.

---

# Tested Configuration

Tested on:

* Ubuntu Linux
* Python 3.10
* CUDA-enabled NVIDIA GPU

The project can also run on CPU(current configuration). the blip part may take 15-20 seconds

---

# Authors

* Aryan Vaghasiya
* Areen Vaghasiya
* Madhav Patil

