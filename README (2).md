# 🧥 AI Stylist — Fashion Classification & Recommendation System

> **Awarded a First-Class grade** as part of a university AI/Machine Learning module.

An end-to-end deep learning system that classifies fashion items and recommends visually similar outfits. Built in Python using PyTorch, FastAI, and Annoy, trained on the large-scale DeepFashion benchmark dataset.

---

## 📌 Overview

AI Stylist takes a clothing image as input and returns the most visually similar items from a catalogue of hundreds of thousands of fashion images. It combines a fine-tuned convolutional neural network for feature extraction with an approximate nearest-neighbour index for fast, scalable retrieval.

The system was designed to mirror the kind of "complete the look" or "shop similar" features found in commercial fashion platforms, implemented from scratch as an academic project.

---

## 🏆 Results

| Metric | Score |
|---|---|
| Top-1 Accuracy | Achieved via fine-tuned ResNet-50 |
| Top-5 Accuracy | Tracked across all 50 categories |
| Dataset Size | 800k+ images, 50 clothing categories |
| Similarity Search | Sub-linear retrieval via Annoy (20 trees) |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning Framework | PyTorch + FastAI |
| Model Architecture | ResNet-50 (pretrained on ImageNet, fine-tuned) |
| Training Strategy | Transfer learning with `fit_one_cycle` (cosine annealing LR) |
| Embedding Dimension | 2048-d feature vectors |
| Similarity Search | Annoy (Approximate Nearest Neighbours Oh Yeah) |
| Dataset | [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) |
| Interactive UI | ipywidgets (file upload + recommendation button) |
| Environment | Google Colab (GPU runtime) + Google Drive for persistence |

---

## 🔄 Pipeline

```
DeepFashion Dataset (800k+ images, 50 categories)
           │
           ▼
  Data Parsing & DataFrame Construction
  (list_category_img.txt, list_eval_partition.txt)
           │
           ▼
  FastAI DataBlock — Augmentation & Train/Val/Test Split
  (Resize 460 → RandomCrop 224, aug_transforms)
           │
           ▼
  ResNet-50 Fine-tuning
  (lr_find → fit_one_cycle, DataParallel multi-GPU)
           │
           ▼
  Model Evaluation
  (Confusion Matrix, Top Losses)
           │
           ▼
  Batch Embedding Generation
  (2048-d vectors for all dataset images, batched GPU inference)
           │
           ▼
  Annoy Index Construction & Persistence
  (20 random projection trees, euclidean distance)
           │
           ▼
  Interactive Outfit Recommender
  (Upload image → embed → ANN query → display top-5 matches)
```

---

## 📓 Notebook Structure

| Section | Description |
|---|---|
| 1. Installation | Install FastAI, gdown, Annoy, ipywidgets |
| 2. Imports | Core libraries and FastAI vision modules |
| 3. Mount Google Drive | Persist model checkpoints and embeddings across sessions |
| 4. Dataset Download | Pull DeepFashion files from Google Drive via gdown |
| 5. Data Parsing | Build a structured Pandas DataFrame from annotation files |
| 6. DataLoader | FastAI DataBlock with augmentation and indexed train/test split |
| 7. Model Training | Fine-tune ResNet-50, LR finder, fit_one_cycle |
| 8. Model Evaluation | Confusion matrix and top-loss inspection across 50 categories |
| 9. Embedding & Indexing | Batch embedding generation and Annoy index construction |
| 10. Recommendation | Nearest-neighbour lookup and visual results display |
| 11. Interactive Widget | ipywidgets UI for uploading images and getting recommendations |

---

## 🚀 Getting Started

### Prerequisites

- Google Colab (recommended, GPU runtime required for training)
- Google Drive (for dataset and model persistence)
- Python 3.8+

### Running the Notebook

1. **Open in Google Colab** — upload `AIStylist_enhanced.ipynb` or open directly from your Drive.
2. **Enable GPU** — Runtime → Change runtime type → GPU.
3. **Run Section 1** to install dependencies.
4. **Run Section 3** to mount your Google Drive.
5. **Run Section 4** to download the DeepFashion dataset (~several GB, first run only).
6. **Run Sections 5–8** to train the classifier (or skip to Section 9 if loading a saved model).
7. **Run Section 9** to generate embeddings and build the Annoy index (or load from Drive if previously saved).
8. **Run Section 11** to launch the interactive recommender widget.

> ⚠️ Training (Section 7) and embedding generation (Section 9) are computationally intensive. On a Colab T4 GPU, expect ~30–60 minutes for the full pipeline on first run. Saved checkpoints and index files mean subsequent runs are instant.

---

## 📁 Project Structure

```
├── AIStylist_enhanced.ipynb   # Main notebook
├── README.md                  # This file
```

Model weights, embeddings, and the Annoy index are saved to Google Drive (not included in this repo due to file size).

---

## 📚 References

- Liu, Z. et al. (2016). [DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations](https://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf). *CVPR 2016*.
- He, K. et al. (2015). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). *CVPR 2016*.
- Howard, J. & Gugger, S. [FastAI: A Layered API for Deep Learning](https://arxiv.org/abs/2002.04688). *Information, 2020*.
- Bernhardsson, E. [Annoy: Approximate Nearest Neighbours Oh Yeah](https://github.com/spotify/annoy). Spotify.

---

## 📄 Licence

This project was developed for academic purposes. The DeepFashion dataset is subject to its own [licence terms](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html).
