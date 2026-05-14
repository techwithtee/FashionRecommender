## 🧥 AI Stylist — Fashion Classification & Recommendation System

Awarded a First-Class grade for my Final Year Capstone Project

An end-to-end deep learning system that classifies fashion items and recommends visually similar outfits. Built in Python using PyTorch, FastAI, and Annoy, trained on the large-scale DeepFashion benchmark dataset.

## 📌 Overview
AI Stylist takes a clothing image as input and returns the most visually similar items from a catalogue of hundreds of thousands of fashion images. It combines a fine-tuned convolutional neural network for feature extraction with an approximate nearest-neighbour index for fast, scalable retrieval.
The system was designed to mirror the kind of "complete the look" or "shop similar" features found in commercial fashion platforms, implemented from scratch as an academic project.

## 🏆 Results
MetricScoreTop-1 AccuracyAchieved via fine-tuned ResNet-50Top-5 AccuracyTracked across all 50 categoriesDataset Size800k+ images, 50 clothing categoriesSimilarity SearchSub-linear retrieval via Annoy (20 trees)

## 🛠️ Tech Stack
ComponentTechnologyDeep Learning FrameworkPyTorch + FastAIModel ArchitectureResNet-50 (pretrained on ImageNet, fine-tuned)Training StrategyTransfer learning with fit_one_cycle (cosine annealing LR)Embedding Dimension2048-d feature vectorsSimilarity SearchAnnoy (Approximate Nearest Neighbours Oh Yeah)DatasetDeepFashionInteractive UIipywidgets (file upload + recommendation button)EnvironmentGoogle Colab (GPU runtime) + Google Drive for persistence

## 🔄 Pipeline
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

## 📓 Notebook Structure
SectionDescription1. InstallationInstall FastAI, gdown, Annoy, ipywidgets2. ImportsCore libraries and FastAI vision modules3. Mount Google DrivePersist model checkpoints and embeddings across sessions4. Dataset DownloadPull DeepFashion files from Google Drive via gdown5. Data ParsingBuild a structured Pandas DataFrame from annotation files6. DataLoaderFastAI DataBlock with augmentation and indexed train/test split7. Model TrainingFine-tune ResNet-50, LR finder, fit_one_cycle8. Model EvaluationConfusion matrix and top-loss inspection across 50 categories9. Embedding & IndexingBatch embedding generation and Annoy index construction10. RecommendationNearest-neighbour lookup and visual results display11. Interactive Widgetipywidgets UI for uploading images and getting recommendations

## 🚀 Getting Started
Prerequisites

Google Colab (recommended, GPU runtime required for training)
Google Drive (for dataset and model persistence)
Python 3.8+

Running the Notebook

Open in Google Colab — upload AIStylist_enhanced.ipynb or open directly from your Drive.
Enable GPU — Runtime → Change runtime type → GPU.
Run Section 1 to install dependencies.
Run Section 3 to mount your Google Drive.
Run Section 4 to download the DeepFashion dataset (~several GB, first run only).
Run Sections 5–8 to train the classifier (or skip to Section 9 if loading a saved model).
Run Section 9 to generate embeddings and build the Annoy index (or load from Drive if previously saved).
Run Section 11 to launch the interactive recommender widget.


⚠️ Training (Section 7) and embedding generation (Section 9) are computationally intensive. On a Colab T4 GPU, expect ~30–60 minutes for the full pipeline on first run. Saved checkpoints and index files mean subsequent runs are instant.


## 📁 Project Structure
├── AIStylist_enhanced.ipynb   # Main notebook
├── README.md                  # This file
Model weights, embeddings, and the Annoy index are saved to Google Drive (not included in this repo due to file size).

## 📚 References

Liu, Z. et al. (2016). DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations. CVPR 2016.
He, K. et al. (2015). Deep Residual Learning for Image Recognition. CVPR 2016.
Howard, J. & Gugger, S. FastAI: A Layered API for Deep Learning. Information, 2020.
Bernhardsson, E. Annoy: Approximate Nearest Neighbours Oh Yeah. Spotify.


##📄 Licence
This project  was developed for academic purposes. The DeepFashion dataset is subject to its own licence terms.
