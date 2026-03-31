"""
Configuration file for Brain Tumor Detection System
"""

import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')
DATABASE_DIR = os.path.join(BASE_DIR, 'database')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)

# Model Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0001

# Tumor Types
TUMOR_TYPES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

# Database Configuration
DATABASE_PATH = os.path.join(DATABASE_DIR, 'brain_tumor.db')

# Model Paths
DETECTION_MODEL_PATH = os.path.join(MODEL_DIR, 'tumor_detection_model.h5')
CLASSIFICATION_MODEL_PATH = os.path.join(MODEL_DIR, 'tumor_classification_model.h5')
SEGMENTATION_MODEL_PATH = os.path.join(MODEL_DIR, 'tumor_segmentation_model.h5')

# Dataset URLs (You'll need to download these manually)
DATASET_INFO = """
Recommended Datasets:
1. Brain Tumor MRI Dataset (Kaggle): https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
2. BraTS Dataset: https://www.med.upenn.edu/cbica/brats2020/data.html
3. Brain MRI Images for Tumor Detection: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
"""
