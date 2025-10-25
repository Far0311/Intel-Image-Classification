# 🧠 Intel Image Classification

This notebook contains the **source code and documentation** for a deep learning project that classifies natural scene images into six categories — **buildings, forest, glacier, mountain, sea,** and **street** — using **Convolutional Neural Networks (CNN)**, **ResNet50**, and **EfficientNet-B0** architectures.

---

## 📘 Project Description
This project demonstrates how computer vision and deep learning can be used to **automate scene recognition**.  
The work is focused on applying machine learning techniques for image classification.

The included **PDF file (`Intel Image Classification.pdf`)** provides:
- Complete code snippets for data preprocessing, model training, and evaluation  
- Explanations of the methodology, dataset, and model selection process  
- Results from **Optuna hyperparameter tuning**  
- Visualizations of performance metrics and loss curves  

---

## 🗂️ Files Included
- `PMA_Intel_Image_Classification_Cleaned.ipynb` — Jupyter Notebook containing all code to train and evaluate CNN, ResNet50, and EfficientNet-B0 models.  
- `Intel Image Classification.pdf` — Exported report version of the notebook containing **code cells and outputs**, suitable for review and documentation purposes.  

---

## 📊 Dataset
- **Source:** [Intel Image Classification Dataset – Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)  
- Contains over **25,000 labeled RGB images** across six scene classes.  
- All images are resized to **150×150 pixels** and normalized for CNN input.

---

## ⚙️ Models Implemented
1. **Simple CNN** — Baseline model trained from scratch  
2. **ResNet50** — Transfer learning with frozen feature extractor  
3. **EfficientNet-B0** — State-of-the-art architecture optimized for accuracy and efficiency  

Hyperparameter tuning was performed using **Optuna**, optimizing:
- Learning rate  
- Weight decay  
- Dropout rate  
- Optimizer type (Adam / SGD)  
- Batch size and momentum (for SGD)  

---

## 🧾 Results Summary

| Model | Best Validation Accuracy | Optimizer | Learning Rate |
|--------|--------------------------|------------|----------------|
| CNN | 87.78% | Adam | 0.00044 |
| ResNet50 | **89.63%** | SGD | 0.00587 |
| EfficientNet-B0 | 86.89% | SGD | 0.00271 |

---
