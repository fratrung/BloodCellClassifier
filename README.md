# Blood Cell Classifier

**Author:** Francesco Trungadi
**Course:** Artificial Intelligence (Master’s Level)  
**Date:** 10/02/2025

---

## 📌 Project Overview

This project implements an **image classification system** for blood cells, aiming to automatically identify the type of cell from microscopic images.  
The project was developed as part of the **Master’s course in Artificial Intelligence** and demonstrates practical application of machine learning and deep learning techniques on biomedical image data.

---

## 📂 Dataset

The dataset used is the [**Blood Cells Image Dataset**](https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset) from Kaggle.  

- Images are organized by folder, where each folder represents a cell type.  
- Cell types included:  
  - **EOSINOPHIL** (white blood cell)  
  - **LYMPHOCYTE** (white blood cell)  
  - **MONOCYTE** (white blood cell)  
  - **NEUTROPHIL** (white blood cell)  
  - **BASOPHIL** (white blood cell)  
  - **PLATELET** (platelets)  
  - **RBC** (red blood cells)  

### Data Preprocessing

- Images are loaded with **OpenCV** and resized to **64×64 pixels** to reduce memory usage and speed up training while preserving essential visual features.  
- Pixel values are normalized to the **[0,1] range**.  
- Labels are encoded with **LabelEncoder** and converted to **one-hot vectors** for categorical cross-entropy loss.  
- Dataset is split into **training, validation, and test sets** with stratification to preserve class distribution.  
- **Class weights** are computed to address class imbalance during training.

---

## 🧪 Baseline Model: Logistic Regression

Before the CNN, a **multinomial logistic regression** was trained as a baseline:  

- Images were **flattened** into vectors.  
- Accuracy and classification report were calculated to evaluate the problem difficulty.  

This step provides a **performance reference** and highlights the advantage of deep learning for image classification.

---

## 🧠 Main Model: Convolutional Neural Network (CNN)

The main model is a **CNN** designed to automatically extract visual features from cell images.

### Architecture

- Input: 64×64×3  
- Convolutional layers with increasing filters (32 → 256)  
- Batch normalization after each convolution  
- MaxPooling2D for downsampling  
- Dropout layers to reduce overfitting  
- Fully connected layers with **softmax output** matching the number of classes  

### Training

- **Optimizer:** Adam (learning rate 1e-5)  
- **Loss:** Categorical cross-entropy  
- **Metrics:** Accuracy  
- **Early stopping** on validation loss (patience=10)  
- **Class weights** applied to mitigate imbalance  

### Results

- The model achieves **high accuracy** on the test set, correctly classifying multiple blood cell types.  
- Confusion matrices and classification reports provide detailed insights into class-specific performance.

---

## 📊 Visualization

- Sample images with predictions are displayed from the test set.  
- Training and validation curves for **accuracy and loss** show learning behavior over epochs.  

---

## 📝 Conclusion

This project demonstrates the **application of deep learning to biomedical image classification**.  
The CNN outperforms the baseline logistic regression, highlighting the advantage of automatic feature extraction.  
It serves as a practical implementation for the **Artificial Intelligence Master’s course**, combining data preprocessing, classical ML, and deep learning techniques.