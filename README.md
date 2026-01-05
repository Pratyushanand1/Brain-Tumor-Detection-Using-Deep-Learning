# Brain Tumor Detection Using Deep Learning

A deep learning–based MRI image classification system that detects the presence of a brain tumor and identifies its type. The project uses **transfer learning with VGG16** to achieve high accuracy on multi-class brain MRI data.

This repository is designed so that **when users upload MRI images**, the system automatically **processes the images and predicts whether a tumor is present**, along with the tumor category and confidence score.

---

## 🚀 What This Project Does

* Accepts **MRI brain scan images** as input
* Preprocesses and normalizes uploaded images
* Uses a **pre-trained VGG16 CNN** for feature extraction
* Classifies images into one of the following categories:

  * Glioma
  * Meningioma
  * Pituitary Tumor
  * No Tumor
* Displays prediction results with **confidence percentage**
* Achieves ~95% overall test accuracy

---

## 🧠 Model Overview

* Architecture: **VGG16 (Transfer Learning)**
* Input Size: `128 × 128 × 3`
* Trainable Layers: Last 3 convolutional layers of VGG16
* Loss Function: `Sparse Categorical Crossentropy`
* Optimizer: `Adam (lr=0.0001)`
* Frameworks: TensorFlow / Keras

---

## 📁 Repository Structure

```
brain-tumor-detection/
│
├── model/
│   └── model.h5                # Trained VGG16-based model
│
├── data/
│   ├── Training/
│   └── Testing/
│
├── notebooks/
│   └── brain_tumor_detection.ipynb
│
├── inference.py                 # Image upload & prediction logic
├── requirements.txt             # Python dependencies
├── README.md
└── .gitignore
```

---

## 📸 Image Upload & Detection Workflow

1. User uploads an MRI image
2. Image is resized and normalized
3. Model predicts tumor class
4. System checks:

   * If prediction = `notumor` → **No Tumor Detected**
   * Else → **Tumor Detected + Tumor Type**
5. Confidence score is displayed

---

## 🧪 Example Prediction Output

```
Tumor: Glioma
Confidence: 96.45%
```

```
No Tumor
Confidence: 98.12%
```

---

## 🧩 Inference Code (Core Logic)

```python
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']
model = load_model('model/model.h5')

def predict_image(image_path, image_size=128):
    img = load_img(image_path, target_size=(image_size, image_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    return predicted_class, confidence
```

---

## 📊 Performance Metrics

* Test Accuracy: **95%**
* Precision / Recall / F1-score evaluated per class
* ROC-AUC curves generated for all tumor categories
* Confusion matrix included for interpretability

---

## 🛠️ Installation & Setup

```bash
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
pip install -r requirements.txt
```

---

## 📦 Requirements

```
tensorflow
keras
numpy
pandas
matplotlib
scikit-learn
pillow
seaborn
```

---

## 🎯 Use Cases

* Medical image analysis demos
* AI/ML portfolio projects
* Academic research prototypes
* Healthcare screening systems (non-clinical)

---

## ⚠️ Disclaimer

This project is intended **for educational and research purposes only** and should not be used for clinical diagnosis.

---

## 👤 Author

**Pratyush Anand**
Computer Science Undergraduate | Backend & ML Enthusiast
GitHub • LinkedIn • Portfolio

---

If you want, this can be easily extended into a **FastAPI / Flask web app** where users upload images through a browser.
