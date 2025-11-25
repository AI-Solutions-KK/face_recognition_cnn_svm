# Face Recognition System — Documentation

## Overview
This project demonstrates a complete **Face Recognition System** using:
- **FaceNet (InceptionResnetV1)** for deep face embeddings  
- **SVM classifier** for identity prediction  
- A celebrity dataset (~18k images, 105 classes)  
- A Streamlit-based UI for demo, analysis, and reports  

The system is modular and split into:
1. **Dataset Repo**: https://huggingface.co/datasets/AI-Solutions-KK/face_recognition_dataset  
2. **Model Repo**: https://huggingface.co/AI-Solutions-KK/face_recognition  
3. **App Repo** (UI): https://huggingface.co/spaces/AI-Solutions-KK/face_recognition_model_demo_app  
   Use if above not worked LIVE_APP - https://facerecognition-tq32v5qkt4ltslejzwymw8.streamlit.app/
---

## Project Architecture
```
FaceNet → Embeddings (512-d) → SVM Classifier → Predictions
```

### Components
- **Face Embedding Model**: FaceNet (pretrained VGGFace2)
- **ML Classifier**: Trained SVM with probability output
- **Evaluation Tools**: Reports, confusion matrix, prediction analysis
- **Frontend**: Streamlit App deployed to HF Spaces
- **Backend Logic**: Custom inference pipeline

---

## Training Summary
- Dataset: 105 classes, ~18,000 cleaned images
- Train / Test Split: 80/20
- Embedding model: FaceNet (fixed)
- Classifier: SVM (RBF kernel)
- Cross‑validation: **99.17% ± 0.13**
- Final test accuracy: **99–99.7%**
- Optional centroid baseline included

---

## Prediction Pipeline
1. Load image  
2. Detect + align face using MTCNN  
3. Generate 512‑d embedding via FaceNet  
4. Normalize using L2 normalizer  
5. Predict class probabilities using SVM  
6. Output:
   - Label  
   - Confidence  
   - Top‑k predictions  

---

## Dataset Structure
```
face_recognition_dataset/
   ├── class_1/
   ├── class_2/
   ├── ...
```
Used for both training and demo inside the app.

Dataset Repo:  
**https://huggingface.co/datasets/AI-Solutions-KK/face_recognition_dataset**

---

## Model Artifacts
Stored separately for reuse:

- `svc_model.pkl`  
- `classes.npy`  
- `centroids.npy`  
- (optional) `X_emb_augmented.npy`  
- (optional) `y_lbl_augmented.npy`

Model Repo:  
**https://huggingface.co/AI-Solutions-KK/face_recognition**

---

## Deployment (App)
The app:
- Downloads dataset using `snapshot_download`
- Downloads model using `hf_hub_download`
- Caches both inside the Space
- Provides:
  - Home: Select an image → Predict  
  - Training Report  
  - Prediction Report  
  - Confusion Matrix page  
  - About page  

App Repo (this demo):  1. Streamlit 2. Hugging face

1. Recomanded

**1) https://facerecognition-tq32v5qkt4ltslejzwymw8.streamlit.app/**
   
**2) https://huggingface.co/spaces/AI-Solutions-KK/face_recognition_model_demo_app**

---

## Run Locally
```bash
git clone https://huggingface.co/spaces/AI-Solutions-KK/face_recognition_model_demo_app
cd face_recognition_model_demo_app
pip install -r requirements.txt
streamlit run src/streamlit_app.py
```

The app automatically downloads model + dataset.

---

## Full Training Code
All training, cleaning, model building & evaluation:
👉 **https://github.com/AI-Solutions-KK/face_recognition_model_domain_specific.git**

---

## Author
**Karan — AI-Solutions-KK**

If you find this helpful, ⭐ star the repo and try the live demos.
