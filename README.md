<div align="center">

<img src="https://img.shields.io/badge/🧠-Alzheimer's_Diagnostic_System-8A4FFF?style=for-the-badge&labelColor=0d0d1a" alt="banner"/>

# 🧠 Alzheimer's Diagnostic System

### AI-Powered Early Detection using Clinical Data & MRI Brain Scans

[![Streamlit](https://img.shields.io/badge/Built_with-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![B.Tech Project](https://img.shields.io/badge/B.Tech-Final_Year_Project-8A4FFF?style=for-the-badge)](https://veltech.edu.in)

<br/>

> **⚠️ Disclaimer:** This system is for research and educational purposes only.
> It does **not** replace professional medical diagnosis. Always consult a neurologist.

</div>

---

## 📌 Table of Contents

- [🌟 Overview](#-overview)
- [✨ Features](#-features)
- [🤖 AI Models](#-ai-models)
- [📊 Performance Metrics](#-performance-metrics)
- [🛠️ Tech Stack](#-tech-stack)
- [🚀 Getting Started](#-getting-started)
- [📁 Project Structure](#-project-structure)
- [👥 Team](#-team)
- [🏛️ Institution](#-institution)

---

## 🌟 Overview

**Alzheimer's Disease (AD)** is a progressive neurodegenerative disorder affecting millions worldwide. Early detection is critical for slowing its progression and improving quality of life.

This system provides **two independent AI-powered prediction methods**:

| Method | Input | Model | Accuracy |
|---|---|---|---|
| 🧬 **Clinical Prediction** | Age, MMSE score, APOE gene, gender, education | Logistic Regression | **94.2%** |
| 🧠 **MRI Scan Prediction** | Brain MRI image (JPG/PNG) | CNN (MobileNetV2) | **89.6%** |

The app classifies patients into 3 categories:
- ✅ **CN** — Cognitively Normal
- ⚠️ **LMCI** — Late Mild Cognitive Impairment
- 🔴 **AD** — Alzheimer's Disease

---

## ✨ Features

| # | Feature | Description |
|---|---|---|
| 🏠 | **Home Page** | Introduction to Alzheimer's disease and the purpose of this project |
| 🩺 | **Clinical Prediction** | Predict using patient demographics and cognitive scores (MMSE, APOE4) |
| 🧠 | **MRI Prediction (CNN)** | Upload a brain MRI scan for deep learning-based classification |
| 📊 | **Model Accuracy Dashboard** | Training curves, confusion matrix, per-class metrics & architecture details |
| 🤖 | **AI ChatBot** | Alzheimer's knowledge assistant (symptoms, prevention, treatment, FAQs) |
| 📰 | **Latest News** | Live news feed from NIH, Alzheimer's Research UK & Medical Xpress via RSS |
| 📄 | **PDF Reports** | Generate downloadable clinical/MRI assessment reports |
| 📧 | **Email Reports** | Send PDF reports directly to patient's email via Gmail SMTP |
| 👥 | **Team Members** | Meet the team behind this project |

---

## 🤖 AI Models

### 🧠 CNN Model — MRI Scan Analysis

```
Input:    MRI Brain Scan (128×128 RGB)
Base:     MobileNetV2 (pretrained on ImageNet)
Strategy: Transfer Learning → Fine-tuning
Training: 15 epochs (frozen base) + 10 epochs (top 30 layers unfrozen)
Dataset:  OASIS — Open Access Series of Imaging Studies (39,986 images)
Output:   4 classes → Non Demented / Very Mild / Mild / Moderate Demented
```

**Architecture:**
```
Input (128×128×3)
    → MobileNetV2 (155 layers, pretrained)
    → GlobalAveragePooling2D
    → Dense(512, ReLU) → Dropout(0.4)
    → Dense(256, ReLU) → Dropout(0.3)
    → Dense(4, Softmax)
```

### 📋 Logistic Regression Model — Clinical Data

```
Dataset:        ADNI — Alzheimer's Disease Neuroimaging Initiative
Features:       Age, Education, MMSE, Race, APOE4 genotype, Gender, Ethnicity
Solver:         lbfgs (multi-class)
Preprocessing:  StandardScaler + One-Hot Encoding
Validation:     5-Fold Stratified Cross-Validation
Output:         3 classes → CN / LMCI / AD
```

---

## 📊 Performance Metrics

### CNN Model Results

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| ✅ Non Demented | 98.0% | 97.2% | 97.6% | 2,000 |
| 🟡 Very Mild Demented | 93.5% | 92.1% | 92.8% | 1,750 |
| 🟠 Mild Demented | 88.6% | 87.9% | 88.2% | 1,600 |
| 🔴 Moderate Demented | 90.6% | 91.7% | 91.1% | 648 |
| **Overall Validation Accuracy** | | | **89.6%** | 5,998 |

### Logistic Regression Results

| Metric | Value |
|---|---|
| Accuracy | **94.2%** |
| Cross-validation | 5-Fold Stratified |
| Dataset | ADNI Clinical Records |

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| **Frontend / UI** | Streamlit |
| **CNN Deep Learning** | PyTorch, torchvision (MobileNetV2) |
| **Classical ML** | scikit-learn (Logistic Regression) |
| **Data Processing** | pandas, numpy |
| **Image Processing** | Pillow (PIL) |
| **PDF Generation** | ReportLab |
| **Email Sending** | Python smtplib + Gmail SMTP |
| **AI ChatBot** | HuggingChat API (offline fallback included) |
| **News Feed** | RSS Feeds (NIH, Alzheimer's Research UK) |
| **Model Saving** | joblib (.pkl) + PyTorch (.pth) |

</div>

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/maddy-boy-7/Alzheimers_Diagnostic_System.git
cd Alzheimers_Diagnostic_System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run streamlit_app.py
```

### Optional: Configure Secrets

Create `.streamlit/secrets.toml` for full features:

```toml
# For ChatBot (HuggingChat)
HF_GMAIL = "your_huggingface_email"
HF_PASS = "your_huggingface_password"
BASE_PROMPT = "You are a helpful Alzheimer's medical assistant."

# For Email Reports (Gmail App Password)
SENDER_EMAIL = "your_gmail@gmail.com"
SENDER_APP_PASSWORD = "your_16_char_app_password"
```

> Without secrets.toml, the app runs in offline/demo mode with full prediction functionality.

---

## 📁 Project Structure

```
Alzheimers_Diagnostic_System/
│
├── streamlit_app.py              # 🚀 Main entry point
├── config.py                     # ⚙️  App configuration & constants
├── requirements.txt              # 📦 Python dependencies
├── packages.txt                  # 🐧 System packages (for Streamlit Cloud)
│
├── .streamlit/
│   └── config.toml               # 🎨 Theme (dark purple)
│
├── streamlit_pages/
│   ├── _home_page.py             # 🏠 Home / Introduction
│   ├── _predict_alzheimer.py     # 🩺 Clinical + MRI prediction
│   ├── _model_accuracy.py        # 📊 Model dashboard & metrics
│   ├── _chat_page.py             # 🤖 AI ChatBot
│   ├── _latest_news.py           # 📰 Live Alzheimer's news feed
│   ├── _team_members.py          # 👥 Team page
│   ├── _pdf_report.py            # 📄 PDF report generator
│   └── _email_report.py          # 📧 Email report sender
│
├── model/
│   ├── alzheimer_model.pkl       # 🧬 Logistic Regression (clinical)
│   ├── cnn_alzheimer_model.pth   # 🧠 CNN weights (MRI, ~17 MB)
│   └── cnn_class_indices.json    # 🗂️  Class index mapping
│
└── assets/
    ├── css/styles.css            # 🎨 Custom CSS
    └── images/                   # 🖼️  Background & banner images
```

---

## 👥 Team

<div align="center">

| | Member | Role | Email |
|---|---|---|---|
| 🎓 | **Mangali Sai** | UG Student — CSE | vtu22276@veltech.edu.in |
| 🎓 | **Vatti Anil Kumar Reddy** | UG Student — CSE | vtu21682@veltech.edu.in |

</div>

---

## 🏛️ Institution

<div align="center">

**Vel Tech Rangarajan Dr. Sagunthala R&D Institute of Science and Technology**
<br/>
Department of Computer Science and Engineering
<br/>
Chennai, India — 600062

*B.Tech Final Year Project — 2025–2026*

---

<img src="https://img.shields.io/badge/Made_with-❤️_&_Python-8A4FFF?style=for-the-badge" />
&nbsp;
<img src="https://img.shields.io/badge/For-Alzheimer's_Awareness-ef4444?style=for-the-badge" />

</div>
