<div align="center">

# 🧠 NeuroMarket
### EEG-Based Purchase Prediction System

*An Explainable AI framework for predicting consumer purchase decisions from EEG brain signals using Machine Learning.*

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-green)
![SHAP](https://img.shields.io/badge/Explainable-AI-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-red?logo=scikitlearn)
![Dash](https://img.shields.io/badge/Dash-Interactive%20Dashboard-blue)
![Status](https://img.shields.io/badge/Status-Research%20Project-brightgreen)

---

### 🧠 Predicting Consumer Purchase Intent from Brain Signals using Explainable AI

</div>

---

# 📖 Overview

NeuroMarket is an Explainable AI (XAI) framework that predicts consumer purchase decisions using Electroencephalography (EEG) brain signals from the **SEED dataset**. The system extracts frequency-band features from EEG recordings and trains an **XGBoost** classifier using **Leave-One-Subject-Out (LOSO) Cross-Validation** to evaluate performance on previously unseen individuals.

To improve model transparency, **SHAP (SHapley Additive exPlanations)** is integrated to identify the contribution of individual brainwave features, enabling interpretable and trustworthy predictions.

---

# ✨ Key Features

- 🧠 EEG-based consumer purchase prediction
- 🤖 Machine Learning using XGBoost
- 📊 Explainable AI with SHAP
- 🔬 Leave-One-Subject-Out Cross Validation
- 📈 Interactive Dash dashboard
- ⚡ Real-time prediction visualization
- 🎯 Generalization across unseen subjects
- 📉 Comprehensive performance evaluation

---

# 🛠️ Tech Stack

| Category | Technologies |
|-----------|--------------|
| Programming | Python |
| Machine Learning | XGBoost, Scikit-learn |
| Explainable AI | SHAP |
| Data Analysis | NumPy, Pandas |
| Visualization | Plotly, Dash |
| Model Persistence | Joblib |

---

# 🏗️ System Architecture

```text
              SEED EEG Dataset
                     │
                     ▼
            EEG Trial Loading
                     │
                     ▼
          Feature Extraction
        (310 EEG Features)
                     │
                     ▼
      XGBoost Classification Model
                     │
                     ▼
 Leave-One-Subject-Out Validation
                     │
                     ▼
      SHAP Explainability Analysis
                     │
                     ▼
      Interactive Dash Dashboard
```

---

# 📂 Project Structure

```text
NeuroMarket-EEG-Purchase-Prediction
│
├── app/
│   └── dashboard.py
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── prediction.py
│   ├── explainability.py
│   ├── evaluation.py
│   └── utils.py
│
├── data/
├── models/
├── notebooks/
├── screenshots/
├── assets/
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🚀 Getting Started

## Clone the Repository

```bash
git clone https://github.com/yourusername/NeuroMarket-EEG-Purchase-Prediction.git

cd NeuroMarket-EEG-Purchase-Prediction
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Dashboard

```bash
python app/dashboard.py
```

The Dash application will launch locally in your browser.

---

# 📊 Model Performance

| Metric | Score |
|--------|-------:|
| Accuracy | **80.3%** |
| Validation Strategy | Leave-One-Subject-Out |
| Dataset | SEED EEG Dataset |
| Subjects | 15 |
| EEG Trials | 34,110 |

---

# 🧠 Explainable AI

The model integrates **SHAP** to provide transparent predictions by identifying the contribution of EEG frequency-band features.

Key insights include:

- Gamma-band activity showed the highest contribution to purchase prediction.
- Feature importance was visualized using SHAP summary and force plots.
- Model decisions can be interpreted at both global and individual prediction levels.

---

# 🎯 Applications

- Neuromarketing Research
- Consumer Behavior Analysis
- Brain-Computer Interfaces
- Explainable AI Research
- Predictive Analytics
- Human Decision Intelligence
- Cognitive Computing

---

# 📈 Future Enhancements

- Deep Learning-based EEG classification
- Real-time EEG signal processing
- Multiclass purchase intent prediction
- Cloud deployment using AWS
- REST API integration
- Docker containerization

---

# 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

Feel free to fork the repository, submit issues, or create pull requests to enhance NeuroMarket.

---

# 👩‍💻 Author

**Gnana Shree M B**

M.Tech in Data Science  
JSS Science and Technology University

📧 gnanashree006@gmail.com

🔗 LinkedIn: *(Add your LinkedIn URL here)*

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
