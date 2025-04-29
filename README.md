# ML-IDS-Analyzer

**Authors:** Alexander Zimpher (@AlexZimpher) & Spencer Hendren (@SnakAttack)

---

## 🚀 Overview
ML-IDS-Analyzer is a portfolio project by two final‑year cybersecurity students, designed to demonstrate applied machine learning techniques in network defense. The system classifies IDS alerts (e.g., from Suricata) into **valid (true positive)** or **invalid (false positive)**, reducing noise and helping small‑scale SOC environments focus on actionable threats.

Key highlights:
- **Supervised ML pipeline** with hyperparameter search and probability threshold tuning
- **Explainability** via SHAP integration to visualize feature impacts
- **End‑to‑end demo** including Jupyter notebook and slide deck materials
- **Modular design** for batch and (future) real‑time inference

---

## 📁 Repository Structure
```
ml-ids-analyzer/
├── ml_ids_analyzer/         # Core library package
│   ├── config.py            # YAML configuration loader
│   ├── preprocessing/       # Data cleaning & feature engineering
│   ├── modeling/            # Training, tuning & evaluation
│   ├── inference/           # Batch & streaming prediction scripts
│   └── evaluate.py          # Metrics, plots & explainability
├── notebooks/               # Jupyter notebooks (EDA, demo)
├── data/                    # Raw & processed datasets
├── outputs/                 # Model, scaler & prediction outputs
├── Dockerfile               # Containerize the pipeline
├── setup.py                 # Packaging metadata & console scripts
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🛠 Features & Pipeline
1. **Data Ingestion & Cleaning**
   - Load CICIDS2017 flow‑level intrusion data
   - Handle missing/infinite values and drop low‑quality rows
   - Scale features with `StandardScaler`

2. **Model Training & Hyperparameter Search**
   - Baseline `RandomForestClassifier`
   - `RandomizedSearchCV` over key parameters (`n_estimators`, `max_depth`, `min_samples_leaf`)
   - 5‑fold cross‑validation for robust performance estimates

3. **Threshold Tuning**
   - Precision–Recall curve plotting
   - Automatic selection of probability cutoff maximizing F1‑score

4. **Evaluation & Explainability**
   - Classification report, ROC AUC, confusion matrix visualizations
   - SHAP summary plots to interpret feature contributions

5. **Inference**
   - Batch predictions via `mlids-predict` console script
   - Applies the tuned threshold for real‑world decision making

6. **Demo & Documentation**
   - Jupyter notebook `03_demo.ipynb` showcasing end‑to‑end workflow
   - Slide deck summarizing Purpose, Data, Methods, Results, and Next Steps

---

## 📚 Quickstart
```bash
# Clone the repository
git clone https://github.com/AlexZimpher/ml-ids-analyzer.git
cd ml-ids-analyzer

# Set up a virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: .\venv\Scripts\Activate.ps1

# Install dependencies and the package
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Run the pipeline
mlids-preprocess          # Clean & feature-engineer data
mlids-train               # Train model with hyperparameter search & tuning
mlids-train --no-search   # Fast training with default parameters
mlids-predict             # Generate predictions on new alerts
```

---

## 📊 Demo Notebook
Explore `notebooks/03_demo.ipynb` to see:
- Model & scaler loading
- Sample data ingestion
- Probability generation & threshold application
- Performance metrics and plots (confusion matrix, ROC & PR curves)

---

## 🎯 Tech Stack
| Area                | Tools                             |
|---------------------|-----------------------------------|
| Language            | Python 3.11                       |
| ML & Modeling       | scikit-learn, XGBoost             |
| Data & EDA          | pandas, seaborn, matplotlib       |
| Explainability      | SHAP                              |
| Packaging & CLI     | setuptools, console_scripts       |
| Deployment          | Docker, docker-compose            |

---

## 🔮 Future Directions
- **Real‑time integration:** Stream Suricata EVE JSON for live inference
- **Web dashboard:** Interactive threshold adjustment via Flask/Streamlit
- **Continuous retraining:** Automate hyperparameter tuning on data drift
