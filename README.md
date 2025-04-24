# ML-IDS-Analyzer (Private Planning)

A machine learning-based alert validation system that classifies IDS alerts (e.g., from Suricata) as likely valid (true positive) or invalid (false positive). Designed and developed by final-year cybersecurity students, this project showcases applied ML in network defense by reducing alert noise and surfacing actionable threats.

---

## 🧠 Project Purpose

- Predict the likelihood that an IDS alert is valid using supervised ML.
- Minimize false positives in student SOC environments and small-scale deployments.
- Demonstrate practical application of cybersecurity data analysis and ML/AI modeling.
- Build a professional-grade portfolio project with real-world impact.

---

## 🗂️ Folder Structure

ml-ids-analyzer/

├── data/                   # Raw and processed datasets
│   └── cicids2017/         # Dataset folder
├── notebooks/              # Jupyter notebooks for EDA, modeling
│   └── 01_eda.ipynb
│   └── 02_model_training.ipynb
├── src/                    # Core source code (modular Python scripts)
│   ├── __init__.py
│   ├── preprocess.py       # Data parsing and feature engineering
│   ├── model.py            # Model training and inference
│   └── evaluate.py         # Evaluation metrics and reports
├── outputs/                # Model outputs, prediction results
├── config/                 # Config files for paths, parameters
├── requirements.txt        # Python package requirements
├── .gitignore
└── README.md               # Project overview and usage

## ✅ Comprehensive To-Do List

### 🔹 Phase 1: Setup & Data
- [x] Create GitHub repo and structure
- [x] Initialize Git LFS and attempt push
- [x] Move CICIDS2017 CSVs into `data/cicids2017/`

### 🔹 Phase 2: Preprocessing
- [ ] Write `preprocess.py` to:
  - Merge CSVs
  - Clean columns
  - Handle missing values
  - Label encode target
  - Save merged dataset
- [ ] Engineer useful features from:
  - IPs, ports, protocols
  - Byte/packet statistics
  - Frequency/entropy heuristics

### 🔹 Phase 3: EDA & Baseline Modeling
- [ ] Load dataset in notebook
- [ ] Plot class imbalance and distributions
- [ ] Train baseline models (logistic regression, decision tree)
- [ ] Evaluate with ROC-AUC, F1, confusion matrix
- [ ] Write `model.py` to automate training

### 🔹 Phase 4: Evaluation
- [ ] Write `evaluate.py` for consistent metrics
- [ ] Add model explainability (feature importance)
- [ ] Store output: CSV with predictions + confidence

### 🔹 Phase 5: Presentation & Documentation
- [ ] Create demo notebook (end-to-end pipeline)
- [ ] Prepare internal slide deck
- [ ] Clean `README.md` for public view
- [ ] (Optional) Record project walkthrough video

---

## 🧪 Dataset Info

- **Dataset:** CICIDS2017
- **Type:** Labeled flow-level intrusion detection data
- **Size:** 8 CSVs (~200MB total)
- **Source:** Canadian Institute for Cybersecurity (UNB)
- **Current status:** Stored locally under `data/cicids2017/`, not pushed due to GitHub LFS limit

---

## 🧰 Tech Stack

| Area          | Tool                        | Purpose                                 |
|---------------|-----------------------------|-----------------------------------------|
| Language      | Python 3.11                 | Core programming                        |
| ML            | scikit-learn, XGBoost       | Modeling & evaluation                   |
| EDA           | pandas, seaborn, matplotlib | Data exploration                        |
| File Handling | Git LFS, `os`, `gdown`      | Large file management, scripting        |
| Notebook Env  | Jupyter                     | EDA, experimentation                    |
| Optional      | MLflow, joblib              | Tracking & model serialization          |

---

## 💡 Notes & Ideas

- Incorporate time-based alert patterns (burst rate, session grouping)
- Explore feature selection techniques (correlation matrix, recursive elimination)
- Consider using entropy of destination ports or payload sizes
- Keep output human-readable (valid/invalid + % confidence)
- Optional extension: integrate with Suricata live output + web dashboard
- Stretch goal: publish results in a student cybersecurity research forum

---

git clone https://github.com/your-username/ml-ids-analyzer.git
cd ml-ids-analyzer
pip install -r requirements.txt

## 👥 Authors
- Alexander Zimpher @AlexZimpher
- Spencer Hendren @SnakAttack

---
