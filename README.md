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
![image](https://github.com/user-attachments/assets/e9960899-8304-4053-b067-3fdf32a27516)

## :rocket: Current To-Do List (Post-Baseline & Graph Up)

### 1. Model Improvement
- [ ] **Hyperparameter Search**  
  - Run `GridSearchCV` or `RandomizedSearchCV` over key RF params (`n_estimators`, `max_depth`, `min_samples_leaf`) with `n_jobs=-1` and `verbose=1`
- [ ] **Threshold Tuning**  
  - Plot Precision–Recall curve  
  - Select an optimal probability cutoff to balance false positives vs. false negatives
- [ ] **Class-Weight Experiments**  
  - Test `class_weight` adjustments (e.g. `{0:1, 1:2}`) or SMOTE/undersampling

### 2. Feature Engineering & Selection
- [ ] **Derive New Features**  
  - Burst rate (alerts per second)  
  - Session duration (first→last packet timestamp)  
  - Entropy of destination ports and packet sizes  
  - Protocol-specific flag counts (e.g. TCP SYN)
- [ ] **Automated Selection**  
  - Generate correlation heatmap to drop collinear features  
  - Apply Recursive Feature Elimination (RFE) to isolate top predictors

### 3. End-to-End Demo
- [ ] **Demo Notebook** (`notebooks/03_demo.ipynb`)  
  - Load saved model+scaler and run inference on sample alerts  
  - Include inline confusion matrix, ROC curve, and PR curve
- [ ] **Slide Deck**  
  - 10–12 slides covering Purpose → Data → Methods → Results → Next Steps  
  - Embed feature-importance and threshold trade-off visuals

### 4. Documentation & Release
- [ ] **README.md**  
  - Quickstart: clone → install → `python preprocess.py` → `python -m src.model` → `python -m src.predict`  
  - Instructions for Git LFS or release asset for `cicids2017_clean.csv`
- [ ] **GitHub Release v1.0**  
  - Tag stable commit and attach cleaned CSV if not in LFS

### 5. Optional “Stand-Out” Extensions
- [ ] **Real-Time Integration**  
  - Stream Suricata EVE JSON into `predict.py` for live inference  
- [ ] **Explainability**  
  - Integrate SHAP or LIME to visualize per-alert feature contributions  
- [ ] **Deployment**  
  - Dockerize the pipeline with `docker-compose.yml` or Helm chart  
- [ ] **User Interface**  
  - Build a simple Streamlit/Flask dashboard for alert browsing, threshold adjustment, and export  

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
