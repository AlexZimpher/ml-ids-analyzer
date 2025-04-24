# ML-IDS-Analyzer

A machine learning-based system to predict the validity of intrusion detection system (IDS) alerts, built by cybersecurity students to assist in real-time alert triage and reduce false positives.

## 🚀 Project Goals
- Classify IDS alerts (e.g., from Suricata) as valid or false
- Reduce noise in SOC workflows using ML
- Provide explainable and reproducible results

## 📦 Features
- Supports labeled datasets like CICIDS2017
- Modular preprocessing, training, and evaluation
- Works with Suricata EVE JSON logs
- Binary classification with confidence score

## 📁 Structure
- `data/` – raw and processed IDS datasets
- `notebooks/` – EDA and modeling workflows
- `src/` – modular Python codebase
- `outputs/` – model predictions and reports

## 🔧 Setup

```bash
git clone https://github.com/your-username/ml-ids-analyzer.git
cd ml-ids-analyzer
pip install -r requirements.txt
