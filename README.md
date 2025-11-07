# 🧠 GitHub Activity Signal — Developer Activeness Analyzer

An intelligent web app that predicts **how active a developer is on GitHub** using real-time profile and repository data.

### 🚀 Live Demo
🔗 [Streamlit App](https://app-activity-analyzer-ryszds8fhteqky776w6xe9.streamlit.app/)

---

## 🌟 Overview
This project uses **machine learning** to analyze GitHub developer activity.  
It predicts whether a developer is **Active**, **Moderate**, or **Inactive**, based on their public repositories, recent pushes, stars, forks, and followers.

The goal:  
💼 Help **recruiters** quickly assess developer activeness from GitHub data.

---

## 🧩 Features
- 📊 Fetches real GitHub data (via GitHub API)
- 🧠 Predicts developer activeness using a trained **XGBoost model (96% accuracy)**
- ⚙️ Performs automatic **feature engineering**
- 📝 Generates a **recruiter-friendly summary note**
- 🌐 Fully deployed on **Streamlit Cloud**

---

## 🧮 How It Works

### 🔹 Data Features
| Feature | Description |
|----------|--------------|
| followers | Number of GitHub followers |
| avg_stars | Average stars per repo |
| avg_forks | Average forks per repo |
| repo_count | Total repositories owned |
| days_since_recent_push | Recency of last code push |
| account_age_days | Days since account creation |
| popularity_score | Weighted score of stars + followers |
| signal_score | Weighted signal from pushes, forks, and repo activity |

### 🔹 Model
- Algorithm: **XGBoost Classifier**
- Accuracy: **96.2%**
- F1 (macro): **0.94**
- Trained on ~800 profiles labeled by activity

---

## ⚙️ Tech Stack
- **Python**
- **Streamlit**
- **Pandas / NumPy**
- **Scikit-learn**
- **XGBoost**
- **Requests (GitHub API)**

---

## 🧠 Results
| Metric | Value |
|--------|--------|
| Accuracy | 0.962 |
| Precision (macro) | 0.951 |
| Recall (macro) | 0.942 |
| F1 (macro) | 0.946 |

---

## 💻 Run Locally
```bash
git clone https://github.com/sriharizz/github-activity-analyzer.git
cd github-activity-analyzer
pip install -r requirements.txt
streamlit run app.py
# github-activity-analyzer

