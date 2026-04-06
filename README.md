# 🏆 PulseGuard-AI

### Patient-Specific Anomaly Detection using Isolation Forest & AutoEncoder Ensemble

---

## 🚀 Overview

**PulseGuard-AI** is an advanced anomaly detection system designed to identify critical abnormalities in patient vital signals.

Built for a **Data Science Hackathon**, this solution focuses on maximizing:

* ROC-AUC (40%)
* PR-AUC (30%)
* F1 Score (15%)
* Recall (15%)

The model combines **machine learning + deep learning + patient-specific modeling** to deliver high-performance anomaly ranking.

---

## 🧠 Key Idea

Instead of using a single global model, PulseGuard-AI:

* Learns **individual patient baselines**
* Detects **deviations over time**
* Combines multiple models to improve ranking quality

---

## ⚙️ Architecture

### 🥇 Per-Patient Isolation Forest

* Trained separately for each patient
* Captures personalized anomaly patterns
* Strong contributor to ROC-AUC

### 🥈 AutoEncoder (Neural Network)

* Trained on normal data only
* Learns compressed representation of vitals
* High reconstruction error = anomaly

### 🏆 Final Ensemble

* Rank-based blending of models
* Optimized for leaderboard metrics

---

## 📊 Features Engineered

* `spo2_hr` → Combined oxygen-heart signal
* `shock_index` → HR / SBP (critical medical indicator)
* `pulse_pressure` → SBP - DBP
* Rolling statistics:

  * `mbp_rollmin`
  * `hr_rollmean`
  * `hr_std`
* Change detection:

  * `spo2_diff`
  * `hr_diff`
  * `mbp_diff`
* Patient-wise normalization (z-score)

---

## 🧪 Model Pipeline

1. Feature Engineering
2. Per-patient Isolation Forest
3. AutoEncoder training (normal data only)
4. Score normalization
5. Rank-based ensemble
6. Final anomaly score generation

---

## 📂 Project Structure

```
PulseGuard-AI/
│── train.csv
│── test.csv
│── main.py
│── submission_final.csv
│── README.md
```

---

## ▶️ How to Run

```bash
pip install numpy pandas scikit-learn torch
python main.py
```

---

## 📤 Submission Format

The final output must follow:

```
case_id | time_sec | anomaly_score
```

---

## 🏆 Results Strategy

To maximize leaderboard performance:

* Focus on **ranking quality (not classification)**
* Use **continuous anomaly scores**
* Tune ensemble weights:

  * 0.7 IF + 0.3 AE (recommended)
* Try multiple submissions with slight variations

---

## 💡 Key Learnings

* Patient-specific modeling > global models
* Simpler ensembles often outperform complex stacks
* Feature engineering is more important than model complexity
* PR-AUC is the hardest metric — optimize for it

---

## 🚀 Future Improvements

* Transformer-based time series modeling
* Temporal attention mechanisms
* Advanced pseudo-labeling with confidence weighting

---

## 👨‍💻 Author

**Sanidhya Gupta**
B.Tech CSE (AI & Data Science)
Poornima University

---

## ⭐ If you like this project

Give it a star ⭐ and share it!

---
