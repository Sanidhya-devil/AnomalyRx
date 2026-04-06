# =========================================
# 🔥 IMPORTS
# =========================================
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import rankdata

import torch
import torch.nn as nn
import torch.optim as optim

# =========================================
# 📂 LOAD DATA
# =========================================
train = pd.read_csv("train_vitals.csv")
test = pd.read_csv("test_vitals.csv")

# =========================================
# ⚙️ FEATURE ENGINEERING
# =========================================
def create_features(df):
    df = df.copy()
    
    df['spo2_hr'] = df['SpO2'] * df['HR']
    df['shock_index'] = df['HR'] / (df['MBP'] + 1)
    
    df['mbp_rollmin'] = df.groupby('case_id')['MBP'].transform(
        lambda x: x.rolling(30, min_periods=1).min()
    )
    
    df['hr_rollmean'] = df.groupby('case_id')['HR'].transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    )
    
    df['hr_std'] = df.groupby('case_id')['HR'].transform(
        lambda x: x.rolling(10, min_periods=1).std()
    )
    
    df['spo2_diff'] = df.groupby('case_id')['SpO2'].diff().fillna(0)
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df

train = create_features(train)
test = create_features(test)

features = [c for c in train.columns if c not in ['case_id', 'time_sec']]

# =========================================
# 🥇 PER-PATIENT ISOLATION FOREST
# =========================================
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train[features])
test_scaled = scaler.transform(test[features])

train_scaled = pd.DataFrame(train_scaled, columns=features)
test_scaled = pd.DataFrame(test_scaled, columns=features)

train_scaled['case_id'] = train['case_id']
test_scaled['case_id'] = test['case_id']

if_scores = np.zeros(len(test))

for pid in train_scaled['case_id'].unique():
    tr = train_scaled[train_scaled['case_id'] == pid][features]
    te = test_scaled[test_scaled['case_id'] == pid][features]
    
    if len(tr) < 20 or len(te) == 0:
        continue
    
    model = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42
    )
    
    model.fit(tr)
    scores = -model.decision_function(te)
    
    idx = test_scaled[test_scaled['case_id'] == pid].index
    if_scores[idx] = scores

if_scores = MinMaxScaler().fit_transform(if_scores.reshape(-1,1)).ravel()
if_rank = rankdata(if_scores) / len(if_scores)

# =========================================
# 🥈 AUTOENCODER
# =========================================
train_clean = train.copy()

X_train = train_clean[features].values
X_test = test[features].values

scaler_ae = StandardScaler()
X_train = scaler_ae.fit_transform(X_train)
X_test = scaler_ae.transform(X_test)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

class AutoEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

model = AutoEncoder(X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(40):
    optimizer.zero_grad()
    out = model(X_train)
    loss = criterion(out, X_train)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    recon = model(X_test)
    ae_scores = torch.mean((X_test - recon) ** 2, dim=1).numpy()

ae_scores = MinMaxScaler().fit_transform(ae_scores.reshape(-1,1)).ravel()
ae_rank = rankdata(ae_scores) / len(ae_scores)

# =========================================
# 🥉 LOF
# =========================================
lof = LocalOutlierFactor(n_neighbors=25, contamination=0.05)
lof_scores = -lof.fit_predict(test[features])
lof_scores = MinMaxScaler().fit_transform(lof_scores.reshape(-1,1)).ravel()
lof_rank = rankdata(lof_scores) / len(lof_scores)

# =========================================
# 🧠 INITIAL ENSEMBLE
# =========================================
base_scores = (
    0.5 * if_rank +
    0.3 * ae_rank +
    0.2 * lof_rank
)

# =========================================
# 🔥 FINAL HACK: PSEUDO LABELING
# =========================================
top_idx = base_scores > np.percentile(base_scores, 98)
low_idx = base_scores < np.percentile(base_scores, 2)

pseudo = test.copy()
pseudo['label'] = -1
pseudo.loc[top_idx, 'label'] = 1
pseudo.loc[low_idx, 'label'] = 0
pseudo = pseudo[pseudo['label'] != -1]

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(pseudo[features], pseudo['label'])

rf_scores = clf.predict_proba(test[features])[:,1]
rf_rank = rankdata(rf_scores) / len(rf_scores)

# =========================================
# 🏆 FINAL SUPER ENSEMBLE
# =========================================
final_scores = (
    0.4 * if_rank +
    0.3 * ae_rank +
    0.2 * lof_rank +
    0.1 * rf_rank
)

# =========================================
# ⚡ SMOOTHING (PER PATIENT)
# =========================================
test['final'] = final_scores

test['final'] = test.groupby('case_id')['final'].transform(
    lambda x: x.rolling(5, min_periods=1).mean()
)

final_scores = test['final'].values

# =========================================
# 📤 FINAL SUBMISSION
# =========================================
submission = pd.DataFrame({
    'case_id': test['case_id'],
    'time_sec': test['time_sec'],
    'anomaly_score': final_scores
})

submission.to_csv("submission_final.csv", index=False)

print("🏆 FINAL SUBMISSION READY — GO WIN THIS HACKATHON")