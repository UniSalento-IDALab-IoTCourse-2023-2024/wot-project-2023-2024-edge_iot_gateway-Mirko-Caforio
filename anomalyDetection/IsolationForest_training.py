from datetime import datetime, timezone
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, \
    matthews_corrcoef, cohen_kappa_score, accuracy_score
from tqdm import tqdm

print(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}: Start loading data")

X_train = pd.read_csv('X_train.csv')
X_val = pd.read_csv('X_val.csv')

# Separazione delle annotazioni prima della normalizzazione
X_train_data = X_train.drop(columns=['health'])
X_val_data = X_val.drop(columns=['health'])
y_train = X_train['health']
y_val = X_val['health']

# X_train = X_train.groupby('health', group_keys=False).apply(lambda x: x.sample(frac=0.5))

print(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}: Data loaded")

# Create a RandomUnderSampler object
rus = RandomUnderSampler(random_state=42, sampling_strategy='majority')

# Balancing the data
X_resampled, y_resampled = rus.fit_resample(X_train_data, y_train)

# Conteggio delle occorrenze di 0 e 1
count_0 = (y_resampled == 0).sum()
count_1 = (y_resampled == 1).sum()

# Stampa dei conteggi
print(f"Number of 0: {count_0}")
print(f"Number of 1: {count_1}")

# Normalizzazione dei dati
scaler = StandardScaler()
# scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(X_resampled)
X_val_scaled = scaler.transform(X_val_data)
# X_test_scaled = scaler.transform(X_test)

print(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}: Data normalized")
print(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}: Start training")

# Addestramento del modello
model = IsolationForest(n_estimators=500, contamination=0.5, max_samples="auto", random_state=42)

for _ in tqdm(range(100), desc="Training Isolation Forest"):
    model.fit(X_train_scaled)

print(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}: Training completed")

# Salvataggio del modello
joblib.dump(model, 'isolation_forest_model.pkl')

print(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}: Model saved")

"""# Predizione sui dati di test
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_val_scaled)

# Le predizioni di IsolationForest sono -1 per anomalie e 1 per normali
# Convertiamo queste predizioni per matchare le nostre etichette
y_pred_train = [0 if x == 1 else 1 for x in y_pred_train]
y_pred_test = [0 if x == 1 else 1 for x in y_pred_test]

# Valutazione delle performance
print(f"Train Accuracy: {accuracy_score(y_resampled, y_pred_train)}")
print(f"Test Accuracy: {accuracy_score(y_val, y_pred_test)}")
print(f"Precision: {precision_score(y_val, y_pred_test)}")
print(f"Recall: {recall_score(y_val, y_pred_test)}")
print(f"F1 Score: {f1_score(y_val, y_pred_test)}")
print(f"Matthews Correlation Coefficient: {matthews_corrcoef(y_val, y_pred_test)}")
print(f"Cohen Kappa Score: {cohen_kappa_score(y_val, y_pred_test)}")
print(f"Confusion Matrix:\n {confusion_matrix(y_val, y_pred_test)}")
print(f"Classification Report:\n {classification_report(y_val, y_pred_test)}")"""
