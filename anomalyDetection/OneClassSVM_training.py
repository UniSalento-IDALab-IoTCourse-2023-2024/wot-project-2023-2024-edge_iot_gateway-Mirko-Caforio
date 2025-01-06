from datetime import datetime, timezone

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, hinge_loss
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

print(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}: Start loading data")

# Carica il file CSV
df = pd.read_csv('X_train_MO1.csv')

# Separazione delle colonne delle feature e del target
X = df.drop(columns='health')
Y = df['health']

# Dividi il dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.90, random_state=42)

# Normalizzazione dei dati
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Seleziona le istanze non anomale
non_anomalous_X_train = X_train_scaled[y_train == 0]

svm = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale', shrinking=False, cache_size=800, verbose=True)

for _ in tqdm(range(100), desc="Training One Class SVM"):
    svm.fit(non_anomalous_X_train)

# Salvataggio del modello
joblib.dump(svm, 'one_class_svm_model.pkl')
joblib.dump(scaler, 'one_class_svm_scaler.pkl')

print(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}: Starting prediction")

current_y_pred = svm.predict(X_test_scaled)

# Impostare una soglia per classificare le predizioni
threshold = -0.5
current_y_pred = np.where(current_y_pred < threshold, 1, 0)

loss = hinge_loss(y_test, svm.decision_function(X_test))

print(f"Hinge Loss: {loss}")

print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, current_y_pred)))

print('AUC-ROC score: {0:0.4f}'.format(roc_auc_score(y_test, current_y_pred)))

print(classification_report(y_test, current_y_pred))

cm = confusion_matrix(y_test, current_y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0, 0])

print('\nTrue Negatives(TN) = ', cm[1, 1])

print('\nFalse Positives(FP) = ', cm[0, 1])

print('\nFalse Negatives(FN) = ', cm[1, 0])

# Visualizza la matrice di confusione utilizzando seaborn e matplotlib per l'SVM
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'],
            yticklabels=['Classe 0', 'Classe 1'])
plt.xlabel('Valori predetti')
plt.ylabel('Valori reali')
plt.title('Confusion Matrix OneClassSVM')
plt.show()
