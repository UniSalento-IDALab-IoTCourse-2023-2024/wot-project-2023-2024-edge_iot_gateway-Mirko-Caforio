from datetime import datetime, timezone

import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDOneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, hinge_loss
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

print(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}: Start loading data")

# Carica il file CSV
df = pd.read_csv('X_train.csv')

# Separazione delle colonne delle feature e del target
X = df.drop(columns=['x', 'z', 'health'])
Y = df['health']

# Dividi il dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.90, random_state=42)

# Normalizzazione dei dati
scaler = StandardScaler()
#scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Seleziona le istanze non anomale
non_anomalous_X_train = X_train_scaled[y_train == 0]

# svm = SGDOneClassSVM(nu=0.05, shuffle=True, verbose=True)
svm = SGDOneClassSVM(nu=0.05, shuffle=True)

for _ in tqdm(range(100), desc="Training SDG One Class SVM"):
    svm.fit(non_anomalous_X_train)

# Salvataggio del modello
joblib.dump(svm, 'sdg_one_class_svm_model.pkl')
joblib.dump(scaler, 'sdg_one_class_svm_scaler.pkl')

print(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}: Model saved")

print(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}: Starting prediction")

current_y_pred = svm.predict(X_test_scaled)

# Impostare una soglia per classificare le predizioni
threshold = -0.5
current_y_pred = np.where(current_y_pred < threshold, 1, 0)

# Calcola l'AUC-ROC
current_auc_roc = roc_auc_score(y_test, current_y_pred)
loss = hinge_loss(y_test, svm.decision_function(X_test))

print("Hinge Loss {0:0.4f}: ".format(loss))

print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, current_y_pred)))

print('AUC-ROC: {0:0.4f}'.format(current_auc_roc))

cm = confusion_matrix(y_test, current_y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0, 0])

print('\nTrue Negatives(TN) = ', cm[1, 1])

print('\nFalse Positives(FP) = ', cm[0, 1])

print('\nFalse Negatives(FN) = ', cm[1, 0])

print(classification_report(y_test, current_y_pred))

# Visualizza la matrice di confusione utilizzando seaborn e matplotlib per l'SVM
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'],
            yticklabels=['Classe 0', 'Classe 1'])
plt.xlabel('Valori predetti')
plt.ylabel('Valori reali')
plt.title('Matrice di Confusione SVM')
plt.show()
