from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

print(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}: Start loading data")

# Carica il file CSV
df = pd.read_csv('X_train.csv')

# Separazione delle colonne delle feature e del target
X = df.drop(columns='health')
Y = df['health']

# Dividi il dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.95, random_state=0)

# Normalizzazione dei dati
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# instantiate the model
knn = KNeighborsClassifier(n_neighbors=10)

# fit the model to the training set
for _ in tqdm(range(100), desc="Training k-NN"):
    knn.fit(X_train, y_train)

# Salvataggio del modello
joblib.dump(knn, 'knn_model.pkl')

print(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}: Model saved")

print(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}: Start testing")

y_pred = knn.predict(X_test)

print(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}: Test completed")

print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

# print('Training set score: {:.4f}'.format(knn.score(X_train, y_train)))

# print('Test set score: {:.4f}'.format(knn.score(X_test, y_test)))

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0, 0])

print('\nTrue Negatives(TN) = ', cm[1, 1])

print('\nFalse Positives(FP) = ', cm[0, 1])

print('\nFalse Negatives(FN) = ', cm[1, 0])

print(classification_report(y_test, y_pred))
