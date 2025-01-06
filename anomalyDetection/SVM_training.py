from datetime import datetime, timezone
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, hinge_loss
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

print(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}: Start loading data")

# Carica il file CSV
df = pd.read_csv('extracted_train_machine_4096.csv')

# Bilanciamento del dataset
class_0 = df[df['health'] == 1]
class_1 = df[df['health'] == -1]

# Riduci la classe dominante per eguagliare il numero di campioni della classe minoritaria
if len(class_0) > len(class_1):
    class_0_balanced = class_0.sample(n=len(class_1), random_state=42)
    balanced_df = pd.concat([class_0_balanced, class_1], axis=0)
else:
    class_1_balanced = class_1.sample(n=len(class_0), random_state=42)
    balanced_df = pd.concat([class_0, class_1_balanced], axis=0)

# Shuffle dei dati bilanciati
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Separazione delle feature e della label
X = balanced_df.drop(columns=['health'])
Y = balanced_df['health']

"""X = df.drop(columns=['health'])
Y = df['health']"""

# Dividi il dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.01, random_state=42, stratify=Y)

# Normalizzazione dei dati
scaler = MinMaxScaler(feature_range=(-1, 1))
#scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configurazione del modello SVC
svm = SVC(kernel='rbf', C=1, gamma='scale', shrinking=False, cache_size=2000, verbose=True)

# Addestramento del modello SVC
for _ in tqdm(range(100), desc="Training SVM"):
    svm.fit(X_train_scaled, y_train)

# Salvataggio del modello
joblib.dump(svm, 'svm_model.pkl')
joblib.dump(scaler, 'svm_scaler.pkl')

# Predizioni
current_y_pred = svm.predict(X_test_scaled)

# Valutazione del modello
loss = hinge_loss(y_test, svm.decision_function(X_test_scaled))
print(f"Hinge Loss: {loss}")
print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, current_y_pred)))
print(f"AUC-ROC: {roc_auc_score(y_test, current_y_pred)}")

# Matrice di confusione
cm = confusion_matrix(y_test, current_y_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0, 0])
print('\nTrue Negatives(TN) = ', cm[1, 1])
print('\nFalse Positives(FP) = ', cm[0, 1])
print('\nFalse Negatives(FN) = ', cm[1, 0])

print(classification_report(y_test, current_y_pred))

# Visualizza la matrice di confusione
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'],
            yticklabels=['Classe 0', 'Classe 1'])
plt.xlabel('Valori predetti')
plt.ylabel('Valori reali')
plt.title('Confusion Matrix')
plt.show()
