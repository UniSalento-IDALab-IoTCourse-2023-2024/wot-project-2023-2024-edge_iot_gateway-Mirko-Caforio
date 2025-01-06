from datetime import datetime, timezone
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, matthews_corrcoef, \
    cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

print(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}: Start loading data")

# Carica il file CSV
df = pd.read_csv('extracted_train_machine_2048.csv')

# Bilanciamento del dataset
class_0 = df[df['health'] == 1]
class_1 = df[df['health'] == -1]

print(f"Class 0: {len(class_0)}")
print(f"Class 1: {len(class_1)}")

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
scaler = StandardScaler()
#scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configurazione del modello Random Forest
rf = RandomForestClassifier(
    n_estimators=1000,
    random_state=42,
    verbose=1
)

# Addestramento del modello Random Forest
rf.fit(X_train_scaled, y_train)

# Salvataggio del modello
joblib.dump(rf, 'random_forest_model.pkl')
joblib.dump(scaler, 'rf_scaler.pkl')

# Predizioni
current_y_pred = rf.predict(X_test_scaled)

# Valutazione del modello
print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, current_y_pred)))
print(f"AUC-ROC: {roc_auc_score(y_test, current_y_pred)}")

# Matrice di confusione
cm = confusion_matrix(y_test, current_y_pred)
tn, fp, fn, tp = cm.ravel()

print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0, 0])
print('\nTrue Negatives(TN) = ', cm[1, 1])
print('\nFalse Positives(FP) = ', cm[0, 1])
print('\nFalse Negatives(FN) = ', cm[1, 0])

# Calcolo della Specificity
specificity = tn / (tn + fp)

# Calcolo dell'MCC
mcc = matthews_corrcoef(y_test, current_y_pred)

# Calcolo del Kappa value
kappa = cohen_kappa_score(y_test, current_y_pred)

# Stampa del classification report
print(classification_report(y_test, current_y_pred))

# Stampa delle metriche aggiuntive
print(f"Specificity: {specificity:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"Kappa value: {kappa:.4f}")

# Visualizza la matrice di confusione
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'],
            yticklabels=['Classe 0', 'Classe 1'])
plt.xlabel('Valori predetti')
plt.ylabel('Valori reali')
plt.title('Confusion Matrix')
plt.show()

# Importanza delle feature
feature_importances = rf.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature importances:")
print(importance_df)

plt.figure(figsize=(8, 6))
sns.barplot(x=importance_df['Importance'], y=importance_df['Feature'])
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
