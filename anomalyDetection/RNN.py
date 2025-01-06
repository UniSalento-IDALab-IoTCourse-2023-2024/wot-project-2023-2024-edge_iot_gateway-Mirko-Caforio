import numpy as np
import pandas as pd
from keras import Sequential, Input
from keras.src.layers import SimpleRNN, Dense
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import compute_class_weight

# Caricamento del dataset
data_path = 'extracted_train_machine.csv'
df = pd.read_csv(data_path)

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
X = balanced_df.drop(columns=['health']).values
y = balanced_df['health'].values

# Normalizzazione dei dati
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aggiunta della dimensione temporale per la RNN
X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])  # (campioni, sequenze, feature)

# Suddivisione in training (80%) e test (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42, stratify=y)

# Configurazione della RNN
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),  # Input esplicito
    SimpleRNN(units=50, activation='relu'),
    Dense(units=2, activation='softmax')  # Softmax per output binario
])

# Compilazione del modello
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print(f"Class Weights: {class_weights_dict}")

# Training con pesi delle classi
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Valutazione del modello
y_pred = np.argmax(model.predict(X_test), axis=1)

# Calcolo delle metriche
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

print(f"AUC-ROC: {roc_auc_score(y_test, y_pred)}")

# Matrice di confusione
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0, 0])
print('\nTrue Negatives(TN) = ', cm[1, 1])
print('\nFalse Positives(FP) = ', cm[0, 1])
print('\nFalse Negatives(FN) = ', cm[1, 0])

print(classification_report(y_test, y_pred))

# Visualizzazione della matrice di confusione
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Classe 0", "Classe 1"],
            yticklabels=["Classe 0", "Classe 1"])
plt.xlabel("Valori Predetti")
plt.ylabel("Valori Reali")
plt.title("Confusion Matrix")
plt.show()

# Visualizzazione della curva di apprendimento
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Salvataggio del modello nel formato Keras
model.save('rnn_fault_detection_model.keras')
print("Modello salvato come rnn_fault_detection_model.keras")
