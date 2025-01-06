import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, matthews_corrcoef, \
    cohen_kappa_score


class TestPhase:
    def __init__(self):
        # Caricamento del modello
        self.model = joblib.load('random_forest_model.pkl')
        self.scaler = joblib.load('rf_scaler.pkl')

    def predict(self, data):
        # Bilanciamento del dataset
        class_1 = data[data['health'] == 1]
        class_minus1 = data[data['health'] == -1]

        # Riduci la classe dominante per eguagliare il numero di campioni della classe minoritaria
        if len(class_1) > len(class_minus1):
            class_0_balanced = class_1.sample(n=len(class_minus1), random_state=42)
            balanced_df = pd.concat([class_0_balanced, class_minus1], axis=0)
        else:
            class_1_balanced = class_minus1.sample(n=len(class_1), random_state=42)
            balanced_df = pd.concat([class_1, class_1_balanced], axis=0)

        # Shuffle dei dati bilanciati
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        balanced_labels = balanced_df['health']
        balanced_df = balanced_df.drop(columns=['health'])
        """y_data = data['health']
        data = data.drop(columns=['health'])"""

        normalized_data = self.scaler.transform(balanced_df)

        # Predizione
        predictions = self.model.predict(normalized_data)

        return balanced_labels, predictions


if __name__ == "__main__":
    # Caricamento dei dati di esempio
    data = pd.read_csv('extracted_test_machine_2048.csv')
    #y_data = data['health']
    #data = data.drop(columns=['health'])

    # Creazione dell'istanza della classe TestPhase
    test_phase = TestPhase()

    # Predizione sui dati di esempio
    y_data, predictions = test_phase.predict(data)

    # Calcolo delle metriche
    print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_data, predictions)))
    print('AUC-ROC score: {0:0.4f}'.format(roc_auc_score(y_data, predictions)))

    # Matrice di confusione
    cm = confusion_matrix(y_data, predictions)
    tn, fp, fn, tp = cm.ravel()

    print('Confusion matrix\n\n', cm)
    print('\nTrue Positives(TP) = ', cm[0, 0])
    print('\nTrue Negatives(TN) = ', cm[1, 1])
    print('\nFalse Positives(FP) = ', cm[0, 1])
    print('\nFalse Negatives(FN) = ', cm[1, 0])

    # Calcolo della Specificity
    specificity = tn / (tn + fp)

    # Calcolo dell'MCC
    mcc = matthews_corrcoef(y_data, predictions)

    # Calcolo del Kappa value
    kappa = cohen_kappa_score(y_data, predictions)

    # Stampa del classification report
    print(classification_report(y_data, predictions))

    # Stampa delle metriche aggiuntive
    print(f"Specificity: {specificity:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Kappa value: {kappa:.4f}")
