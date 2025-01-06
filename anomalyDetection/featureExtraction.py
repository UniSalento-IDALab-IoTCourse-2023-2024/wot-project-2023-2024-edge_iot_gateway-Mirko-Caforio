import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew


class FeatureExtractor:
    def __init__(self, csv, window_size, overlap):
        self.df = pd.read_csv(csv)
        self.window_size = window_size
        self.overlap = overlap
        self.step = int(window_size * (1 - overlap))
        self.W = len(self.df)
        self.phi = (self.W - window_size) / self.step
        self.p = int(np.floor(self.phi)) + (1 if np.floor(self.phi) == self.phi else 2)
        print(f"Initialized FeatureExtractor with {self.step} step")
        print(f"Initialized FeatureExtractor with {self.W} W")
        print(f"Initialized FeatureExtractor with {self.phi} phi")
        print(f"Initialized FeatureExtractor with {self.p} windows")

    @staticmethod
    def extract_features(window):
        features = {
            'mean_x': np.mean(window[:, 0]),
            'mean_y': np.mean(window[:, 1]),
            'mean_z': np.mean(window[:, 2]),
            'rms_x': np.sqrt(np.mean(window[:, 0] ** 2)),
            'rms_y': np.sqrt(np.mean(window[:, 1] ** 2)),
            'rms_z': np.sqrt(np.mean(window[:, 2] ** 2)),
            'min_x': np.min(window[:, 0]),
            'min_y': np.min(window[:, 1]),
            'min_z': np.min(window[:, 2]),
            'max_x': np.max(window[:, 0]),
            'max_y': np.max(window[:, 1]),
            'max_z': np.max(window[:, 2]),
            'std_x': np.std(window[:, 0]),
            'std_y': np.std(window[:, 1]),
            'std_z': np.std(window[:, 2]),
            'kurtosis_x': kurtosis(window[:, 0]),
            'kurtosis_y': kurtosis(window[:, 1]),
            'kurtosis_z': kurtosis(window[:, 2]),
            'skewness_x': skew(window[:, 0]),
            'skewness_y': skew(window[:, 1]),
            'skewness_z': skew(window[:, 2]),
            'peak_to_peak_x': np.ptp(window[:, 0]),
            'peak_to_peak_y': np.ptp(window[:, 1]),
            'peak_to_peak_z': np.ptp(window[:, 2]),
            'energy_x': np.sum(window[:, 0] ** 2),
            'energy_y': np.sum(window[:, 1] ** 2),
            'energy_z': np.sum(window[:, 2] ** 2)
        }

        return features

    def process_all_data(self):
        results = []
        data = self.df[['x', 'y', 'z']].values
        labels = self.df['health'].values

        for i in range(0, (self.p - 1) * self.step + 1, self.step):
            window = data[i:i + self.window_size]
            window_labels = labels[i:i + self.window_size]
            # Usa l'etichetta dominante nella finestra
            window_label = 1 if np.sum(window_labels == 1) > np.sum(window_labels == -1) else -1
            features = self.extract_features(window)
            features['health'] = window_label
            results.append(features)
            print(f"Processed window {i // self.step + 1}/{self.p}")

        print("Feature extraction completed for all windows")

        return pd.DataFrame(results)

    @staticmethod
    def toCsv(filename, extracted_features):
        extracted_features.to_csv(filename, index=False)


if __name__ == "__main__":
    featureExtractor = FeatureExtractor(csv='test.csv', window_size=4096, overlap=0.75)
    extracted_features = featureExtractor.process_all_data()
    featureExtractor.toCsv('extracted_test_PROVA_Y.csv', extracted_features)
    print(f"Feature extraction completata. File salvato: extracted_val_op.csv")
