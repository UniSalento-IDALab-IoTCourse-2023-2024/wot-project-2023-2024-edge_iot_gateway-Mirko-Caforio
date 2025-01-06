import logging
import threading

import joblib
import numpy as np
from datetime import datetime, timezone
from time import sleep
import pandas as pd

from anomalyDetection.featureExtraction import FeatureExtractor
from mqtt.MosquittoConfiguration import MosquittoConfiguration
from mqtt.MosquittoPublisher import MosquittoPublisher


class DataGenerator:
    def __init__(self):
        self.config = MosquittoConfiguration()
        self.mosquittoPublisher = MosquittoPublisher(config=self.config)
        self.mosquittoPublisher.connect()
        self.data = []
        # self.real_data = pd.read_csv('../anomalyDetection/X_train.csv')
        self.real_data = pd.read_csv('anomalyDetection/val.csv')
        self.stats = self.real_data[['x', 'y', 'z']].agg(['mean', 'std']).transpose()
        self.window_size = 4096
        self.sample_rate = 1000.0
        self.machines = {}

        # self.model = joblib.load('..\\anomalyDetection\\isolation_forest_model.pkl')
        # self.model = joblib.load('../anomalyDetection/one_class_svm_model.pkl')
        self.model = joblib.load('anomalyDetection/random_forest_model.pkl')
        self.scaler = joblib.load('anomalyDetection/rf_scaler.pkl')

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def predict(self, datapoint):
        data = pd.DataFrame([datapoint])

        data = self.scaler.transform(data)
        prediction = self.model.predict(data)

        return prediction == 1

    def generate_data(self, machine_name):
        # Probabilità di generare un'anomalia
        anomaly_probability = 0.1  # 10% di probabilità di generare un'anomalia

        for i in range(self.window_size):
            if np.random.rand() < anomaly_probability:
                # Filtra i dati anomali
                anomaly_data = self.real_data[self.real_data['health'] == -1]
                stats = anomaly_data[['x', 'y', 'z']].agg(['mean', 'std']).transpose()
                self.logger.info("Anomaly generated")
            else:
                # Filtra i dati normali
                normal_data = self.real_data[self.real_data['health'] == 1]
                stats = normal_data[['x', 'y', 'z']].agg(['mean', 'std']).transpose()
                self.logger.info("Normal data generated")

            x_position = round(np.random.normal(stats.loc['x', 'mean'], stats.loc['x', 'std']), 1)
            y_position = round(np.random.normal(stats.loc['y', 'mean'], stats.loc['y', 'std']), 1)
            z_position = round(np.random.normal(stats.loc['z', 'mean'], stats.loc['z', 'std']), 1)

            data_point = {
                "x": x_position,
                "y": y_position,
                "z": z_position
            }

            self.data.append(data_point)
            sleep(1.0 / self.sample_rate)

        data_point = FeatureExtractor.extract_features(self.data)

        data_point["machine_name"] = machine_name
        data_point["timestamp"] = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        data_point["prediction"] = self.predict(data_point)

        return data_point

    def start(self, machine_name):
        if machine_name not in self.machines or not self.machines[machine_name]["running"]:
            self.machines[machine_name] = {"running": True}
            thread = threading.Thread(target=self.start_generating, args=(machine_name,))
            self.machines[machine_name]["thread"] = thread
            thread.start()
            self.logger.info(f"Data generation started for {machine_name}")
        else:
            self.logger.info(f"Data generation for {machine_name} is already running.")

    def stop(self, machine_name):
        if machine_name in self.machines and self.machines[machine_name]["running"]:
            self.machines[machine_name]["running"] = False
            self.machines[machine_name]["thread"].join()
            self.logger.info(f"Data generation stopped for {machine_name}")
        else:
            self.logger.info(f"No active data generation found for {machine_name}.")

    def start_generating(self, machine_name):
        while self.machines[machine_name]["running"]:
            data_point = self.generate_data(machine_name)
            self.logger.info(data_point)
            self.mosquittoPublisher.publish(data_point, self.config.acquisition_topic)
            self.logger.info(f"Data point for {machine_name} published.")

        # Cleanup dopo l'interruzione
        self.machines[machine_name]["thread"] = None
        self.logger.info(f"Thread per {machine_name} terminato.")
