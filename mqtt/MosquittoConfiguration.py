import os
from dotenv import load_dotenv


class MosquittoConfiguration:
    def __init__(self):
        load_dotenv()

        self.broker_hostname = os.getenv("MOSQUITTO_HOST")
        self.port = int(os.getenv("MOSQUITTO_PORT"))
        self.username = os.getenv("MOSQUITTO_USERNAME")
        self.password = os.getenv("MOSQUITTO_PASSWORD")
        self.work_topic = os.getenv("MQTT_EDGE_WORK_TOPIC")
        self.stop_topic = os.getenv("MQTT_EDGE_REQUEST_STOP_TOPIC")
        self.acquisition_topic = os.getenv("MQTT_ACQUISITION_DATA_TOPIC")
