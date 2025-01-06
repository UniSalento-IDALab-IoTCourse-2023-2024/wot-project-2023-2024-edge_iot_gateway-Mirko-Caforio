import logging
import paho.mqtt.client as mqtt

from generator.dataGenerator import DataGenerator
from mqtt.MosquittoConfiguration import MosquittoConfiguration


class MosquittoSubscriber:
    def __init__(self, config):
        self.client = mqtt.Client(client_id="edgeSubscriber", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.client.username_pw_set(username=config.username, password=config.password)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.failed_connect = False
        self.broker_hostname = config.broker_hostname
        self.port = config.port
        self.topic_work = config.work_topic
        self.topic_stop = config.stop_topic
        self.data_generator = DataGenerator()

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self.logger.info("MosquittoSubscriber connected to broker")
            self.client.subscribe([
                (self.topic_work, 0),
                (self.topic_stop, 0)
            ], 1)
            self.logger.info(f"Subscribed to topics: {self.topic_work}, {self.topic_stop}")
        else:
            self.logger.error(f"Could not connect, return code: {reason_code}")
            self.client.failed_connect = True

    def on_message(self, client, userdata, message):
        topic = message.topic
        machine_name = message.payload.decode("utf-8").strip('"')
        self.logger.info(f"Received message on topic: {topic} with payload: {machine_name}")

        if topic == self.topic_work:
            self.logger.info(f"Start generating data for {machine_name}...")
            self.start_data_generation(machine_name)
        elif topic == self.topic_stop:
            self.logger.info(f"Stop generating data for {machine_name}...")
            self.stop_data_generation(machine_name)

    def start_data_generation(self, machine_name):
        self.data_generator.start(machine_name)

    def stop_data_generation(self, machine_name):
        self.data_generator.stop(machine_name)

    def start(self):
        try:
            self.client.connect(self.broker_hostname, self.port)
            self.client.loop_forever()
        except Exception as e:
            self.logger.exception(f"Error in connecting to broker: {e}")


if __name__ == "__main__":
    config = MosquittoConfiguration()
    subscriber = MosquittoSubscriber(config)
    subscriber.start()
