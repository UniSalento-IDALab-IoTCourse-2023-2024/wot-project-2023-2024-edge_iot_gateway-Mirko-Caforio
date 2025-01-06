import json
import logging
import paho.mqtt.client as mqtt


class MosquittoPublisher:
    def __init__(self, config):
        self.client = mqtt.Client(client_id="edgePublisher", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.username_pw_set(username=config.username, password=config.password)
        self.broker_hostname = config.broker_hostname
        self.port = config.port

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def on_connect(self, client, userdata, flags, reason_code, properties):
        if flags.session_present:
            self.logger.info("Session present")
        if reason_code == 0:
            self.logger.info("MosquittoPublisher connected to broker")
        else:
            self.logger.error(f"Could not connect, return code: {reason_code}")
            self.client.failed_connect = True
        if reason_code > 0:
            self.logger.error(f"Could not connect, return code: {reason_code}")

    def connect(self):
        try:
            self.client.connect(self.broker_hostname, self.port)
            self.client.loop_start()
        except Exception as e:
            self.logger.exception(f"Error in connecting to broker: {e}")

    def publish(self, data, topic):
        try:
            # TODO: VEDERE SE IMPOSTARE ANCHE QoS e RETAIN
            result = self.client.publish(topic, json.dumps(data), qos=0)
            status = result[0]
            if status == 0:
                self.logger.info(f"Message {json.dumps(data)} is published to topic {topic}")
            else:
                self.logger.error(f"Failed to send message to topic {topic}")
                if not self.client.is_connected():
                    self.logger.error("Client not connected, exiting...")
        except Exception as e:
            self.logger.exception(f"Error in publishing message: {e}")
