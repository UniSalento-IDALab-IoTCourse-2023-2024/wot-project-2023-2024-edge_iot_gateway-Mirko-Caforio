# Python Node - IoT Gateway of SENTINEL

The Python node of the **SENTINEL** project acts as an IoT gateway, managing communication between the simulated industrial devices and the backend via the MQTT protocol. This component is crucial for collecting, processing and forwarding data from sensors, enabling real-time monitoring of machinery operations.

---

## Main functionalities

### MQTT communication
- **Subscription**: The node subscribes to specific MQTT topics to receive data sent from industrial machinery.
- **Publication**: Sends messages to the backend on specific topics to notify events or anomalies detected.

### Data pre-processing
- **Filtering**: Performs an initial analysis of the collected data to remove noise or inconsistent values.
- **Conversion**: Transforms raw data into a standardised format before sending it to the backend.

### Anomaly handling
- **Local detection**: Integrates anomaly detection algorithms to quickly detect suspicious behaviour without waiting for the backend.
- **Instant notification**: In case of anomalies, sends notifications to the backend for processing and visualisation.

---

## Prerequisites

### Software
- **Python**: Version 3.12 or higher.
- **Docker**: For running the node in a Docker container.

### Hardware
- **Raspberry Pi 4 or higher**: Used to run the node and interface with sensors.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/UniSalento-IDALab-IoTCourse-2023-2024/wot-project-2023-2024-edge_iot_gateway-Mirko-Caforio.git
   docker compose up -d
   ```