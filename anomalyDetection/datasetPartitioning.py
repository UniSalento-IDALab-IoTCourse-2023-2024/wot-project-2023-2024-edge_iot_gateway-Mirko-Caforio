import os
import pandas as pd
from sklearn.model_selection import train_test_split


# Function to load data from the original dataset
def load_data(base_path):
    data = []
    for machine in os.listdir(base_path):  # Explore machines (M01, M02, M03)
        machine_path = os.path.join(base_path, machine)
        for op in os.listdir(machine_path):  # Explore operations (OP01, OP02, ...)
            op_path = os.path.join(machine_path, op)
            for label in ["good", "bad"]:  # Explore "good" and "bad" folders
                label_path = os.path.join(op_path, label)
                if not os.path.exists(label_path):
                    continue
                for file in os.listdir(label_path):  # Read CSV files
                    file_path = os.path.join(label_path, file)
                    print(f"Reading file: {file_path}")
                    df = pd.read_csv(file_path)
                    df['Machine'] = machine
                    df['Operation'] = op
                    df['Label'] = label
                    data.append(df)
    print("Data loading completed.")
    return pd.concat(data)


# Partition by machine
def partition_by_machine(data):
    machines = data['Machine'].unique()
    print(f"Machines found: {machines}")
    train_machines, test_machine = train_test_split(machines, test_size=1, random_state=42)
    validate_machine = train_machines[0]
    train_machines = train_machines[1:]

    print(f"Training machines: {train_machines}")
    print(f"Validation machine: {validate_machine}")
    print(f"Test machine: {test_machine}")

    train_data = data[data['Machine'].isin(train_machines)]
    validate_data = data[data['Machine'] == validate_machine]
    test_data = data[data['Machine'] == test_machine[0]]  # Ensure test_machine is a single value

    print("Data partitioning completed.")
    return train_data, validate_data, test_data


# Save partitions
def save_partitions(train, validate, test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    validate_path = os.path.join(output_dir, "validate.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train.to_csv(train_path, index=False)
    print(f"Training data saved to {train_path}")
    validate.to_csv(validate_path, index=False)
    print(f"Validation data saved to {validate_path}")
    test.to_csv(test_path, index=False)
    print(f"Test data saved to {test_path}")


# Execution
base_path = "..\\datasets\\DATASET_CNC_BOSH"
output_dir = "."
print("Starting data loading...")
data = load_data(base_path)
print("Starting data partitioning...")
train_data, validate_data, test_data = partition_by_machine(data)
print("Starting data saving...")
save_partitions(train_data, validate_data, test_data, output_dir)
print("Script execution completed.")
