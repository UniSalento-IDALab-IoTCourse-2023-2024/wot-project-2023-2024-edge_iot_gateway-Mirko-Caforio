import os
import pandas as pd
from sklearn.model_selection import train_test_split


# Function to load data from a specific machine and operation
def load_data(machine, operation):
    good_path = f'..\\datasets\\DATASET_CNC_BOSH\\{machine}\\{operation}\\good'
    bad_path = f'..\\datasets\\DATASET_CNC_BOSH\\{machine}\\{operation}\\bad'

    # Check if directories exist
    if os.path.exists(good_path):
        good_directory = os.listdir(good_path)
        good_files = [os.path.join(good_path, file) for file in good_directory if file.endswith('.csv')]
    else:
        good_files = []

    if os.path.exists(bad_path):
        bad_directory = os.listdir(bad_path)
        bad_files = [os.path.join(bad_path, file) for file in bad_directory if file.endswith('.csv')]
    else:
        bad_files = []

    good_data = [pd.read_csv(file) for file in good_files]
    bad_data = [pd.read_csv(file) for file in bad_files]

    good_labels = [1] * sum(len(df) for df in good_data)  # Label 1 for normal data
    bad_labels = [-1] * sum(len(df) for df in bad_data)  # Label -1 for anomalous data

    data = good_data + bad_data
    labels = good_labels + bad_labels

    if data:
        return pd.concat(data, ignore_index=True), labels
    else:
        return pd.DataFrame(), labels


# Load data for each machine
def load_machine_data(machine):
    operations = [f'OP{i:02}' for i in range(15)]
    machine_data = []
    machine_labels = []

    for op in operations:
        print(f"Loading data for machine: {machine}, operation: {op}")
        data, labels = load_data(machine, op)
        machine_data.append(data)
        machine_labels.extend(labels)
        print(f"Loaded {len(data)} rows and {len(labels)} labels for operation: {op}")

    return pd.concat(machine_data, ignore_index=True), machine_labels


# Function to split data into training, validation, and test sets
def split_data(X_M01, y_M01, X_M02, y_M02, X_M03, y_M03):
    # Ensure the lengths of the data and labels are consistent
    assert len(X_M01) == len(y_M01), f"Inconsistent lengths for M01: {len(X_M01)} data rows, {len(y_M01)} labels"
    assert len(X_M02) == len(y_M02), f"Inconsistent lengths for M02: {len(X_M02)} data rows, {len(y_M02)} labels"
    assert len(X_M03) == len(y_M03), f"Inconsistent lengths for M03: {len(X_M03)} data rows, {len(y_M03)} labels"

    # Split data for training, validation, and test sets
    X_train_M01, X_temp_M01, y_train_M01, y_temp_M01 = train_test_split(X_M01, y_M01, test_size=0.3, random_state=42, stratify=y_M01)
    X_val_M01, X_test_M01, y_val_M01, y_test_M01 = train_test_split(X_temp_M01, y_temp_M01, test_size=0.3333, random_state=42, stratify=y_temp_M01)

    X_train_M02, X_temp_M02, y_train_M02, y_temp_M02 = train_test_split(X_M02, y_M02, test_size=0.3, random_state=42, stratify=y_M02)
    X_val_M02, X_test_M02, y_val_M02, y_test_M02 = train_test_split(X_temp_M02, y_temp_M02, test_size=0.5, random_state=42, stratify=y_temp_M02)

    # Combine the splits to form the final datasets
    X_train = pd.concat([X_train_M01, X_train_M02], ignore_index=True)
    y_train = y_train_M01 + y_train_M02

    X_val = pd.concat([X_val_M01, X_val_M02], ignore_index=True)
    y_val = y_val_M01 + y_val_M02

    X_test = pd.concat([X_test_M01, X_test_M02, X_M03], ignore_index=True)
    y_test = y_test_M01 + y_test_M02 + y_M03

    return X_train, y_train, X_val, y_val, X_test, y_test


# Load data for all machines
print("Loading data for machine M01...")
X_M01, y_M01 = load_machine_data('M01')
print(f"Data for machine M01: {X_M01.shape[0]} rows")

print("Loading data for machine M02...")
X_M02, y_M02 = load_machine_data('M02')
print(f"Data for machine M02: {X_M02.shape[0]} rows")

print("Loading data for machine M03...")
X_M03, y_M03 = load_machine_data('M03')
print(f"Data for machine M03: {X_M03.shape[0]} rows")

# Split the data
X_train, y_train, X_val, y_val, X_test, y_test = split_data(X_M01, y_M01, X_M02, y_M02, X_M03, y_M03)

# Save training, validation, and test data to CSV files
X_train.to_csv('train.csv', index=False)
print("Training data saved to train.csv")

X_val.to_csv('val.csv', index=False)
print("Validation data saved to val.csv")

X_test.to_csv('test.csv', index=False)
print("Test data saved to test.csv")

print("Script execution completed.")
