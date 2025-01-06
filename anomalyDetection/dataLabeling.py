import os
import pandas as pd


def load_data(machine, operation):
    good_path = f'..\\datasets\\DATASET_CNC_BOSH\\{machine}\\{operation}\\good'
    bad_path = f'..\\datasets\\DATASET_CNC_BOSH\\{machine}\\{operation}\\bad'

    # Controlla se le directory esistono
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

    data_frames = []

    # Carica i file good e aggiungi la colonna health
    for file in good_files:
        print(f"Reading good file: {file}")
        df = pd.read_csv(file)
        df['health'] = 1
        data_frames.append(df)
        save_path = file  # Mantiene la stessa struttura di cartelle
        df.to_csv(save_path, index=False)
        print(f"Saved good file with health column: {save_path}")

    # Carica i file bad e aggiungi la colonna health
    for file in bad_files:
        print(f"Reading bad file: {file}")
        df = pd.read_csv(file)
        df['health'] = -1
        data_frames.append(df)
        save_path = file  # Mantiene la stessa struttura di cartelle
        df.to_csv(save_path, index=False)
        print(f"Saved bad file with health column: {save_path}")


# Esempio di utilizzo
machine = ['M01', 'M02', 'M03']
operations = [f'OP{i:02}' for i in range(15)]

for m in machine:
    for op in operations:
        print(f"Processing machine: {m}, operation: {op}")
        load_data(m, op)
print("Script execution completed.")
