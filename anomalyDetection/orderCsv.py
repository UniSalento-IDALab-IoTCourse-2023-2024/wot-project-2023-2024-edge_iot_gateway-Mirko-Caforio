import pandas as pd

# Load the CSV file
df = pd.read_csv('val.csv')

# Sort the rows by the 'health' column
df_sorted = df.sort_values(by='health', ascending=False)

# Save the sorted data to a new CSV file
df_sorted.to_csv('val.csv', index=False)

print("Rows sorted and saved to 'sorted_output.csv'")
