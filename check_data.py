import pandas as pd

# Load the dataset
data = pd.read_csv("dataset/creditcard.csv")
# Display basic information
print("Dataset loaded successfully!")
print("Number of rows and columns:", data.shape)

print("\nFirst 5 transactions:")
print(data.head())

print("\nColumn names:")
print(data.columns.tolist())

print("\nFraud and Normal transaction count:")
print(data["Class"].value_counts())