import pandas as pd

# Load the formal dataset
formal = pd.read_csv(
    "data/ru-en.formal.tsv",
    sep="\t",
    header=None,
    names=["Russian", "English"],
    on_bad_lines="skip"
)

# Load the informal dataset
informal = pd.read_csv(
    "data/ru-en.informal.tsv",
    sep="\t",
    header=None,
    names=["Russian", "English"],
    on_bad_lines="skip"
)

# Check both datasets
for name, data in [("FORMAL", formal), ("INFORMAL", informal)]:
    print(f"\n--- {name} DATASET ---")
    print("Number of rows:", len(data))
    print("Missing values:")
    print(data.isnull().sum())
    print("Duplicate rows:", data.duplicated().sum())


# Remove rows with missing translations
formal = formal.dropna(subset=["Russian", "English"])
informal = informal.dropna(subset=["Russian", "English"])

# Remove duplicate translations
formal = formal.drop_duplicates()
informal = informal.drop_duplicates()

# Display final dataset sizes
print("\nAFTER CLEANING")
print("Formal:", formal.shape)
print("Informal:", informal.shape)

# Save cleaned datasets
formal.to_csv("data/formal_cleaned.csv", index=False)
informal.to_csv("data/informal_cleaned.csv", index=False)

print("\nCleaned datasets saved successfully!")