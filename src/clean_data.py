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

# Remove rows with missing Russian sentences or English translations
formal = formal.dropna(subset=["Russian", "English"])
informal = informal.dropna(subset=["Russian", "English"])

# Remove duplicate sentence pairs
formal = formal.drop_duplicates()
informal = informal.drop_duplicates()

# Save the cleaned datasets as CSV files
formal.to_csv("data/formal_cleaned.csv", index=False)
informal.to_csv("data/informal_cleaned.csv", index=False)
