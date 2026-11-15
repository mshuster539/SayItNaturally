import pandas as pd
from pathlib import Path

# Find the main project folder
project_folder = Path(__file__).resolve().parent.parent

# Define the data folder
data_folder = project_folder / "data"


# Load the formal dataset
formal = pd.read_csv(
    data_folder / "ru-en.formal.tsv",
    sep="\t",
    header=None,
    names=["Russian", "English"]
)

# Load the informal dataset
informal = pd.read_csv(
    data_folder / "ru-en.informal.tsv",
    sep="\t",
    header=None,
    names=["Russian", "English"]
)

# Load my own examples
custom = pd.read_csv(data_folder / "my_own_examples.csv")
