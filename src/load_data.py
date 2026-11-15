
import pandas as pd

print("Loading formal dataset...")
formal = pd.read_csv(
    "data/ru-en.formal.tsv",
    sep="\t",
    header=None,
    names=["Russian", "English"]
)

print("Loading informal dataset...")
informal = pd.read_csv(
    "data/ru-en.informal.tsv",
    sep="\t",
    header=None,
    names=["Russian", "English"]
)

print("Loading custom dataset...")
custom = pd.read_csv("data/my_own_examples.csv")

print("All datasets loaded successfully!")
