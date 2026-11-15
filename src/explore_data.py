
from pathlib import Path

# Find the two dataset files
dataset_files = [
    Path("data/ru-en.formal.tsv"),
    Path("data/ru-en.informal.tsv")
]

# Inspect each dataset
for file_path in dataset_files:

    print("\nDataset:", file_path.name)

    # Open the file and read its contents
    with open(file_path, "r", encoding="utf-8") as file:

        lines = file.readlines()

    print("Total rows:", len(lines))

    # Examine the first five rows
    print("\nFirst five rows:")

    for line in lines[:5]:
        print(repr(line))

    # Count how many rows contain different numbers of tabs
    tab_counts = {}

    for line in lines:
        number_of_tabs = line.count("\t")

        if number_of_tabs not in tab_counts:
            tab_counts[number_of_tabs] = 0

        tab_counts[number_of_tabs] += 1

    print("\nNumber of tabs per row:")
    print(tab_counts)

from pathlib import Path

for filename in ["ru-en.formal.tsv", "ru-en.informal.tsv"]:
    path = Path("data") / filename

    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    print(f"\nChecking {filename}")
    print("Line 531:")
    print(repr(lines[530]))
    print("Number of tabs:", lines[530].count("\t"))

