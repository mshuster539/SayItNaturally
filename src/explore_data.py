from pathlib import Path

# Find the main project folder
project_folder = Path(__file__).resolve().parent.parent

# Identify the formal and informal datasets
dataset_files = [
    project_folder / "data" / "ru-en.formal.tsv",
    project_folder / "data" / "ru-en.informal.tsv"
]

# Inspect each dataset
for file_path in dataset_files:

    print("\nDataset:", file_path.name)

    # Read the dataset
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Count the total number of rows
    print("Total rows:", len(lines))

    # Display the first five rows to inspect the data
    print("\nFirst five rows:")

    for line in lines[:5]:
        print(repr(line))

    # Count the number of tabs in each row
    tab_counts = {}

    for line in lines:
        number_of_tabs = line.count("\t")

        if number_of_tabs not in tab_counts:
            tab_counts[number_of_tabs] = 0

        tab_counts[number_of_tabs] += 1

    # Display the tab distribution to identify inconsistent rows
    print("\nNumber of tabs per row:")
    print(tab_counts)
