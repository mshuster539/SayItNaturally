import joblib
from pathlib import Path

from translate import translate_text
from adapt_register import adapt_register


# Find the main project folder
project_folder = Path(__file__).resolve().parent.parent

# Load the trained register classifier
model_path = project_folder / "models" / "register_classifier.joblib"
classifier = joblib.load(model_path)


def process_text(russian_text, desired_register):

    # Step 1: Translate Russian into English
    english_translation = translate_text(russian_text)

    # Step 2: Adapt the English sentence into the selected register
    adapted_text = adapt_register(
        russian_text,
        english_translation,
        desired_register
    )

    # Step 3: Predict the register of the adapted sentence
    predicted_register = classifier.predict([adapted_text])[0]

    # Step 4: Check whether the prediction matches the selected register
    if str(predicted_register).lower() == desired_register.lower():
        status = "Register check successful!"
    else:
        status = "The predicted register does not match the selection."

    # Return all the results
    return {
        "russian": russian_text,
        "translation": english_translation,
        "register": desired_register,
        "adapted_text": adapted_text,
        "predicted_register": predicted_register,
        "status": status
    }


# Run the command-line application
if __name__ == "__main__":

    russian_text = input("Enter a sentence in Russian: ").strip()

    desired_register = input(
        "Choose a register (formal/informal): "
    ).lower().strip()

    if not russian_text:
        print("Please enter a Russian sentence.")

    elif desired_register not in ["formal", "informal"]:
        print("Please choose formal or informal.")

    else:
        result = process_text(russian_text, desired_register)

        print("\nEnglish translation:", result["translation"])
        print("Adapted sentence:", result["adapted_text"])
        print("Selected register:", result["register"])
        print("Classifier prediction:", result["predicted_register"])
        print("Register check:", result["status"])
