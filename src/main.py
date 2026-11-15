
import joblib
from pathlib import Path

from translate import translate_text
from adapt_register import adapt_register


# Load the trained register classifier
project_folder = Path(__file__).resolve().parent.parent
model_path = project_folder / "models" / "register_classifier.joblib"

classifier = joblib.load(model_path)

print("SayItNaturally is ready!")


def process_text(russian_text, desired_register):

    # Step 1: Translate Russian into English
    english_translation = translate_text(russian_text)

    print("\nOriginal Russian:", russian_text)
    print("English translation:", english_translation)

    # Step 2: Adapt the English register using Qwen
    adapted_text = adapt_register(
        russian_text,
        english_translation,
        desired_register
    )

    print("\nSelected register:", desired_register)
    print("Adapted sentence:", adapted_text)

    # Step 3: Evaluate the adapted sentence using the trained classifier
    predicted_register = classifier.predict([adapted_text])[0]

    print("\nClassifier prediction:", predicted_register)

    # Step 4: Check whether the predicted register matches the selection
    if str(predicted_register).lower() == desired_register.lower():
        print("Register check: Successful!")
    else:
        print("Register check: The predicted register does not match.")

    return {
        "russian": russian_text,
        "translation": english_translation,
        "register": desired_register,
        "adapted_text": adapted_text,
        "predicted_register": predicted_register
    }


# Run the application
if __name__ == "__main__":

    russian_text = input("\nEnter a sentence in Russian: ")

    desired_register = input(
        "Choose a register (formal/informal): "
    ).lower().strip()

    if desired_register not in ["formal", "informal"]:
        print("Please choose formal or informal.")

    else:
        result = process_text(
            russian_text,
            desired_register
        )
