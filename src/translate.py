import torch
import joblib
from transformers import MarianMTModel, MarianTokenizer

# Identify the pretrained translation model
model_name = "Helsinki-NLP/opus-mt-ru-en"

# Load the tokenizer and translation model
tokenizer = MarianTokenizer.from_pretrained(model_name)
translation_model = MarianMTModel.from_pretrained(model_name)

# Select the device
device = "mps" if torch.backends.mps.is_available() else "cpu"

translation_model = translation_model.to(device)
translation_model.eval()
print("Translation model loaded successfully!")
print("Device:", device)

# Create a reusable translation function
def translate_text(russian_text):

    # Convert Russian text into tokens
    inputs = tokenizer(
        russian_text,
        return_tensors="pt"
    ).to(device)

    # Generate the English translation
    with torch.no_grad():
        translated = translation_model.generate(**inputs)

    # Convert the output tokens into English text
    english_translation = tokenizer.decode(
        translated[0],
        skip_special_tokens=True
    )

    return english_translation


# Test the function
result = translate_text("Я так устала...")

print("\nEnglish translation:", result)



classifier = joblib.load("models/register_classifier.joblib")


def translate_and_classify(russian_text):

    # Translate Russian into English
    english_translation = translate_text(russian_text)

    # Predict whether the English sounds formal or informal
    predicted_register = classifier.predict([english_translation])[0]

    return english_translation, predicted_register

result = translate_and_classify("Я хотела бы попросить вас о помощи.")

print("\nEnglish translation:", result[0])
print("Predicted register:", result[1])