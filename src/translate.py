import torch
from transformers import MarianMTModel, MarianTokenizer

# Identify the pre-trained Russian-to-English translation model
model_name = "Helsinki-NLP/opus-mt-ru-en"

# Load the tokenizer and translation model
tokenizer = MarianTokenizer.from_pretrained(model_name)
translation_model = MarianMTModel.from_pretrained(model_name)

# Select the available processing device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Move the model to the selected device
translation_model = translation_model.to(device)

# Set the model to evaluation mode
translation_model.eval()


def translate_text(russian_text):
    """
    Translate a Russian sentence into English.

    Parameters:
        russian_text: A sentence written in Russian

    Returns:
        The English translation
    """

    # Convert Russian text into tokens
    inputs = tokenizer(
        russian_text,
        return_tensors="pt"
    ).to(device)

    # Generate the English translation
    with torch.inference_mode():
        translated = translation_model.generate(**inputs)

    # Convert the output tokens into readable English text
    english_translation = tokenizer.decode(
        translated[0],
        skip_special_tokens=True
    )

    return english_translation
