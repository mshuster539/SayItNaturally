import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the pre-trained Qwen language model
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# Load the tokenizer, which converts text into tokens
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the pre-trained language model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

# Select the available processing device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Move the model to my device
model = model.to(device)

# Set the model to evaluation (not training)mode
model.eval()


def adapt_register(russian_text, english_translation, desired_register):
    """
    Adapt an English translation into the requested
    formal or informal register using Qwen.

    Parameters:
        russian_text: Original Russian sentence
        english_translation: English translation
        desired_register: Formal or informal

    Returns:
        An adapted English sentence
    """

    # Check that the requested register is valid
    desired_register = desired_register.lower().strip()

    if desired_register not in ["formal", "informal"]:
        return "Please choose formal or informal."

    # Create instructions for the language model
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert English register adaptation assistant. "
                "You must follow the requested register exactly. "
                "FORMAL: Use professional vocabulary, complete expressions "
                "and a respectful tone. "
                "INFORMAL:Use natural, fluent, everyday English. Avoid exaggerated expressions, childish wording, awkward phrases, and overly enthusiastic language. Do not force slang or contractions. Prioritise natural phrasing over making the text sound noticeably informal."
            )
        },
        {
            "role": "user",
            "content": (
                f"Russian original: {russian_text}\n"
                f"English sentence: {english_translation}\n"
                f"Required register: {desired_register.upper()}\n"
                f"Rewrite this sentence in {desired_register.upper()} English. "
                "Return only the rewritten sentence."
            )
        }
    ]

    # Convert the instructions into tokens
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    # Move the input tokens to my device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Generate the adapted sentence
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.1
        )

    # Remove the original prompt from the generated output
    generated_tokens = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]

    # Convert the generated tokens into readable text
    adapted_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return adapted_text

