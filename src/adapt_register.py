
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

model.eval()

print("Model loaded successfully!")


def adapt_register(russian_text, english_translation, desired_register):

    desired_register = desired_register.lower()

    if desired_register not in ["formal", "informal"]:
        return "Choose formal or informal."

    # Create instructions for the AI model
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert English register adaptation assistant. "
                "You must follow the requested register exactly. "
                "FORMAL: Use professional vocabulary, complete expressions "
                "and a respectful tone. "
                "INFORMAL: Use relaxed everyday vocabulary, contractions "
                "and natural conversational expressions. "
                "Never use formal expressions when informal is requested. "
                "Never use informal expressions when formal is requested. "
                "Preserve the original meaning."
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

    # Convert instructions into tokens
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    # Generate the adapted sentence
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.1
        )

    # Extract generated tokens
    generated_tokens = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]

    # Convert tokens into readable text
    adapted_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return adapted_text


# Test formal register
formal_result = adapt_register(
    "Я хотела бы попросить преподавателя о помощи.",
    "Can you help me with this?",
    "formal"
)

# Test informal register
informal_result = adapt_register(
    "Я хотела бы попросить преподавателя о помощи.",
    "Can you help me with this?",
    "informal"
)

# Display results
print("\nFORMAL:", formal_result)
print("INFORMAL:", informal_result)
