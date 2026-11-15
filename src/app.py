import gradio as gr
import joblib
from pathlib import Path

from translate import translate_text
from adapt_register import adapt_register


# Load the trained register classifier
project_folder = Path(__file__).resolve().parent.parent
model_path = project_folder / "models" / "register_classifier.joblib"

classifier = joblib.load(model_path)


def process_text(russian_text, desired_register):

    # Check that the user has entered a sentence
    if not russian_text or not russian_text.strip():
        return "", "", "Please enter a Russian sentence."

    # Check that a valid register has been selected
    if desired_register not in ["Formal", "Informal"]:
        return "", "", "Please select a register."

    try:
        # Step 1: Translate Russian into English
        english_translation = translate_text(russian_text)

        # Step 2: Adapt the English translation into the selected register
        adapted_text = adapt_register(
            russian_text,
            english_translation,
            desired_register.lower()
        )

        if not adapted_text:
            return "", "", "No adapted sentence was generated."

        # Step 3: Use the classifier to predict the register
        predicted_register = classifier.predict([adapted_text])[0]

        # Step 4: Compare the prediction with the selected register
        if str(predicted_register).lower() == desired_register.lower():
            status = (
                "Register verified! The classifier's prediction "
                "matches your selected register."
            )
        else:
            status = (
                "Register mismatch! The classifier's prediction "
                "does not match your selected register. "
                "You may want to review the sentence."
            )

        return (
            adapted_text,
            str(predicted_register).capitalize(),
            status
        )

    except Exception as error:
        return "", "", f"An error occurred: {error}"


# Create the Gradio interface
with gr.Blocks(title="SayItNaturally") as app:

    gr.Markdown(
        """
        # SayItNaturally

        ### Express yourself naturally in English.

        Enter a sentence in Russian and choose how you
        would like to express it in English.
        """
    )

    russian_input = gr.Textbox(
        label="Your Russian sentence",
        placeholder="Введите предложение на русском...",
        lines=3
    )

    register_choice = gr.Radio(
        choices=["Formal", "Informal"],
        label="Choose your desired register",
        value="Formal"
    )

    submit_button = gr.Button(
        "Translate and Adapt",
        variant="primary"
    )

    gr.Markdown("## Your Results")

    adapted_output = gr.Textbox(
        label="Your adapted English sentence",
        lines=3,
        interactive=False
    )

    classifier_output = gr.Textbox(
        label="Classifier prediction",
        interactive=False
    )

    status_output = gr.Textbox(
        label="Register verification",
        interactive=False
    )

    submit_button.click(
        fn=process_text,
        inputs=[
            russian_input,
            register_choice
        ],
        outputs=[
            adapted_output,
            classifier_output,
            status_output
        ]
    )


# Launch the application
if __name__ == "__main__":
    app.launch()

