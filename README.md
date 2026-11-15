# SayItNaturally

**An AI-powered Russian-to-English translation tool that adapts
sentences to formal or informal English.**

SayItNaturally allows users to enter a sentence in Russian, choose
whether they want the English translation to sound formal or informal,
and receive an adapted translation.

## Demo

**Demo video:** https://drive.google.com/file/d/1NWec_iVExkqqHGdxcipwz-D0zAMRLPoG/view?usp=sharing

## How it works

The application uses three models in a simple pipeline:

1.  **Translation --- MarianMT (`Helsinki-NLP/opus-mt-ru-en`):**
    translates the user's Russian sentence into English.
2.  **Register adaptation --- Qwen2.5-1.5B-Instruct:** adapts the
    English translation to the selected formal or informal register.
3.  **Register verification --- TF-IDF and Logistic Regression:** my
    trained classifier predicts whether the adapted sentence is formal
    or informal and compares its prediction with the user's selection.

The interface is built using Gradio.

## Technologies

-   Python
-   Gradio
-   Hugging Face Transformers
-   Helsinki-NLP MarianMT
-   Qwen2.5-1.5B-Instruct
-   Pandas
-   Scikit-learn (TF-IDF and Logistic Regression)

## Dataset

The formal and informal sentence datasets used to train the register
classifier were sourced from **FAME-MT**.

-   [FAME-MT research paper (Wisniewski et al.,
    2024)](https://aclanthology.org/2024.eamt-1.16/)
-   [FAME-MT GitHub
    repository](https://github.com/laniqo-public/fame-mt)

The original formal and informal datasets each contained 50,000 sentence
pairs. After removing missing values and duplicate sentence pairs, the
cleaned datasets contained 47,536 formal and 41,616 informal sentences.

## Classifier results

  Model                          Evaluation set                Accuracy
  ------------------------------ -------------------------- -----------
  Majority-class baseline        17,831 unseen examples           53.3%
  TF-IDF + Logistic Regression   17,831 unseen examples       **87.7%**
  TF-IDF + Logistic Regression   40 self-written examples     **97.5%**

The self-written examples were evaluated separately and are not part of
the held-out test set. The classifier's accuracy measures register
classification, not the overall translation quality of the application.

## Limitations

-   Formal and informal language can overlap, which can make register
    difficult to classify.
-   The classifier learns language patterns and does not fully
    understand the context in which a sentence is used.
-   Adapted translations can sometimes sound exaggerated or unnatural,
    particularly when the model tries to make a sentence informal.
-   The register classifier provides an additional prediction step; it
    does not guarantee that a translation is completely natural or
    correct.

## Installation and running

### 1. Clone the repository

``` bash
git clone https://github.com/mshuster539/SayItNaturally.git
cd SayItNaturally
```

### 2. Create and activate a virtual environment

**macOS / Linux**

``` bash
python -m venv .venv
source .venv/bin/activate
```

**Windows**

``` bash
python -m venv .venv
.venv\\Scripts\\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Run the application

``` bash
python src/app.py
```


## Sources and acknowledgements

-   Wisniewski, D., Rostek, Z., & Nowakowski, A. (2024). *FAME-MT
    Dataset: Formality Awareness Made Easy for Machine Translation
    Purposes*. Proceedings of the 25th Annual Conference of the European
    Association for Machine Translation, 164--180.
    [Paper](https://aclanthology.org/2024.eamt-1.16/)
-   [Helsinki-NLP OPUS-MT Russian-to-English
    model](https://huggingface.co/Helsinki-NLP/opus-mt-ru-en)
-   [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
-   [Scikit-learn documentation](https://scikit-learn.org/stable/)
-   [Gradio documentation](https://www.gradio.app/docs)

ChatGPT was used for debugging code and assistance with implementing
pretrained models.
