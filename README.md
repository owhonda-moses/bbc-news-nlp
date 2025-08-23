# NLP Analysis of BBC News Articles

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20Transformers%20%7C%20spaCy-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains the complete workflow for an advanced Natural Language Processing (NLP) project performed on the BBC News dataset from 2004-2005. The project demonstrates a series of advanced techniques, including model fine-tuning, handling class imbalance, custom Named Entity Recognition (NER), and abstractive summarization.

---
## Table of Contents
1.  [Project Goal](#project-goal)
2.  [Repository Structure](#repository-structure)
3.  [Setup and Installation](#setup-and-installation)
4.  [Usage: Reproducing the Results](#usage-reproducing-the-results)
5.  [Methodology and Iterations](#methodology-and-iterations)
    * [Sub-Category Classification](#sub-category-classification)
    * [Custom Named Entity Recognition](#custom-named-entity-recognition)
    * [Conditional Summarization](#conditional-summarization)
6.  [Models](#models)
7.  [Limitations](#limitations)
8.  [Future Work](#future-work)
9.  [Acknowledgments](#acknowledgments)

---
## Project Goal

The primary goal of this project was to perform a multi-faceted NLP analysis on a dataset of news articles. The defined problem consisted of three core tasks:

1.  **Advanced Classification:** To classify news articles from broad categories ('Business', 'Entertainment', 'Sport') into more granular, meaningful sub-categories (e.g., 'Stock Market', 'Cinema', 'Football') using weakly supervised labels.
2.  **Custom Entity Extraction:** To identify and extract named entities of media personalities from the text, while also classifying their professional roles (e.g., Politician, Musician, TV/Film Personality).
3.  **Conditional Summarization:** To filter the dataset for all articles pertaining to events in the month of "April" and generate a concise, abstractive summary for each.

---
## Repository Structure

The project is organized into a standard data science structure for clarity and reproducibility.

```
bbc-nlp/
├── .gitignore
├── README.md
├── poetry.lock
├── pyproject.toml
└── setup.sh

├── data/
│   ├── bbc_raw/
│   ├── bbc_prep/
│   ├── april_events_summaries.csv
│   ├── ner_training_data.csv
│   ├── train_data_augmented.csv
│   ├── train_data.csv
│   └── val_data.csv
│
├── img/
│   └── confusion_matrix.png
│
├── models/
│   ├── baseline-classifier/
│   ├── weighted-classifier/
│   ├── augmented-classifier/
│   └── ner-model/
│
└── src/
    ├── __init__.py
    ├── april_events.py
    ├── augment_data.py
    ├── evaluate.py
    ├── prepare_data.py
    ├── prepare_ner_data.py
    ├── test_ner.py
    ├── train_augmented.py
    ├── train_baseline.py
    ├── train_ner.py
    └── train_weighted.py
```

---
## Setup and Installation

This project is designed to be run in a containerized environment with GPU support, such as a Paperspace Gradient Notebook using the `paperspace/gradient-base:pt211-tf215-cudatk120-py311` image.

**Prerequisites:**
* Python 3.11
* [Poetry](https://python-poetry.org/) for dependency management
* Git

### Steps:
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/owhonda-moses/bbc-news-nlp.git
    cd bbc-news-nlp
    ```

2.  **Create Personal Access Token File**
    This project requires a GitHub Personal Access Token (PAT) for the setup script. Create a file named `pat.env` in the root directory and add it to .gitignore:
    ```
    # pat.env
    GITHUB_TOKEN=your_personal_access_token_here
    ```

3.  **Run the Setup Script**
    The `setup.sh` script automates the entire environment setup. It will install dependencies, configure the virtual environment to use the system's PyTorch (for efficiency), download NLP models, and perform a final cleanup.
    ```bash
    bash setup.sh
    ```
    This script must be run once at the beginning of each new ephemeral session.

---
## Usage: Reproducing the Results

The project is broken down into a series of scripts within the `src/` directory. They should be run in the following order to reproduce the entire workflow. All commands should be run from the root `bbc-nlp` directory.

1.  **Prepare Main Dataset (`src/prepare_data.py`)**
    This script loads the raw text, generates heuristic labels for classification, and splits the data into training and validation sets.
    ```bash
    python -m src.prepare_data
    ```

2.  **Train Classifiers**
    We iterated through three models. You can train any or all of them.
    * **Baseline**: `python -m src.train_baseline`
    * **With Class Weights**: `python -m src.train_weighted`
    * **With Augmented Data** (requires `augment_data.py` to be run first): `python -m src.train_augmented`

3.  **Evaluate Classifier (`src/evaluate.py`)**
    This script generates a classification report and confusion matrix. Remember to change the `MODEL_PATH` variable inside the script to point to the model you want to evaluate (e.g., `./models/weighted-classifier`).
    ```bash
    python -m src.evaluate
    ```

4.  **Prepare NER Data (`src/prepare_ner_data.py`)**
    This script programmatically annotates the data and creates the training set for our custom NER model.
    ```bash
    python -m src.prepare_ner_data
    ```

5.  **Train NER Model (`src/train_ner.py`)**
    Fine-tunes a `DistilBertForTokenClassification` model on our custom data.
    ```bash
    python -m src.train_ner
    ```

6.  **Test NER Model (`src/test_ner.py`)**
    Runs inference with the custom NER model on test sentences to demonstrate its performance.
    ```bash
    python -m src.test_ner
    ```

7.  **Summarize Events (Task C) (`src/april_events.py`)**
    Filters for articles mentioning "April" and generates abstractive summaries using a BART model.
    ```bash
    python -m src.april_events
    ```

---
## Methodology and Iterations

### Sub-Category Classification
The primary challenge was the lack of pre-existing sub-category labels. Our approach was:
1.  **Weak Supervision:** We generated initial noisy labels using a keyword-matching script.
2.  **Baseline Model:** We fine-tuned a `DistilBERT` model on this data, achieving high accuracy (~90%) but a low macro F1-score (~0.51), indicating poor performance on rare classes.
3.  **Class Imbalance:** We diagnosed the issue using a detailed classification report, which confirmed that the model was ignoring minority classes.
4.  **Iteration 1: Class Weights:** We re-trained the model using class weights to penalize errors on minority classes more heavily, which dramatically improved the F1-score to ~0.68.
5.  **Iteration 2: Data Augmentation:** We used a T5 model to generate synthetic data for the most underperforming classes. This provided a slight additional boost to the F1-score but showed signs of diminishing returns.

### Custom Named Entity Recognition
To extract personalities and their jobs, we built a custom NER model.
1.  **Programmatic Annotation:** We used `spaCy` to find `PERSON` entities and then applied keyword-based rules to assign custom labels (`POLITICIAN`, `MUSICIAN`, etc.).
2.  **IOB Formatting:** This annotated data was converted into the standard IOB (Inside-Outside-Beginning) format required for training.
3.  **Model Fine-Tuning:** A `DistilBertForTokenClassification` model was fine-tuned on this custom-generated dataset. The final model successfully identified the custom entities in test sentences.

### Task C: Conditional Summarization
This task was accomplished by:
1.  **Filtering:** We filtered the entire dataset for articles containing the word "April".
2.  **Abstractive Summarization:** We used a powerful, pre-trained `facebook/bart-large-cnn` model to generate high-quality, human-like summaries for each of the filtered articles.

---
## Models
The final models are saved in the `models/` directory:
* `baseline-classifier`: The initial classification model.
* `weighted-classifier`: The improved classifier trained with class weights. **(Recommended)**
* `augmented-classifier`: The classifier trained with augmented data.
* `ner-model`: The custom NER model for identifying personalities and jobs.

---
## Limitations
* **Label Quality:** The performance of both the classifier and the NER model is fundamentally limited by the quality of our programmatically generated labels as they contain inherent noise and errors.
* **Data Diversity:** The data augmentation was limited by the small number of source articles for the rarest classes. The generated text, while syntactically different, was semantically very similar to the originals.
* **Summarization Scope:** The summarization model was a pre-trained, general-purpose model. It was not fine-tuned on our specific BBC News dataset, which could have improved its stylistic consistency.

---
## Future Work
* **Gold-Standard Evaluation:** Manually annotate a small, held-out test set for both classification and NER. This would allow for a true evaluation of the models' performance against human-level accuracy.
* **Experiment with Larger Models:** Re-train the final models using a more powerful base architecture like `RoBERTa` or `DeBERTa` to potentially increase performance.
* **Advanced Sampling:** Implement SMOTE or other advanced oversampling techniques as an alternative to data augmentation to see if it yields better results on the minority classes.
* **Fine-Tune Summarization Model:** Fine-tune the BART model on the BBC News dataset to make its summaries better match the specific journalistic style of the source text.

---
## Acknowledgments
* This project uses the [BBC News Dataset](http://mlg.ucd.ie/datasets/bbc.html), originally collected for the publication: D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.
* The project heavily relies on the open-source work of Hugging Face, spaCy, PyTorch, and the broader Python data science community.
