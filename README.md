# NLP Analysis of BBC News Articles

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20Transformers%20%7C%20spaCy-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains the complete workflow for an advanced Natural Language Processing (NLP) project performed on the BBC News dataset from 2004-2005. The project demonstrates a series of state-of-the-art techniques, including data-centric AI for custom Named Entity Recognition (NER) using a Large Language Model (LLM), systematic hyperparameter tuning, advanced class imbalance handling, and abstractive summarization.

---
## Table of Contents
1.  [Project Goal](#project-goal)
2.  [Repository Structure](#repository-structure)
3.  [Setup and Installation](#setup-and-installation)
4.  [Usage: Reproducing the Results](#usage-reproducing-the-results)
5.  [Methodology](#methodology)
    * [Sub-Category Classification](#sub-category-classification)
    * [Custom Named Entity Recognition](#custom-named-entity-recognition)
    * [Conditional Summarization](#conditional-summarization)
6.  [Models](#models)
7.  [Limitations and Future Work](#limitations-and-future-work)
8.  [Acknowledgments](#acknowledgments)

---
## Project Goal

The primary goal of this project was to perform a multi-faceted NLP analysis on a dataset of news articles. The defined problem consisted of three core tasks:

1.  **Advanced Classification:** To classify news articles from broad categories into granular sub-categories (e.g., 'Economy', 'Cinema', 'Football') using weakly supervised labels and advanced training techniques.
2.  **Custom Entity Extraction:** To build a high-performance NER model capable of identifying and classifying media personalities into specific professional roles (e.g., Politician, Musician, Athlete).
3.  **Conditional Summarization:** To filter the dataset for all articles pertaining to events in the month of "April" and generate a concise, abstractive summary for each.

---
## Repository Structure

The project is organized into a modular structure for clarity and reproducibility.

```
bbc-nlp/
├── .env
├── README.md
├── archive/
├── data/
│   ├── bbc_raw/
│   └── output/
├── models/
│   ├── augmented-classifier/
│   ├── distilbert-model-v2/
│   └── roberta-model-v2/
├── params/
└── src/
    ├── classification/
    ├── ner/
    ├── preprocessing/
    └── summarization/
```

---
## Setup and Installation

This project is designed to be run in a containerized environment with GPU support, such as a Paperspace Gradient Notebook using the `paperspace/gradient-base:pt211-tf215-cudatk120-py311` image.

**Prerequisites:**
* Python 3.11
* [Poetry](https://python-poetry.org/) for dependency management
* Git
* A Gemini API Key

### Steps:
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/owhonda-moses/bbc-news-nlp.git
    cd bbc-nlp
    ```

2.  **Create Personal Access Token File**

    This project requires a GitHub Personal Access Token for the setup script. Create a file named `pat.env` in the root directory and add it to `.gitignore`:
    ```bash
    GITHUB_TOKEN=your_personal_access_token
    ```

3.  **Create Environment File**

    Create a file named `.env` in the root directory for your Gemini API key:
    ```bash
    GEMINI_API_KEY=your_api_key
    ```

4.  **Run the Setup Script**

    The `setup.sh` script automates the environment setup, including dependency installation and downloading NLP models.
    ```bash
    chmod +x setup.sh
    bash setup.sh
    ```

---
## Usage: Reproducing the Results

The project is broken down into a series of scripts within the `src/` subdirectories. They should be run in the following order.

### 1. Initial Data Split
This is the first step for the entire project.

```
    python -m src.preprocessing.main_split
```

### 2. Test Set Creation
This creates the manually verified validation and test sets.
##### - Manually create annotations in src/preprocessing/annotations.json
##### - Apply annotations to create the final _test_set.csv_

    ```bash
    python -m src.preprocessing.apply_annotations
    ```
##### - Split into _ner_val.csv_ and _ner_test.csv_

    ```bash
    python -m src.preprocessing.ner_split
    ```

### 3. Sub-Category Classification Pipeline
This trains the text classifier used by the NER pipeline.
##### - Prepare training data from weakly-supervised labels

    ```bash
    python -m src.classification.prepare_zeroshot
    ```
##### - Augment the data

    ```bash
    python -m src.classification.augment_data
    ```
##### - Train the final classifier

    ```bash
        python -m src.classification.train_augmented
    ```

### 4. Custom Named Entity Recognition (NER) Pipeline
This is the core data-centric AI workflow for the NER task.
##### - Build the knowledge base from Wikidata

    ```bash
    python -m src.ner.bulkseed
    ```
##### - Augment the knowledge base from Wikipedia

    ```bash
    python -m src.ner.scraper
    ```
##### - Merge knowledge bases

    ```bash
    python -m src.ner.merge
    ```
_labeler_v1 is the base model labeled purely from knowledge base_
##### - Use the LLM to generate high-quality labels for the training data

    ```bash
    python -m src.ner.labeler_v2
    ```
##### - Review and correct the LLM's output (optional but recommended)

    ```bash
    python -m src.ner.correct_ner --filter <keyword>
    ```
##### - Pre-process the final, augmented data for model training

    ```bash
    python -m src.ner.preprocess
    ```
##### - Train the final v2 NER model

    ```bash
    python -m src.ner.train_ner
    ```
##### - Evaluate the final model

    ```bash
    python -m src.ner.evaluate
    ```

### 5. Conditional Summarization

    ```bash
    python -m src.summarization.april_events
    ```

---
## Methodology

### Sub-Category Classification
The primary challenge was the lack of pre-existing sub-category labels. Our final, successful approach was:
1.  **Weak Supervision via Zero-Shot Learning:** We used a `facebook/bart-large-mnli` model to generate high-quality, zero-shot labels for our training data.
2.  **Systematic Hyperparameter Tuning:** We used `Optuna` to perform a robust search for the optimal learning rate, weight decay, and random seed.
3.  **Targeted Data Augmentation:** We used a T5 model with context-aware prompts to generate synthetic data for under-represented classes.

### Custom Named Entity Recognition
This task was approached with a state-of-the-art, data-centric pipeline to create a high-quality training set.
1.  **Knowledge Base Creation:** We programmatically built a large knowledge base of over 1 million names and their professional roles by querying Wikidata (`bulkseed.py`) and scraping Wikipedia (`scraper.py`).
2.  **Document-Level LLM Labeling:** We developed a sophisticated script (`labeler_v2.py`) that uses the Gemini 2.5 Flash model as a reasoning engine. For each article, it predicts the sub-category, links all mentions of the same person, and uses the full article context and the knowledge base to assign a highly accurate NER label in a single pass.  The script uses an advanced one-shot prompt to ensure reliable JSON output and can filter out non-person entities.
3.  **Human-in-the-Loop Correction:** After the automated labeling, the `correct_ner.py` script provides an interactive interface to perform a final, targeted review of the AI-generated labels to fix any subtle errors.
4.  **Hybrid Augmentation:** To create a balanced training set, the `preprocess.py` script applies a hybrid strategy that uses **oversampling** to synthetically increase the number of examples for minority classes.
5.  **Advanced Model Training:** The final `train_ner.py` script uses several state-of-the-art techniques, including a **weighted loss function** and **Automatic Mixed Precision (AMP)** to speed up training.

### Conditional Summarization
1.  **Filtering:** We perform basic filtering on the gold standard test data for articles containing "April".
2.  **Hybrid Summarization:** A sophisticated pipeline performs an **extractive** step, identifying the most relevant sentences in an article using keyword and pattern matching.
2.  **Abstractive Summarization:** These key sentences are then fed to a pre-trained `facebook/bart-large-cnn` model to generate a high-quality, human-like abstractive summary. This is more robust than summarizing the entire article.

---
## Models
The final models are saved in the `models/` directory:
* `augmented-classifier`: The final, best-performing sub-category classifier.
* `distilbert-model-v2`: The final DistilBERT NER model.
* `roberta-model-v2`: The final RoBERTa NER model. **(Recommended)**

---
## Limitations and Future Work
* **Upstream Model Dependence:** The NER pipeline's quality is dependent on the performance of the sub-classification model and the initial entity recognition from spaCy.
* **Knowledge Base Errors:** The programmatically-built knowledge base, while large, may contain some noise and errors that are corrected over time via review.
* **Future Work:** The most impactful next step would be to continue the data-centric loop by using the trained v2 NER model to find more errors, correct them, and re-train a v3 model to further improve performance.

---
## Acknowledgments
* This project uses the [BBC News Dataset](http://mlg.ucd.ie/datasets/bbc.html), originally collected for the publication: D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.
* The project heavily relies on the open-source work of Hugging Face, spaCy, PyTorch, and the broader Python data science community.