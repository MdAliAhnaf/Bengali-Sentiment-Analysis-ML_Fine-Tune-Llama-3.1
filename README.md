# Bengali-Sentiment-Analysis-ML_Fine-Tune-Llama-3.1

This repository contains code and documentation for performing **sentiment analysis** on a Bengali text dataset using traditional machine learning models (Logistic Regression, Random Forest, etc.) and **fine-tuning** the **Dolphin 2.9.4 Llama 3.1 (8B)** model for Bengali text processing. The project also documents challenges faced with data preprocessing, NLP toolkit selection, model performance, and hardware limitations.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Dataset](#dataset)
4. [Preprocessing](#preprocessing)
5. [Vectorization](#vectorization)
5. [Used Models & Results](#Used-Models-&-Results)
6. [Fine-Tuning Llama 3.1](#fine-tuning-llama-31)
7. [Challenges and Limitations](#challenges-and-limitations)
8. [Future Improvements](#future-improvements)

## Project Overview

This project aims to:
- **Classify sentiments (positive, negative, neutral)** of Bengali text data.
- **Evaluate traditional ML models** and deep learning architectures like LSTM for sentiment classification.
- **Fine-tune large language models**, such as Llama 3.1 (8B), despite hardware limitations, using alternatives like Dolphin 2.9.4 Llama 3.1 (8B).
- Provide a **comprehensive analysis of limitations** in NLP toolkit selection, preprocessing issues, and hardware constraints, as well as a detailed **comparison of model performance**.

## Key Features
- **Preprocessing of Bengali text** using BNLP's `CleanText` module and `NLTKTokenizer`.
- **Traditional ML model evaluation** with Logistic Regression, Random Forest, XGBoost, and LightGBM.
- **Deep Learning model** implementation using LSTM for sentiment analysis.
- **Fine-tuning Dolphin 2.9.4 Llama 3.1 (8B)** on limited hardware (4GB VRAM).
- **Comprehensive evaluation** using Stratified K-Fold and K-Fold cross-validation techniques.
- **Extensive documentation of challenges and limitations**, particularly with hardware and NLP toolkit selection.

## Dataset
- **Source:** The dataset consists of an Excel file containing **Bengali text** and **sentiment labels** (positive, negative, neutral). The data is processed using **BNLP** for text tokenization and cleaning.
- **Loading Method:** Utilized `pandas.read_excel()` to load the dataset.

## Preprocessing
- **Text Cleaning:** Used BNLP’s `CleanText` module with custom parameters (`fix_unicode=True`, `unicode_norm=True`) to handle Unicode errors and clean the text output effectively.
- **Unwanted Strings Removal:** Removed the string "See Translation" and reduced duplicate punctuation marks such as ‘।’ (Bengali full stop), ‘,’ (comma), ‘?’ (question mark), and ‘…’ (ellipsis) using `re.sub()`.
- **Sentence Tokenization:** Used BNLP's `NLTKTokenizer` for tokenizing text at the sentence level. This was necessary because `word_tokenize` removed punctuation, which was crucial for sentiment analysis.
- **Stemmer Issue:** Initially employed stemmers from `banglanltk`, but they truncated words undesirably (e.g., ‘আমাকে’ to ‘আমা’), leading to loss of meaning. Consequently, stemming was excluded from the preprocessing pipeline.

## Vectorization
  - Used `TfidfVectorizer` to transform the cleaned text into TF-IDF feature vectors. Converted the sparse matrix to a dense format with `.toarray()` to facilitate model training.


## Used Models & Results

| Model                     | Accuracy (K-Fold) | Accuracy (Stratified) | Best Parameters                           |
|---------------------------|------------------|-----------------------|-------------------------------------------|
| Logistic Regression        | 75%              | 75%                   | {'C': 10, 'solver': 'liblinear'}          |
| Multinomial Naive Bayes    | 65%              | 65%                   | Default                                   |
| Random Forest Classifier   | 70%              | 75%                   | {'max_depth': 10, 'n_estimators': 50'}    |
| XGBoost                    | 60%              | 60%                   | {'learning_rate': 0.2, 'n_estimators': 100'} |
| LightGBM                   | 50%              | 55%                   | {'learning_rate': 0.1, 'n_estimators': 100'} |
| LSTM                       | 55%              | 55%                   | Default                                   |

### Analysis
- **Best Performance:** Logistic Regression and Random Forest were the top performers with 75% accuracy, indicating that simpler models combined with effective text vectorization can be highly effective for sentiment analysis.
- **LSTM Performance:** The LSTM model exhibited poor performance (55% accuracy). The small dataset size and high variance likely contributed to its underperformance.

## Fine-Tuning Llama 3.1
### Initial Issues
- **Licensing Restrictions:** Faced difficulties accessing Llama 3.1 models due to licensing issues. Applied for access, but was pending approval.
- **Model Choice:** Used Dolphin 2.9.4 Llama 3.1 8B model from Hugging Face as an alternative.
Due to **licensing issues with Meta’s Llama 3.1 model**, the Dolphin 2.9.4 Llama 3.1 (8B) model was used as an alternative for fine-tuning. This model is based on Meta's **Llama 3.1 8B** with 8.03 billion parameters.

### Model Details:
- **Context Length:** 128K
- **Training Sequence Length:** 8192
- **Prompt Template:** ChatML
- **Training Hyperparameters:**
  - Learning rate: 5e-06
  - Batch size (training/evaluation): 2
  - Epochs: 3
  - Gradient accumulation steps: 4
  - Optimizer: Adam

### Challenges and Limitations
- **Hardware Limitation:** The project was conducted on a **GTX 1050Ti with 4GB VRAM**, which proved insufficient for fine-tuning large models like Llama 3.1.
- **Workarounds:** Forced the use of CPU (`no_cuda=True`), but kernel crashes occurred frequently despite adjustments in batch size, gradient accumulation, and mixed precision (`fp16=True`).
- Ensuring that important parts of Bengali text, especially punctuation, were preserved during preprocessing.
- Finding a reliable NLP library for Bengali text processing, which led to the exploration of both BNLP and `banglanltk`.
- **Data Format:** Conversion of TF-IDF vectorized data to a dense format suitable for LSTM was required.
- **Dataset Size:** The limited dataset size (99 rows) led to overfitting and hampered the model's ability to generalize.

### NLP Toolkit Challenges:
- **Stemming Issues:** The Bangla stemmer truncated important parts of the text, so stemming was excluded.
- **Punctuation Loss:** Initial tokenization attempts using `word_tokenize` removed punctuation, which affected sentiment classification. Resolved by using `sentence_tokenize`.

### Model Performance:
- **Sparse Matrix Conversion:** The output from TF-IDF vectorization was a sparse matrix, which had to be converted to a dense format for model compatibility.

### Hardware Constraints:
- **GPU VRAM Limitation:** 4GB VRAM was insufficient to fine-tune the Llama 3.1 model, even with optimizations like smaller batch sizes and gradient accumulation.
- **CPU Use:** Forced CPU use due to GPU constraints, leading to frequent kernel crashes despite adjustments.

### Performance Comparison:
- **Best Performing Models:** Logistic Regression and Random Forest, both achieving 75% accuracy with the Stratified K-Fold cross-validation method.
- **Worst Performing Models:** LSTM and LightGBM performed poorly with an accuracy of 55%.

### Fine-Tuning Llama:
- Due to hardware limitations, fine-tuning the Dolphin 2.9.4 Llama 3.1 model was unsuccessful, with frequent kernel crashes.

### Prerequisites:
- Python 3.8+
- Required libraries (install via `requirements.txt`):
  - `bnlp`
  - `nltk`
  - `banglanltk`
  - `bnlp_toolkit`
  - `huggingface_hub`
  - `transformers`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `tensorflow`
  - `torch`
  - `scikit-learn`
  - `xgboost`
  - `lightgbm`

### Running the Project:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Sentiment-Analysis-ML_Fine-Tune-Llama-3.1.git

2. **Install dependencies:**
   pip install -r requirements.txt



## Future Improvements
- Data Augmentation: Augment the dataset with more samples to improve LSTM and other deep learning model performance.
- Hardware Upgrades: Utilize cloud-based GPU services for large model fine-tuning (e.g., AWS-SageMaker, Google Colab, Kaggle).
- Explore Lightweight Models: Investigate parameter-efficient models like QLoRA or 4-bit quantized models for fine-tuning on resource-constrained systems.
- NLP Toolkit Enhancements: Continue refining the preprocessing pipeline, especially for tasks like stemming and tokenization in Bengali.

## License
- This project is licensed under the MIT License. See the LICENSE file for more details.
