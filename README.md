# NLP for Tracking Misinformation in Low-Resourced Public Health Datasets

This repository contains the code, data processing pipelines, experiments, and visualizations for the project “NLP for Tracking Misinformation in Low-Resourced Public Health Datasets.” The project explores the application of state-of-the-art transformer models—DistilBERT, BERT, and RoBERTa—for detecting health-related misinformation in settings where annotated corpora and domain-specific NLP tools are limited.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Visualization](#visualization)
- [Discussion and Future Work](#discussion-and-future-work)
- [Reproducibility](#reproducibility)
- [Author Contributions](#author-contributions)
- [Data Availability](#data-availability)
- [References](#references)
- [Appendix](#appendix)

## Project Overview

This project aims to evaluate transformer-based architectures for classifying health misinformation into four categories:
- **Unreliable**
- **Fake**
- **Reliable**
- **Not Enough Info**

The study leverages a novel corpus of 8,927 English-language instances obtained by merging six publicly available health-misinformation datasets. After standardizing the label taxonomy, models were fine-tuned using Hugging Face Transformers to answer the core challenge: how to effectively track misinformation in low-resourced public health domains.

Key motivations include:
- Addressing the rapid spread of health-related misinformation.
- Evaluating the effectiveness of deep contextual models in resource-constrained settings.
- Providing a scalable foundation for real-time monitoring systems and targeted interventions.

## Dataset Description

The final corpus consists of 8,927 instances with the following label distribution:
- **Unreliable (Code 0):** 1,794 instances (20.1%)
- **Fake (Code 1):** 2,605 instances (29.2%)
- **Reliable (Code 2):** 3,343 instances (37.4%)
- **Not Enough Info (Code 3):** 1,185 instances (13.3%)

Additional Details:
- **Text:** Cleaned natural-language content (lowercased, punctuation-stripped).
- **Tokenizer:** RoBERTa-base subword tokenizer (vocabulary size = 50,265).
- **Document Length:** Ranges from 5 to 512 tokens (median of 84 tokens).

A bar chart of the top-20 most frequent tokens (e.g., “covid”, “vaccine”, “health”) is provided in the Appendix (Figure A1).

## Experimental Setup

### Data Splitting and Preprocessing
- **Training Set:** 80% of the corpus (7,142 instances)
- **Evaluation Set:** 20% hold-out (1,785 instances)
- **Stratified Split:** Maintained label proportions with random seed = 42.
- **Preprocessing:** Texts were truncated or padded to 512 subword tokens according to transformer model requirements.

### Model Training
Models Fine-Tuned:
1. **DistilBERT** (“distilbert-base-uncased”)
2. **BERT** (“bert-base-uncased”)
3. **RoBERTa** (“roberta-base”)

Example hyperparameters (RoBERTa):
- Epochs: 5
- Batch Size (train/eval): 32/32
- Learning Rate: 2×10⁻⁵
- Weight Decay: 0.01
- Logging, evaluation, and save intervals: every 100 steps with a save total limit of 2.

The training process involved:
1. Tokenizing text using the respective AutoTokenizer.
2. Instantiating a classification model with four output labels.
3. Fine-tuning using the Hugging Face `Trainer` API.
4. Saving the final model state for further evaluation.

### Evaluation Metrics
Primary metrics included:
- Accuracy
- Precision, Recall, F1-score (per class and averaged)
- Cross-entropy loss
- Throughput (samples/second during evaluation)

Visualizations such as 2D and 3D t-SNE projections and SHAP token-importance plots were also employed for a qualitative assessment of the learned embeddings.

## Results

### DistilBERT
- **Eval Loss:** 0.6398
- **Accuracy:** 75.7%
- **Macro-F1:** 0.74
- **Throughput:** ~68 samples/s

_Class-Level Performance:_
- Unreliable: F1 = 0.72
- Fake: F1 = 0.59
- Reliable: F1 = 0.78
- Not Enough Info: F1 = 0.88

### BERT (bert-base-uncased)
- **Eval Loss:** 0.7536
- **Accuracy:** 82.1%
- **Macro-F1:** 0.82
- **Throughput:** ~35 samples/s

_Class-Level Performance:_
- Unreliable: F1 = 0.89
- Fake: F1 = 0.84
- Reliable: F1 = 0.77
- Not Enough Info: F1 = 0.75

### RoBERTa (roberta-base)
- **Eval Loss:** 0.4184
- **Accuracy:** 85.7%
- **Macro-F1:** 0.80
- **Throughput:** ~35.9 samples/s

_Class-Level Performance:_
- Unreliable: F1 = 0.86
- Fake: F1 = 0.60
- Reliable: F1 = 0.81
- Not Enough Info: F1 = 0.95

### Comparative Analysis

| Model       | Eval Loss | Accuracy | Macro-F1 | Weighted-F1 | Speed (samples/s) |
|-------------|-----------|----------|----------|-------------|-------------------|
| DistilBERT  | 0.6398    | 75.7%    | 0.74     | 0.76        | 68.3              |
| BERT        | 0.7536    | 82.1%    | 0.82     | 0.82        | 35.0              |
| RoBERTa     | 0.4184    | 85.7%    | 0.80     | 0.86        | 35.9              |

*Summary:*  
While DistilBERT offers the highest inference speed, it suffers from lower accuracy and F1 performance compared to full-sized BERT and RoBERTa. BERT provides the best macro-F1 score, especially improving detection in the “Fake” category, whereas RoBERTa delivers the highest overall accuracy and lowest loss, albeit with marginal challenges in differentiating “Fake” content from “Reliable” information.

## Visualization

- **Figures 1 & 2:** 2D and 3D t-SNE projections of RoBERTa’s [CLS] embeddings show clear separation for "Not Enough Info" and "Unreliable" classes, while "Fake" and "Reliable" overlap in certain regions.
- **Figures 3 & 4:** DistilBERT’s t-SNE embeddings are more diffuse, reflecting its lower accuracy and significant class overlap.

Detailed SHAP token-importance plots for selected classes are included in the Appendix (Figures A2 and A3).

## Discussion and Future Work

The study demonstrates that transformer-based architectures significantly outperform traditional baselines in detecting health misinformation, yet challenges remain—particularly in distinguishing semantically overlapping categories like “Fake” and “Reliable.”

**Limitations:**
1. Class imbalance, notably in the “Not Enough Info” category.
2. English-only corpus restricting cross-cultural generalization.
3. High resource requirements for computationally heavy models (BERT, RoBERTa).
4. Domain specific vocabulary and evolving misinformation narratives.

**Recommendations for Future Work:**
- Data augmentation and re-sampling strategies (e.g., SMOTE).
- Ensembling methods and knowledge distillation to optimize performance/speed trade-offs.
- Domain-adaptive pre-training using large-scale public health corpora.
- Incorporating multimodal data and metadata.
- Extending techniques to cross-lingual and low-resource settings.
- Implementing continual learning to address domain drift.

## Reproducibility

All code, preprocessed data, and model weights are provided through Jupyter notebooks:
- `Model_training.ipynb`
- `Model_training_3.ipynb`
- `Final_Code_Model_training_3.ipynb`

These notebooks are designed to run in a GPU-enabled Colab or local environment.

## Author Contributions

- **Agnivo Basu:** Literature review, data collection, implementation and fine-tuning of transformer models (BERT and RoBERTa), generation of t-SNE and SHAP visualizations, drafting the Methodology, Results, and Discussion sections.
- **Manu ML:** Data preprocessing, experimental design, statistical analysis, implementation and fine-tuning of models (BERT and DistilBERT), and preparation of the Dataset Description and Experimental Setup sections.

## Data Availability

All source datasets are publicly accessible and detailed in Table 1 of the manuscript. The combined pre-processed dataset (8,927 instances), model weights, and training code are provided within this repository.

## References

- Al-Tarawneh, M. A. B., et al. (2024). *Enhancing fake news detection with word embedding: A machine learning and deep learning approach.* Computers, 13, 239. DOI: 10.3390/computers13090239
- Brown, R. C. H., & de Barra, M. (2023). *A taxonomy of non-honesty in public health communication.* Public Health Ethics, 16(1), 86–101. DOI: 10.1093/phe/phad003
- Dai, E., Sun, Y., & Wang, S. (2020). *Ginger cannot cure cancer: Battling fake health news with a comprehensive data repository.* In Proceedings of ICWSM 2020, 853–857.
- Oubenali, N., et al. (2022). *Visualization of medical concepts represented using word embeddings: A scoping review.* BMC Medical Informatics and Decision Making, 22, 83. DOI: 10.1186/s12911-022-01822-9
- Vijaykumar, S., et al. (2021). *How shades of truth and age affect responses to COVID-19 (mis)information.* Humanities and Social Sciences Communications, 8, 88. DOI: 10.1057/s41599-021-00752-7
- Wardle, C., & Derakhshan, H. (2017). *Information disorder: Toward an interdisciplinary framework for research and policy making.* Council of Europe Report.

## Appendix

### Datasets Explored but Not Considered
```list type="issue"
data:
- url: "https://huggingface.co/datasets/ComplexDataLab/Misinfo_Datasets"
  title: "MisInfo Dataset"
- url: "https://ieee-dataport.org/documents/dfnd-dravidianfake-news-data"
  title: "Dravidian Fake News Dataset"
- url: "https://github.com/dwadden/scifact-open/blob/main/doc/data.md"
  title: "SciFact"
- url: "https://github.com/sakibsh/ANTiVax"
  title: "ANTiVax"
- url: "https://github.com/UBC-NLP/megacov/tree/master/tweet_ids"
  title: "MEGA-Cov"
- url: "https://github.com/kinit-sk/multiclaim"
  title: "Multiclaim"
- url: "https://github.com/several27/FakeNewsCorpus?tab=readme-ov-file"
  title: "Fake News Corpus"
- url: "https://www.kaggle.com/code/khsamaha/reddit-vaccine-myths-eda-and-text-analysis-r/report"
  title: "Bing, NRC, Afinn, Reddit Vaccine myth"
- url: "https://toolbox.google.com/factcheck/apis"
  title: "Google Fact Checker"
- url: "https://www.snopes.com/factbot/"
  title: "Snopes FacBot"
- url: "https://github.com/emilurosev/isot-fake-news-dataset"
  title: "ISOT Fake News Dataset"
```

### Figures & Additional Visualizations
- **Figure A1:** Bar chart showing the top-20 most frequent tokens.
- **Figure A2:** SHAP values for label_0 (Unreliable) using DistilBERT.
- **Figure A3:** SHAP values for label_3 (Not Enough Info) using RoBERTa.

For further details, please refer to the accompanying notebooks and supplementary materials.
