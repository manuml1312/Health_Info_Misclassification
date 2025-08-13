# Health Information Misclassification using a fine-tuned LLM

The rapid spread of health-related misinformation on digital platforms poses a 
critical threat to public well‐being, particularly in settings with limited annotated 
corpora and language‐specific NLP tools. This study aims to evaluate the effectiveness 
of transformer‐based models for detecting four categories of misinformation—Reliable, Fake, 
Unreliable, and “Not Enough Info”—in low‐resourced public health datasets. We first compiled 
and preprocessed a novel corpus of 8 927 English‐language instances by merging six 
publicly available health‐misinformation datasets, stratifying labels into a consistent 
four‐way taxonomy. Using Hugging-Face Transformers, we fine-tuned three architectures—DistilBERT, 
BERT, and RoBERTa—on an 80/20 train–test split, optimizing hyperparameters such as learning rate,
batch size, and weight decay. Model performance was assessed via accuracy, precision, recall, 
F1-score, and cross-entropy loss, and embedding separability was visualized through 2D and 3D t-SNE
projections and SHAP token-importance plots. DistilBERT achieved 75.7 % accuracy (macro-F1 = 0.74), 
BERT reached 82.1 % (macro-F1 = 0.82), and RoBERTa attained the highest accuracy of 85.7 % (macro-F1 = 0.80). 
While transformer embeddings distinctly clustered the “Not Enough Info” and “Unreliable” classes, 
“Fake” and “Reliable” remained partially overlapping, indicating semantic proximity. These findings 
demonstrate that deep contextual models substantially outperform lighter baselines but still face 
challenges in discriminating nuanced misinformation. Future work should explore data augmentation, 
ensembling, domain‐adaptive pre-training, and multilingual extensions. By enhancing automated 
detection of public health misinformation, our results offer a scalable foundation for real-time 
monitoring systems and targeted interventions in resource‐constrained environments.

## Models Used
1. BeRT
2. RoBERTa
3. Distill BeRT

## Results
Achieved an accuracy of 84% on BeRT to correctly classify misinformation.
