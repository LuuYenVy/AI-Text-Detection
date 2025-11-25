# AI-Text-Detection

This project was developed for the **Mercor AI Text Detection** competition on Kaggle. It focuses on building models to distinguish human-written text from AI-generated content using NLP and machine learning techniques.  
Link Competition: https://www.kaggle.com/competitions/mercor-ai-detection

🏆 **Achievement:** 3rd Place in the Mercor AI Text Detection competition on Kaggle.

## Contributions
- **Huỳnh Khả Tú:** Data preprocessing, cleaning, feature engineering, initial model building.  
- **Lưu Yến Vy:** Model development, training, hyperparameter tuning, evaluation, and submission pipeline.


## Pipeline Overview

### 1. Feature Extraction
- **Semantic embeddings:** Extracted from [RoBERTa](https://huggingface.co/roberta-base) using multi-layer mean pooling.
- **Linguistic features:** 
  - Text length
  - Punctuation frequency
  - Uppercase ratio
  - Stopword ratio
  - Digit ratio

These features capture both contextual meaning and stylistic patterns that transformers may overlook.

### 2. Base Models
- **Logistic Regression**
- **XGBoost**

Both models are trained on the combined feature set. We use **Stratified K-Fold cross-validation** to generate out-of-fold (OOF) predictions, reducing overfitting.

### 3. Stacking Meta-Model
- **Logistic Regression** as a stacking meta-model.
- Trained on OOF predictions from base models.
- Enhances generalization by combining multiple model perspectives.

### 4. Probability Calibration
- **Isotonic Regression** is applied to calibrate predicted probabilities for better reliability.

### 5. Final Prediction
- Blend two strong model variants using **rank-based assembling** with optimized weights.
- Apply **MinMax scaling** to prevent extreme probability values.

---

## Strengths
- Effective integration of deep embeddings, linguistic features, and diverse models.
- OOF-based stacking improves generalization.
- Rank-based blending provides stable and noise-resistant predictions.

## Possible Improvements
- Explore additional transformer variants to improve embedding diversity.
- Use stronger meta-models (e.g., LightGBM, small neural networks) for stacking.
- Optimize feature set and hyperparameters for XGBoost and blending weights.
- Further probability calibration and ensemble strategies could enhance final performance.

---
