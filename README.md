# AI-Text-Detection

This project was developed for the **Mercor AI Text Detection** competition on Kaggle. It focuses on building models to distinguish human-written text from AI-generated content using NLP and machine learning techniques.  

🏆 **Achievement:** 3rd Place in the Mercor AI Text Detection competition on Kaggle.

## Contributions
- **Huỳnh Khả Tú:** Data preprocessing, cleaning, feature engineering, initial model building.  
- **Lưu Yến Vy:** Model development, training, hyperparameter tuning, evaluation, and submission pipeline.


flowchart LR
    A([📄 Raw Text Data]) --> B([🔢 Text Features])
    A --> C([🤖 RoBERTa Embeddings])
    B & C --> D([🧩 Combine Features & Embeddings])
    D --> E([⚡ Base Models: Logistic Regression & XGBoost])
    E --> F([🔀 Stacking Meta-Model])
    F --> G([📏 Probability Calibration])
    G --> H([🎯 Rank-Based Blending])
    H --> I([📊 Evaluate & Save Submission])

