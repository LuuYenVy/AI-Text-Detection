# AI-Text-Detection

This project was developed for the **Mercor AI Text Detection** competition on Kaggle. It focuses on building models to distinguish human-written text from AI-generated content using NLP and machine learning techniques.  

🏆 **Achievement:** 3rd Place in the Mercor AI Text Detection competition on Kaggle.

## Contributions
- **Huỳnh Khả Tú:** Data preprocessing, cleaning, feature engineering, initial model building.  
- **Lưu Yến Vy:** Model development, training, hyperparameter tuning, evaluation, and submission pipeline.


flowchart TD
    A([📄 Raw Text Data])
    B([🔢 Text Features])
    C([🤖 RoBERTa Embeddings])
    D([🧩 Combine Features & Embeddings])
    E([⚡ Base Models: Logistic Regression & XGBoost])
    F([🔀 Stacking Meta-Model])
    G([📏 Probability Calibration])
    H([🎯 Rank-Based Blending])
    I([📊 Evaluate & Save Submission])

    A --> B
    A --> C
    B --> D
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I

