
Here’s a modified project overview suitable for adding to a resume without any personal information:

Project: Fake News Detection in Python

Overview
Developed a Fake News Detection system using various natural language processing (NLP) techniques and machine learning algorithms. The project utilizes the Scikit-learn Python library to classify news articles as "True" or "False."

Key Contributions

Implemented pre-processing steps, including tokenization and stemming, for data preparation in Python.
Developed feature extraction and selection methods, including Bag-of-Words, N-grams, TF-IDF, and experimented with word2vec and POS tagging for enhanced feature sets.
Built multiple machine learning classifiers (Naive Bayes, Logistic Regression, Linear SVM, Stochastic Gradient Descent, and Random Forest) using Scikit-learn. Achieved optimal model selection through F1-score comparisons and confusion matrix analysis.
Conducted hyperparameter tuning using GridSearchCV, selecting Logistic Regression as the final model for deployment, and extracted top 50 TF-IDF features to understand key indicators of fake news.
Deployed a trained model for real-time prediction, allowing user-inputted news headlines to be classified with an associated probability of truth.
Dataset
Utilized the LIAR benchmark dataset for fake news detection. The dataset was modified to contain binary labels for simplicity.

Technologies

Programming Language: Python
Libraries: Scikit-learn, Numpy, SciPy
Deployment: Saved final Logistic Regression model as a .sav file for reuse and real-time classification through a script interface.
Next Steps
Future enhancements planned include adding more NLP-based feature engineering techniques, such as topic modeling and POS tagging, to improve accuracy, as well as expanding training data size.

Installation and Execution

Clone the repository and install required packages via pip or Anaconda.
Run prediction.py to classify news headlines, receiving both a truth classification and a probability score.
