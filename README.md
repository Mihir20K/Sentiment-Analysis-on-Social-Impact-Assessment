# Sentiment Analysis on Social Impact Assessment using Machine Learning

## Project Overview
This project focuses on sentiment analysis of social media posts related to social impact topics. It aims to classify the sentiment of the posts as positive, negative, or neutral, providing valuable insights into public opinions on social impact issues.

The key steps include:

- Data Preprocessing: Text data is cleaned by removing URLs, stopwords, and punctuation. Tokenization and lemmatization techniques are applied.

- Dataset Splitting: The preprocessed data is split into training and test sets for model evaluation.

- Model Building: A machine learning pipeline is created using a TfidfVectorizer for text vectorization and a RandomForestClassifier for sentiment classification.

- Hyperparameter Tuning: RandomizedSearchCV is utilized to find the best hyperparameters for the model.

- Handling Class Imbalance: The SMOTE technique is employed to address class imbalance in the training data.

- Model Training and Evaluation: The model is trained on the training set and evaluated on the test set using classification metrics.

- By analyzing the sentiment of social media posts, this project provides insights into public opinions on social impact topics, helping to understand the sentiment landscape around these issues

## Project Structure
- `data/`: [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-Click%20here-darkgreen.svg)](https://github.com/Mihir20K/Sentiment-Analysis-on-Social-Impact-Assessment/blob/main/train.csv)

- `notebooks/`: [![Open in nbviewer](https://img.shields.io/badge/Open%20in%20nbviewer-Click%20here-blue.svg)](https://nbviewer.jupyter.org/github/Mihir20K/Sentiment-Analysis-on-Social-Impact-Assessment/blob/main/Sentiment_Analysis_on_Social_Impact_Assessment.ipynb)


- `README.md`: [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-Click%20here-red.svg)](https://github.com/Mihir20K/Sentiment-Analysis-on-Social-Impact-Assessment/edit/main/README.md)

## Graphical Visualization of Sentiment Distribution
![Sentiment Distribution](https://github.com/Mihir20K/Sentiment-Analysis-on-Social-Impact-Assessment/blob/main/sentiment_distribution.png)

