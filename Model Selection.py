from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

# Split the dataset into training and test sets
def split_dataset(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Create a pipeline for vectorization, model selection, and hyperparameter tuning
def build_pipeline():
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', RandomForestClassifier())
    ])
    return pipeline

# Perform hyperparameter tuning using RandomizedSearchCV
def perform_hyperparameter_tuning(X_train, y_train):
    pipeline = build_pipeline()
    parameters = {
        'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vectorizer__min_df': [1, 2, 3],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 5, 10],
    }
    random_search = RandomizedSearchCV(pipeline, parameters, cv=5, n_iter=3, random_state=42)
    random_search.fit(X_train, y_train)
    return random_search.best_params_
