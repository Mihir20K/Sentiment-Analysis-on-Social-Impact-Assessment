from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

# Apply SMOTE to address class imbalance
def apply_smote(X_train, y_train):
    oversampler = SMOTE(random_state=42)
    X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)
    return X_train_oversampled, y_train_oversampled

# Encode target variables
def encode_labels(y_train, y_test):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    return y_train_encoded, y_test_encoded

# Create an ensemble model using VotingClassifier
def create_ensemble_model():
    model1 = RandomForestClassifier()
    model2 = MultinomialNB()
    ensemble = VotingClassifier(estimators=[('rf', model1), ('nb', model2)], voting='hard')
    return ensemble

# Evaluate the model and generate a classification report
def evaluate_model(y_test, y_pred):
    classification_rep = classification_report(y_test, y_pred)
    return classification_rep
