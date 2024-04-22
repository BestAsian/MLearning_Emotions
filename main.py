import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

# Load
@st.cache_data
def load_data():
    return pd.read_csv(".venv/data.csv")


data = load_data()

sampled_data = pd.concat([data[data['label'] == label].sample(n=10000, random_state=42) for label in range(6)], ignore_index=True)
# Separate features and target variable
X = sampled_data['text']
y = sampled_data['label']
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


st.title("Emotion Classification")
# Define Classifier parameters, labels, and everything else
tree_params = {
    "max_depth": [3, 5, 7, 9, 11],
    "splitter": ["best", "random"]
}
svm_params = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf", "poly"],
    "gamma": ["scale", "auto"]
}
nb_params = {
    "alpha": [0.1, 0.5, 1.0]
}

preprocessors = {
    "No Preprocessing": None,
    "Z-score Normalization": StandardScaler(with_mean=False),
    "MaxAbs Normalization": MaxAbsScaler()
}
classification_options = {
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine": SVC(),
    "Naive Bayes": MultinomialNB()
}
emotion_labels = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Suprise"}
preprocessing_choice = st.selectbox("Choose preprocessing technique:", list(preprocessors.keys()))

classifier_choice = st.selectbox("Choose classification technique:", list(classification_options.keys()))

selected_classifier = classification_options[classifier_choice]
if classifier_choice == "Decision Tree":
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('preprocessor', preprocessors[preprocessing_choice]),
        ('classifier', selected_classifier)
    ])
else:
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('preprocessor', preprocessors[preprocessing_choice]),
        ('classifier', selected_classifier)
    ])
if classifier_choice == "Decision Tree":
    max_depth = st.slider("Choose max depth:", min_value=3, max_value=500, value=5, step=1)
    splitter = st.selectbox("Choose splitter:", ["best", "random"])
    pipeline.set_params(classifier__max_depth=max_depth, classifier__splitter=splitter)
elif classifier_choice == "Support Vector Machine":
    svm_params["kernel"] = st.selectbox("Choose kernel:", ["linear", "rbf", "poly"])
    pipeline.set_params(classifier__kernel=svm_params["kernel"])
elif classifier_choice == "Naive Bayes":
    nb_params["alpha"] = st.slider("Choose alpha:", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    pipeline.set_params(classifier__alpha=nb_params["alpha"])

pipeline.fit(X_train, y_train)

# Accuracy
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")

st.title("DEMO")
# Text input for classification
text_input = st.text_input("Enter text to classify:", "")

predicted_emotion = None
while True:
    # Classify the input text when a button is clicked
    if st.button("Classify"):
        prediction = pipeline.predict([text_input])
        predicted_emotion = emotion_labels[prediction[0]]
        st.write(f"Predicted Emotion: {predicted_emotion}")

    # Let the user choose the correct emotion
    if predicted_emotion is not None:
        correct_emotion = st.selectbox("Select the correct emotion:", list(emotion_labels.values()))
        if st.button("Update Dataset"):
            correct_label = {v: k for k, v in emotion_labels.items()}[correct_emotion]
            new_entry = pd.DataFrame({'text': [text_input], 'label': [correct_label]})
            data = pd.concat([data, new_entry], ignore_index=False)
            # Save the updated dataset back to the CSV file
            data.to_csv(".venv/data.csv", index=True)
            break
