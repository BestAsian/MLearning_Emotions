import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# Load the dataset
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
# Define
tree_params = {
    "max_depth": [3, 5, 7, 9, 11],
    "splitter": ["best", "random"]
}
preprocessors = {
    "No Preprocessing": None,
    "Z-score Normalization": StandardScaler(with_mean=False),
    "MaxAbs Normalization": MaxAbsScaler()
}
emotion_labels = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Suprise"}
preprocessing_choice = st.selectbox("Choose preprocessing technique:", list(preprocessors.keys()))
max_depth = st.slider("Choose max depth:", min_value=3, max_value=500, value=5, step=1)
splitter = st.selectbox("Choose splitter:", ["best", "random"])


# Pipeline and fitting
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('preprocessor', preprocessors[preprocessing_choice]),
    ('classifier', DecisionTreeClassifier(max_depth=max_depth, splitter=splitter, random_state=42))
])
pipeline.fit(X_train, y_train)



# Accuracy
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")

st.title("DEMO")
# Text input for classification
text_input = st.text_input("Enter text to classify:", "")

# Classify the input text when a button is clicked
if st.button("Classify"):
    prediction = pipeline.predict([text_input])
    predicted_emotion = emotion_labels[prediction[0]]
    st.write(f"Predicted Emotion: {predicted_emotion}")
