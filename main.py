import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv("data.csv")

data = load_data()

# Separate features and target variable
X = data['Comment']
y = data['Emotion']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing techniques
preprocessors = {
    "No Preprocessing": None,
    "Z-score Normalization": StandardScaler(with_mean=False),
    "MaxAbs Normalization": MaxAbsScaler()
}

# Define decision tree parameters
tree_params = {
    "max_depth": [3, 5, 7, 9, 11],
    "splitter": ["best", "random"]
}

# User interface
st.title("Emotion Classification")

# Choose preprocessing technique
preprocessing_choice = st.selectbox("Choose preprocessing technique:", list(preprocessors.keys()))

# Choose decision tree parameters
max_depth = st.slider("Choose max depth:", min_value=3, max_value=11, value=5, step=1)
splitter = st.selectbox("Choose splitter:", ["best", "random"])

# Train decision tree model
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('preprocessor', preprocessors[preprocessing_choice]),
    ('classifier', DecisionTreeClassifier(max_depth=max_depth, splitter=splitter, random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"Accuracy: {accuracy:.2f}")
