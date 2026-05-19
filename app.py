import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Iris.csv")

# Remove Id column
df = df.drop("Id", axis=1)

# Encode flower names
encoder = LabelEncoder()
df["Species"] = encoder.fit_transform(df["Species"])

# Features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit App
st.title("Iris Flower Classification App")

st.write("Enter flower measurements:")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width", 2.0, 5.0, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 3.0, 1.0)

prediction = model.predict([[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]])

flower_name = encoder.inverse_transform(prediction)

st.success(f"Predicted Flower Species: {flower_name[0]}")