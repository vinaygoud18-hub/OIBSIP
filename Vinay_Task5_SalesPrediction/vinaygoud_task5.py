# Sales Prediction using Python
# Oasis Infobyte Internship - Task 5
# Author: Vinay Goud

# Step 2 - Load the Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('Advertising.csv', index_col=0)
print(df.head())
print("\nShape of dataset:", df.shape)

# Step 3 - EDA
print("\nColumns:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nStatistical Summary:\n", df.describe())

# Step 4 - Visualize
sns.set_style("whitegrid")
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'],
             y_vars='Sales', height=4, aspect=0.8, kind='reg')
plt.suptitle("Ad Spend vs Sales", y=1.02)
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Step 5 - Prepare Data
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)

# Step 6 - Train Model
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel Trained Successfully! ✅")
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Step 7 - Evaluate Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation Results:")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2 Score : {r2:.2f}")

# Step 8 - Predict New Values
new_data = pd.DataFrame({
    'TV': [230.1, 50.0, 100.0],
    'Radio': [37.8, 10.0, 25.0],
    'Newspaper': [69.2, 5.0, 30.0]
})
predicted_sales = model.predict(new_data)
print("\nSales Predictions:")
print("-" * 40)
for i, sales in enumerate(predicted_sales):
    print(f"Campaign {i+1}: Predicted Sales = {sales:.2f} thousand units")