# Car Price Prediction using Machine Learning
# Oasis Infobyte Data Science Internship — Task 3
# Author: Vinay Goud | 3rd Year CSE (Data Science) — MRCET

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   CAR PRICE PREDICTION — OASIS INFOBYTE TASK 3")
print("=" * 60)

# -------------------------------------------------------
# STEP 1 — Load Dataset
# -------------------------------------------------------
df = pd.read_csv("car_data.csv")

print("\n📌 First 5 Rows:")
print(df.head())
print("\n📌 Shape:", df.shape)
print("\n📌 Columns:", df.columns.tolist())
print("\n📌 Data Types:\n", df.dtypes)
print("\n📌 Missing Values:\n", df.isnull().sum())

# -------------------------------------------------------
# STEP 2 — Clean Data
# -------------------------------------------------------
print("\n🔧 Cleaning Data...")

# Drop duplicates
df.drop_duplicates(inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

print("✅ Columns after cleaning:", df.columns.tolist())

# Identify target column (price)
# Common column names in car datasets
price_col = None
for col in df.columns:
    if 'price' in col.lower() or 'selling' in col.lower():
        price_col = col
        break

if price_col is None:
    price_col = df.columns[-1]

print(f"✅ Target column identified: {price_col}")
print("\n📌 Statistical Summary:")
print(df.describe())

# -------------------------------------------------------
# STEP 3 — Feature Engineering
# -------------------------------------------------------
print("\n⚙️ Feature Engineering...")

# Encode categorical columns
le = LabelEncoder()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"📌 Categorical columns: {cat_cols}")

for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print("✅ Encoding done!")

# -------------------------------------------------------
# STEP 4 — Prepare Features
# -------------------------------------------------------
X = df.drop(columns=[price_col])
y = df[price_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\n📌 Training size: {X_train.shape}")
print(f"📌 Testing size : {X_test.shape}")

# -------------------------------------------------------
# STEP 5 — Train Models
# -------------------------------------------------------
print("\n🤖 Training Models...")

# Model 1 — Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
print(f"✅ Linear Regression R² Score: {lr_r2:.2f}")

# Model 2 — Random Forest (better for car prices)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
print(f"✅ Random Forest R²     Score: {rf_r2:.2f}")

# Use best model
best_model = rf if rf_r2 > lr_r2 else lr
best_pred = rf_pred if rf_r2 > lr_r2 else lr_pred
best_name = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"
print(f"\n🏆 Best Model: {best_name}")

# -------------------------------------------------------
# STEP 6 — Evaluate Best Model
# -------------------------------------------------------
mae  = mean_absolute_error(y_test, best_pred)
mse  = mean_squared_error(y_test, best_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, best_pred)

print("\n📊 Model Evaluation Results:")
print(f"   MAE      : {mae:.2f}")
print(f"   MSE      : {mse:.2f}")
print(f"   RMSE     : {rmse:.2f}")
print(f"   R² Score : {r2:.2f}")

# -------------------------------------------------------
# STEP 7 — Feature Importance
# -------------------------------------------------------
if best_name == "Random Forest":
    importances = pd.Series(
        rf.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    importances.head(10).plot(kind='bar', color='#1a73e8', edgecolor='white')
    plt.title('Top 10 Feature Importances', fontsize=13,
              fontweight='bold')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Chart saved: feature_importance.png")

# -------------------------------------------------------
# STEP 8 — Actual vs Predicted Plot
# -------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(y_test, best_pred, color='#1a73e8', alpha=0.5, s=30)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='#ea4335', linewidth=2, linestyle='--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Car Price', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart saved: actual_vs_predicted.png")

print("\n" + "=" * 60)
print("   ✅ ANALYSIS COMPLETE!")
print("=" * 60)