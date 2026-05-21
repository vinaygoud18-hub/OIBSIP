# Unemployment Analysis with Python
# Oasis Infobyte Data Science Internship — Task 2
# Author: Vinay Goud | 3rd Year CSE (Data Science) — MRCET

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Load Dataset ----
df = pd.read_csv("Unemployment.csv")

print("=" * 50)
print("UNEMPLOYMENT ANALYSIS WITH PYTHON")
print("=" * 50)

# ---- Basic Info ----
print("\n📌 First 5 Rows:")
print(df.head())

print("\n📌 Shape:", df.shape)
print("\n📌 Columns:", df.columns.tolist())
print("\n📌 Data Types:\n", df.dtypes)
print("\n📌 Missing Values:\n", df.isnull().sum())

# ---- Clean Column Names ----
df.columns = df.columns.str.strip()
df.rename(columns={
    'Region': 'State',
    'Date': 'Date',
    'Frequency': 'Frequency',
    'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate',
    'Area': 'Area'
}, inplace=True)

# ---- Convert Date ----
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Month_Name'] = df['Date'].dt.strftime('%b %Y')

print("\n✅ Data Cleaned Successfully!")
print("\n📌 Statistical Summary:")
print(df.describe())

# ---- Average Unemployment by State ----
state_unemployment = df.groupby('State')['Unemployment_Rate'].mean().sort_values(ascending=False)
print("\n📌 Top 5 States by Unemployment Rate:")
print(state_unemployment.head())

# ---- Plot 1: Top 10 States ----
plt.figure(figsize=(12, 6))
top_states = state_unemployment.head(10)
sns.barplot(x=top_states.values, y=top_states.index, palette='Reds_r')
plt.title('Top 10 States by Average Unemployment Rate', fontsize=14, fontweight='bold')
plt.xlabel('Average Unemployment Rate (%)')
plt.ylabel('State')
plt.tight_layout()
plt.savefig('top_states.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart 1 saved: top_states.png")

# ---- Plot 2: Monthly Trend ----
monthly = df.groupby('Date')['Unemployment_Rate'].mean().reset_index()
plt.figure(figsize=(12, 5))
plt.plot(monthly['Date'], monthly['Unemployment_Rate'],
         color='#1a73e8', linewidth=2.5, marker='o', markersize=4)
plt.fill_between(monthly['Date'], monthly['Unemployment_Rate'],
                 alpha=0.1, color='#1a73e8')
plt.title('Monthly Unemployment Rate Trend in India', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('monthly_trend.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart 2 saved: monthly_trend.png")

# ---- Plot 3: Urban vs Rural ----
area_unemployment = df.groupby('Area')['Unemployment_Rate'].mean()
plt.figure(figsize=(6, 5))
colors = ['#1a73e8', '#34a853']
plt.pie(area_unemployment.values, labels=area_unemployment.index,
        autopct='%1.1f%%', colors=colors, startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
plt.title('Urban vs Rural Unemployment Rate', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('urban_rural.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart 3 saved: urban_rural.png")

# ---- Plot 4: Heatmap ----
pivot = df.pivot_table(
    values='Unemployment_Rate',
    index='State',
    columns='Month',
    aggfunc='mean'
)
plt.figure(figsize=(14, 10))
sns.heatmap(pivot, cmap='YlOrRd', annot=False, linewidths=0.3)
plt.title('State-wise Monthly Unemployment Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('State')
plt.tight_layout()
plt.savefig('heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart 4 saved: heatmap.png")

print("\n" + "=" * 50)
print("✅ ANALYSIS COMPLETE!")
print("=" * 50)