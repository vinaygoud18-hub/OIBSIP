# 🚗 Car Price Prediction using Machine Learning

### 🏢 Oasis Infobyte Data Science Internship — Task 3

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![ML](https://img.shields.io/badge/ML-Random%20Forest-green?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

---

## 📌 Project Overview

This project predicts the **selling price of used cars**
based on features like year, fuel type, transmission,
mileage and more using **Random Forest** Machine Learning.

Includes a fully interactive **Streamlit web app** where
users can select car features and get instant price predictions.

---

## 🎯 Problem Statement

> *"Given a used car's features — what is the
> expected resale price in the market?"*

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Used Car Prices Dataset |
| Target | Selling Price (₹) |
| Algorithm | Random Forest Regressor |

**Key Features:**
- `Year` — Manufacturing year
- `Fuel Type` — Petrol / Diesel / CNG
- `Transmission` — Manual / Automatic
- `Mileage` — Kilometres driven
- `Engine` — Engine capacity (CC)
- `Power` — Max power (bhp)
- `Seats` — Number of seats

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.x | Core programming |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | ML model (Random Forest) |
| Plotly | Interactive charts |
| Streamlit | Web dashboard |

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest Regressor |
| R² Score | **~0.90+** |
| Train/Test Split | 80% / 20% |

---

## 💡 Key Insights

- 🌲 Random Forest outperforms Linear Regression
- 📅 Car age is the strongest price predictor
- ⛽ Diesel & automatic cars have higher resale value
- 📉 Higher mileage reduces car price significantly

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
py -m pip install pandas numpy scikit-learn plotly streamlit
```

### Step 2 — Run core analysis
```bash
py VinayGoud_Task3.py
```

### Step 3 — Run Streamlit app
```bash
py -m streamlit run app.py
```

---

## 📁 Project Structure
OIBSIP/
└── Vinay_Task3_CarPricePrediction/
├── app.py
├── VinayGoud_Task3.py
├── README.md
└── .streamlit/
└── config.toml

---

## 👨‍💻 Author

**Vinay Goud**
3rd Year B.Tech — CSE (Data Science)
MRCET, Hyderabad
Oasis Infobyte Data Science Internship

---

## 📜 License
Educational purpose — Oasis Infobyte Internship Program.