# 📈 Sales Prediction using Python

### 🏢 Oasis Infobyte Data Science Internship — Task 5

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![ML](https://img.shields.io/badge/Machine%20Learning-Linear%20Regression-green?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

---

## 📌 Project Overview

This project predicts **product sales** based on advertising budgets
spent on **TV, Radio, and Newspaper** channels using
**Linear Regression** Machine Learning model.

The project includes a fully interactive **Streamlit web application**
where users can adjust ad budgets using sliders and get
**real-time sales predictions.**

---

## 🎯 Problem Statement

A company wants to know:
> *"If we spend X amount on TV, Radio, and Newspaper ads —
> how much sales can we expect?"*

This ML model answers that question with **90% accuracy.**

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Advertising Dataset |
| Rows | 200 |
| Columns | 4 (TV, Radio, Newspaper, Sales) |
| Target | Sales (in thousand units) |

**Features:**
- `TV` — Budget spent on TV advertising ($thousands)
- `Radio` — Budget spent on Radio advertising ($thousands)
- `Newspaper` — Budget spent on Newspaper advertising ($thousands)
- `Sales` — Product sales generated (thousand units) *(Target)*

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.x | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical plots |
| Scikit-learn | Machine Learning model |
| Streamlit | Interactive web application |

---

## 🧠 Model Details

- **Algorithm:** Linear Regression
- **Train/Test Split:** 80% / 20%
- **Random State:** 42

### 📈 Model Performance

| Metric | Value |
|--------|-------|
| R² Score | **0.90** |
| Mean Absolute Error | 1.46 |
| RMSE | 1.78 |

> ✅ R² Score of 0.90 means the model explains **90% of sales variation**

---

## 💡 Key Insights

- 📺 **TV** has the strongest correlation (0.78) with Sales
- 📻 **Radio** delivers the highest return per dollar spent (0.576)
- 📰 **Newspaper** has minimal impact on Sales (0.228)

---

## 🚀 How to Run

### Step 1 — Clone the repository
```bash
git clone https://github.com/YourUsername/OIBSIP.git
cd OIBSIP/Sales_Prediction
```

### Step 2 — Install dependencies
```bash
py -m pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### Step 3 — Run the Streamlit app
```bash
py -m streamlit run app.py
```

### Step 4 — Open in browser
http://localhost:8501/

---

## 📁 Project Structure
OIBSIP/
└── Sales_Prediction/
├── app.py                 # Streamlit web application
├── VinayGoud_Task5.py     # Core ML model script
├── Advertising.csv        # Dataset
├── README.md              # Project documentation
└── .streamlit/
└── config.toml    # Streamlit theme config

---

## 📸 App Preview

> Interactive web app with real-time sales prediction,
> data visualizations, and business insights dashboard.

---

## 👨‍💻 Author

**Vinay Goud**
3rd Year B.Tech — CSE (Data Science)
Malla Reddy College of Enginnering and Technology (MRCET)
Oasis Infobyte Data Science Internship

---

## 📜 License

This project is created for educational purposes as part of
the Oasis Infobyte Internship Program.