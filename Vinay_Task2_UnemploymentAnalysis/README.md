# 📊 Unemployment Analysis with Python

### 🏢 Oasis Infobyte Data Science Internship — Task 2

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat&logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-orange?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

---

## 📌 Project Overview

This project analyzes **unemployment rates across Indian states**
during the COVID-19 pandemic (2020) using Python and Data Science.
It includes an interactive **Streamlit dashboard** with
state-wise filters and visualizations.

---

## 🎯 Problem Statement

> *"How did COVID-19 affect unemployment across different
> Indian states? Which regions were most impacted?
> How did Urban vs Rural areas compare?"*

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | India Unemployment Dataset |
| Rows | 267 |
| Columns | 7 |
| Period | 2020 (COVID-19 Era) |

**Features:**
- `State` — Indian state name
- `Date` — Month and year
- `Unemployment_Rate` — Estimated unemployment (%)
- `Employed` — Number of employed people
- `Labour_Participation_Rate` — Labour participation (%)
- `Area` — Urban or Rural

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.x | Core programming |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Static charts |
| Seaborn | Statistical plots |
| Plotly | Interactive charts |
| Streamlit | Web dashboard |

---

## 📈 Key Insights

- 📈 Unemployment **spiked in April–May 2020** during COVID lockdown
- 🗺️ **Tripura, Haryana & Jharkhand** had highest unemployment
- 🏙️ **Urban areas** were more affected than rural areas
- 📉 Recovery began from **June 2020** onwards

---

## 🚀 How to Run

### Step 1 — Clone the repository
```bash
git clone https://github.com/YourUsername/OIBSIP.git
cd OIBSIP/Unemployment_Analysis
```

### Step 2 — Install dependencies
```bash
py -m pip install pandas numpy matplotlib seaborn plotly streamlit
```

### Step 3 — Run core analysis
```bash
py vinaygoud_task2.py
```

### Step 4 — Run Streamlit dashboard
```bash
py -m streamlit run app.py
```

---

## 📁 Project Structure

OIBSIP/
└── Unemployment_Analysis/
├── app.py                 # Streamlit dashboard
├── vinaygoud_task2.py     # Core analysis script
├── README.md              # Documentation
└── .streamlit/
└── config.toml   # Theme config

---

## 👨‍💻 Author

**Vinay Goud**
3rd Year B.Tech — CSE (Data Science)
Malla Reddy College of Engineering and Technology (MRCET)
Oasis Infobyte Data Science Internship

---

## 📜 License

This project is created for educational purposes
as part of the Oasis Infobyte Internship Program.