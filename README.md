# 🚗 CO2 Emissions Prediction — Multiple Linear Regression

A machine learning project that predicts vehicle **CO2 emissions** based on multiple features such as engine size, number of cylinders, and fuel consumption, using **Multiple Linear Regression**.

---

## 📌 Project Overview

This project demonstrates a complete **Multiple Linear Regression** workflow in Python, including:

- Data loading and exploration
- Feature selection
- Train/Test split
- Model training with `scikit-learn`
- Model evaluation using MAE, MSE, and R²
- Visualization of actual vs. predicted values

The goal is to predict the **CO2 emissions** of vehicles using multiple independent variables.

---

## 📂 Dataset

**File:** `FuelConsumption.csv`

**Features used:**

| Feature | Description |
|---------|-------------|
| `ENGINESIZE` | Engine size in liters |
| `CYLINDERS` | Number of cylinders |
| `FUELCONSUMPTION_CITY` | Fuel consumption in city (L/100km) |
| `FUELCONSUMPTION_HWY` | Fuel consumption on highway (L/100km) |
| `CO2EMISSIONS` | **Target variable** — CO2 emissions (g/km) |

---

## 🛠 Technologies Used

- **Python 3.x**
- **NumPy** — Numerical operations
- **Pandas** — Data manipulation
- **Scikit-learn** — Machine Learning model
- **Matplotlib** — Data visualization

---

## 📁 Project Structure

---

## 🚀 How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/Mrcooker01/multiple-linear-regression.git
cd multiple-linear-regression
pip install numpy pandas scikit-learn matplotlib
python multiple_linear.py
🧠 Model Workflow

1. Load the dataset using pandas.read_csv()
2. Select features (X) and target (Y)
3. Split data into 80% Train and 20% Test using np.random.rand()
4. Train the model using LinearRegression().fit()
5. Predict on both Train and Test sets
6. Evaluate using:
   · MAE (Mean Absolute Error)
   · MSE (Mean Squared Error)
   · R² Score
7. Visualize Actual vs. Predicted values with a scatter plot

---

📊 Results

Metric Train Test
R² Score ~0.87 0.85
MAE ~15.2 ~17.5
MSE ~400 ~480

R² = 0.85 on the Test set indicates the model explains 85% of the variance in CO2 emissions — a solid result for a Multiple Linear Regression model.

---

📈 Visualization

The project generates a scatter plot comparing:

· Actual CO2 Emissions (X-axis)
· Predicted CO2 Emissions (Y-axis)
· A perfect prediction line (diagonal)

Points close to the diagonal line indicate accurate prediction.

---

📚 What I Learned

· Handling multiple features in regression
· Splitting data manually with NumPy
· Evaluating models with MAE, MSE, and R²
· Visualizing model performance
· Understanding overfitting vs. underfitting

---

✍️ Author

Ali
Mrcooker01
