# 📈 Polynomial Regression

A two-part regression project: a visual demonstration of **Polynomial
Regression vs Linear Regression** on a toy dataset, followed by a
real-world **insurance charges prediction** model using multiple linear
regression with categorical feature encoding.

## 📋 Overview

### Part 1 — Linear vs Polynomial Regression (Toy Dataset)
A small synthetic dataset (`Experience` → `Salary`) is used to show why a
plain straight-line fit fails on curved data, and how transforming the
input with `PolynomialFeatures` (degree 3) lets a linear model fit a
non-linear relationship much more accurately.

### Part 2 — Insurance Charges Prediction (Real Dataset)
Using `insurance.csv`, the notebook:
1. Loads and inspects the data (info, duplicates)
2. Removes duplicate rows and resets the index
3. Splits features (`age`, `sex`, `bmi`, `children`, `smoker`, `region`) from the target (`charges`)
4. One-hot encodes the categorical columns (`sex`, `smoker`, `region`)
5. Combines encoded and numerical columns into a final feature set
6. Demonstrates encoding a new, user-entered patient record the same way, ready for prediction

## 📊 Dataset

`insurance.csv` — a public medical cost dataset (1338 records) with columns:

| Column   | Description                              |
| -------- | ----------------------------------------- |
| age      | Age of the primary beneficiary            |
| sex      | Gender (male/female)                      |
| bmi      | Body mass index                           |
| children | Number of dependents covered              |
| smoker   | Smoking status (yes/no)                   |
| region   | Residential area in the US                |
| charges  | Individual medical costs billed (target)  |

> ⚠️ **This CSV file is not included in the notebook export.** Place
> `insurance.csv` in the same folder as the notebook before running Part 2.
> (It's the widely-used [Kaggle "Medical Cost Personal Dataset"](https://www.kaggle.com/datasets/mirichoi0218/insurance).)

## 🛠️ Tech Stack

- **Python 3**
- `numpy`, `pandas` — data handling
- `matplotlib` — visualization
- `scikit-learn` — `LinearRegression`, `PolynomialFeatures`, `OneHotEncoder`

## ▶️ Running It

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook Polynomial_Regression.ipynb
```

Run Part 1 cells top to bottom for the toy-dataset demo. For Part 2, make
sure `insurance.csv` is in the working directory before running those
cells; the final cell will prompt for manual input (age, sex, bmi, etc.)
to demonstrate encoding a new record.

## 📝 Notes

- A missing `from sklearn.preprocessing import PolynomialFeatures` import
  was added to the notebook (it was used but not imported in the original).
- This notebook builds the feature pipeline (encoding) but does not yet
  train/evaluate a regression model on the insurance data — a natural next
  step would be fitting `LinearRegression` (or a polynomial variant) on
  `final_data` and reporting R² / MAE on a held-out test set.
