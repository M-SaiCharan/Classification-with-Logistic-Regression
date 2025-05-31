# 🧠 Logistic Regression Classifier - Breast Cancer Detection

> A simple binary classification model using Logistic Regression to detect breast cancer from diagnostic data.

---

## 📊 Dataset Overview

- **File**: `data.csv`
- **Source**: Breast Cancer Wisconsin (Diagnostic) Data Set
- **Target**: `diagnosis`
  - `M` → Malignant (`1`)
  - `B` → Benign (`0`)

---

## 🔧 Tools & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## 🚀 Project Workflow

- 📥 Load and clean dataset
- 🧹 Drop `id`, `Unnamed: 32`
- 🔁 Encode labels (`M` = 1, `B` = 0)
- 📊 Train-test split (80/20)
- ⚖️ Feature standardization
- 🧠 Train Logistic Regression model
- ✅ Evaluate with:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1)
  - ROC-AUC Score and Curve
- 🎯 Threshold tuning (custom threshold = 0.4)
- 📉 Sigmoid function visualization

---

## 💻 How to Run

1. Clone this repo and place `data.csv` in the root directory.
2. Install dependencies:

```bash
pip install pandas scikit-learn matplotlib numpy
```

3. Run the Python script:

```bash
python main.py
```

---

## 📈 Sample Output

<img width="566" alt="Screenshot 2025-05-31 at 21 45 23" src="https://github.com/user-attachments/assets/0135db4c-3d7a-4fc3-84ff-1e51d9b01876" />
<img width="565" alt="Screenshot 2025-05-31 at 21 45 50" src="https://github.com/user-attachments/assets/419cb60b-c326-45f4-8011-8224b2e5d0cf" />


---

## 📘 Sigmoid Function

Used to map any value to a probability between 0 and 1.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```


