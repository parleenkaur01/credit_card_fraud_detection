# Credit Card Fraud Detection

This project builds and evaluates supervised machine learning models to identify fraudulent credit card transactions. It focuses on handling severe class imbalance and analyzing trade-offs between precision and recall in real-world fraud detection systems.

---

## 🚀 Overview

Credit card fraud results in billions of dollars in losses annually. Detecting fraudulent transactions is challenging due to:

- Extreme class imbalance (fraud cases are rare)
- High cost of false negatives (missed fraud)
- Need for real-time decision making

This project implements a complete ML pipeline to address these challenges.

---

## 📊 Dataset

- Source: Kaggle (synthetic credit card transactions dataset)
- Size: ~1,000,000 transactions
- Sample Used: 200,000 (stratified)
- Features: 7 numerical + 1 binary target (`fraud`)
- No missing values

Key features include:
- `distance_from_home`
- `distance_from_last_transaction`
- `ratio_to_median_purchase_price`
- `online_order`, `used_chip`, `used_pin_number`

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Stratified train-test split (80/20)
- Class imbalance handled using `class_weight='balanced'`
- Feature scaling applied only for Logistic Regression

### 2. Models Used
- Logistic Regression (baseline)
- Decision Tree
- Random Forest (ensemble)

### 3. Evaluation Metrics
Due to class imbalance, accuracy is not used.

We evaluate using:
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

---

## 📈 Results

| Model               | Precision | Recall | F1 Score | ROC-AUC |
|---------------------|----------|--------|----------|--------|
| Logistic Regression | ~0.57    | ~0.95  | ~0.72    | ~0.98  |
| Decision Tree       | 1.00     | 1.00   | 1.00     | 1.00   |
| Random Forest       | 1.00     | 0.999  | 1.00     | 1.00   |

> Note: Near-perfect scores are due to the synthetic nature of the dataset and strong feature separability.

---

## 🔍 Key Insights

- **Random Forest** achieved the strongest performance across all metrics.
- **`ratio_to_median_purchase_price`** is the most important feature (~55% importance).
- **Class imbalance handling** is critical — without it, models fail to detect fraud.
- **Threshold tuning** allows control over precision-recall tradeoffs depending on business needs.

---

## ⚖️ Threshold Tuning

Instead of using the default 0.5 threshold, we evaluate multiple thresholds:

- Lower threshold → higher recall (catch more fraud)
- Higher threshold → higher precision (fewer false alarms)

This reflects real-world decision-making in fraud detection systems.

---

## 📉 Limitations

- Dataset is synthetic → unrealistically high performance
- No temporal features (real fraud evolves over time)
- Models not tested on real-world datasets

---

## 🔮 Future Work

- Test on real-world datasets (e.g., Kaggle European dataset)
- Apply advanced models (XGBoost, LightGBM)
- Use SMOTE for oversampling
- Deploy as a real-time API for fraud detection

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## 👥 Team

- Parleen Bagga  
- Adarsh Shresth  
- Khush Patel  
- Marco Novellini Blasco  

---

## 📌 Conclusion

This project demonstrates how machine learning can effectively detect fraudulent transactions when:
- appropriate metrics are used,
- class imbalance is handled correctly,
- and business tradeoffs are considered.

---

## ⭐ Acknowledgments

Dataset sourced from Kaggle.
