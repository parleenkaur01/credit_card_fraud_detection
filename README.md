# Credit Card Fraud Detection

🔗 **GitHub Repository:** https://github.com/parleenkaur01/credit_card_fraud_detection

This project builds and evaluates supervised machine learning models to detect fraudulent credit card transactions. It focuses on handling severe class imbalance and analyzing precision–recall trade-offs, which are critical in real-world fraud detection systems.

---

## 🚀 Overview

Credit card fraud leads to billions of dollars in losses annually. Detecting fraud is challenging because:

* Fraudulent transactions are extremely rare (class imbalance)
* False negatives (missed fraud) are costly
* Systems often require near real-time predictions

This project implements a complete machine learning pipeline to address these challenges and compare model performance across multiple algorithms.

---

## 📊 Dataset

* **Source:** Kaggle (Synthetic Credit Card Transactions Dataset)
* **Total Size:** ~1,000,000 transactions
* **Sample Used:** 200,000 (stratified)
* **Features:** 7 numerical + 1 binary target (`fraud`)
* **Missing Values:** None

### Key Features

* `distance_from_home`
* `distance_from_last_transaction`
* `ratio_to_median_purchase_price`
* `online_order`
* `used_chip`
* `used_pin_number`

---

## ⚙️ Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset Setup

⚠️ **Important:** The dataset is NOT included in this repository.

1. Download the dataset from Kaggle
   (search: *Credit Card Fraud Detection Synthetic Dataset*)

2. Place the file in the following location:

```
data/creditcard.csv
```

---

## ▶️ How to Run

1. Open the notebook:

```
Credit_Card_Fraud_Detection.ipynb
```

2. Run all cells from top to bottom

---

## 🧠 Methodology

### 1. Data Preprocessing

* Stratified train-test split (80/20)
* Class imbalance handled using `class_weight='balanced'`
* Feature scaling applied only for Logistic Regression

### 2. Models Implemented

* Logistic Regression (baseline model)
* Decision Tree
* Random Forest (ensemble model)

### 3. Evaluation Metrics

Due to class imbalance, accuracy is not used.

Instead, we evaluate using:

* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC

---

## 📈 Results

| Model               | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | --------- | ------ | -------- | ------- |
| Logistic Regression | ~0.57     | ~0.95  | ~0.72    | ~0.98   |
| Decision Tree       | 1.00      | 1.00   | 1.00     | 1.00    |
| Random Forest       | 1.00      | 0.999  | 1.00     | 1.00    |

> **Note:** Near-perfect scores are due to the synthetic nature of the dataset and strong feature separability.

---

## ⚖️ Threshold Tuning

Instead of using the default classification threshold (0.5), this project evaluates multiple thresholds:

* Lower threshold → higher recall (detect more fraud)
* Higher threshold → higher precision (reduce false alarms)

This reflects real-world decision-making where businesses must balance fraud detection and customer experience.

---

## 📊 Outputs & Visualizations

Running the notebook will generate:

* Class distribution plots
* Feature distribution visualizations
* Correlation heatmap
* Confusion matrices
* ROC and Precision-Recall curves
* Feature importance analysis (Random Forest)
* Threshold tuning plots
* Model comparison summary

---

## 🔍 Key Insights

* Random Forest achieved the strongest overall performance
* `ratio_to_median_purchase_price` is the most influential feature (~55% importance)
* Handling class imbalance is essential for effective fraud detection
* Threshold tuning enables flexible precision-recall trade-offs based on business needs

---

## 📉 Limitations

* Dataset is synthetic, leading to unrealistically high performance
* No temporal features (real-world fraud evolves over time)
* Models are not validated on real-world datasets

---

## 🔮 Future Work

* Evaluate models on real-world datasets (e.g., European card dataset)
* Apply advanced models (XGBoost, LightGBM)
* Use SMOTE or other resampling techniques
* Deploy as a real-time fraud detection API

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy — data processing
* Scikit-learn — machine learning models
* Matplotlib, Seaborn — visualization

---

## 📁 Project Structure

```
final_project/
├── README.md
├── requirements.txt
├── Credit_Card_Fraud_Detection.ipynb
└── data/           
```

---

## 🔗 Project Repository

GitHub: https://github.com/parleenkaur01/credit_card_fraud_detection

The repository demonstrates collaborative development through version control, including contributions from multiple team members, commit history, and iterative improvements.

---

## 👥 Team

* Parleen Bagga
* Adarsh Shresth
* Khush Patel
* Marco Novellini Blasco

---

## 📌 Conclusion

This project demonstrates that machine learning can effectively detect fraudulent transactions when:

* appropriate evaluation metrics are used,
* class imbalance is properly handled,
* and business trade-offs are incorporated through threshold tuning.

---

## ⭐ Acknowledgments

Dataset sourced from Kaggle.
Built using open-source Python libraries.
