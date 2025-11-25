Email Spam Classification using SVD + Random Forest

This project detects whether an email is Spam or Ham using a Machine Learning pipeline combining Truncated SVD for dimensionality reduction and a Random Forest Classifier for prediction.

🔍 Project Summary

Reads input data from email.csv

Drops unnecessary columns (Email No., Prediction)

Samples 3000 records to improve speed

Splits data into 80% train and 20% test

Uses SVD (100 components) to reduce high-dimensional features

Trains a Random Forest model with 100 trees

Evaluates the model using:

Accuracy

Precision, Recall, F1-score

Confusion Matrix heatmap

📦 Requirements
pip install pandas scikit-learn matplotlib seaborn

▶️ How to Run

Place email.csv in the same folder and run:

python email_spam_classifier.py

📊 Model Performance

The script displays:

Accuracy score → Overall correctness

Classification Report → Detailed metrics

Confusion Matrix → Visual error analysis

🧠 Algorithms Used

Truncated SVD: Reduces dimensionality and noise in text features

Random Forest: Robust ensemble classifier for high accuracy

📁 Dataset Information

Expected columns:

Numerical features extracted from emails

Prediction → 0 = Ham, 1 = Spam

🛠 Technologies

Python

Pandas

Scikit-Learn

Matplotlib

Seaborn

📌 Project Use Cases

Email spam detection

Text classification

Feature reduction + ensemble learning demo# Machine-learning
