in branches another project is added

📄 NLP-Based Resume Classification System (26 Categories | 99% Accuracy)
📌 Project Overview

This project is an advanced NLP-based resume classification system that automatically categorizes resumes into 26 predefined job categories with ~99% classification accuracy. The system is designed to assist HR teams, recruiters, and ATS platforms by enabling fast, accurate, and scalable resume screening.

By leveraging state-of-the-art text preprocessing, feature extraction, and supervised machine learning models, the system efficiently classifies resumes and delivers highly reliable results.

🎯 Objective

Automate resume screening

Reduce manual effort in HR processes

Improve hiring efficiency

Accurately classify resumes into 26 professional domains

🧠 NLP & Machine Learning Pipeline
🔹 Text Preprocessing

Resume text extraction (PDF/DOC)

Lowercasing & normalization

Removal of punctuation & special characters

Stopword removal

Lemmatization / stemming

Tokenization

🔹 Feature Engineering

TF-IDF Vectorization

N-grams (unigrams & bigrams)

Dimensionality optimization

🔹 Model Training

Supervised classification models tested and compared:

Logistic Regression

Support Vector Machine (SVM)

Naive Bayes

Random Forest

Gradient Boosting

Best model selected based on performance

🔹 Performance

Overall Accuracy: ~99%

High precision & recall across all 26 categories

Minimal misclassification even on overlapping job roles

🗂️ Resume Categories (26)

Examples include:

Data Science

Machine Learning

Software Engineering

Web Development

DevOps

Cloud Computing

Cyber Security

Database Administration

Mobile App Development

Business Analyst

HR

Finance

Marketing
(and more)

🏗️ System Architecture
Resume Upload
     ↓
Text Extraction & Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Trained NLP Classification Model
     ↓
Predicted Resume Category

⚙️ Key Technical Highlights

✔️ High-accuracy NLP classification
✔️ Scalable for large resume volumes
✔️ Robust preprocessing pipeline
✔️ Multi-class classification (26 categories)
✔️ Production-ready architecture
✔️ ATS-friendly design

🧰 Tech Stack
Layer	Technology
Language	Python
NLP	NLTK / spaCy
ML	Scikit-learn
Vectorization	TF-IDF
Evaluation	Accuracy, Precision, Recall, F1-Score
