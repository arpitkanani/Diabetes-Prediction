# 🩺 Diabetes Prediction System

## 📌 Project Overview
The **Diabetes Prediction System** is a machine learning–based application designed to predict whether a person is likely to have diabetes based on medical and demographic attributes.  
The system provides a binary output indicating whether the individual is **Diabetic** or **Non-Diabetic**.

This project demonstrates an end-to-end workflow including data preprocessing, model training, evaluation, and deployment using Flask.

---

## 🎯 Objective
- Predict the likelihood of diabetes in an individual  
- Support early detection and preventive healthcare  
- Apply supervised machine learning techniques on healthcare data  

---

## 🧠 Problem Type
- **Machine Learning Type:** Supervised Learning  
- **Task:** Binary Classification  
- **Target Variable:** Outcome  

---

## 📊 Dataset Description
The dataset consists of medical predictor variables and one target variable.  
Each row represents a single patient record.

---

## 🔹 Column Description

| Column Name | Description |
|------------|-------------|
| **Pregnancies** | Number of times the patient has been pregnant |
| **Glucose** | Plasma glucose concentration (mg/dL) |
| **BloodPressure** | Diastolic blood pressure (mm Hg) |
| **Insulin** | 2-hour serum insulin level (mu U/ml) |
| **BMI** | Body Mass Index (weight in kg / height in m²) |
| **DiabetesPedigreeFunction** | A score representing diabetes risk based on family history |
| **Age** | Age of the patient in years |
| **Outcome** | Target variable (1 = Diabetic, 0 = Non-Diabetic) |

> **Note:** Some features may contain zero values indicating missing measurements. These are handled during preprocessing.

---

## 🔄 Project Workflow
1. Data Ingestion  
2. Data Preprocessing  
3. Feature Engineering  
4. Model Training  
5. Model Evaluation  
6. Best Model Selection  
7. Deployment using Flask  

---

## 🧪 Models Implemented
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree Classifier  
- Random Forest Classifier  
- Support Vector Classifier (SVC)  
- Gradient Boosting Classifier  
- AdaBoost Classifier  
- XGBoost Classifier  
- CatBoost Classifier  
- Gaussian Naive Bayes  

---

## 📈 Evaluation Metric
- **Accuracy Score**  
The model with the highest accuracy on test data is selected for deployment.

---

## 🌐 Web Application
- Developed using **Flask**
- Users enter medical parameters via a web form
- The system predicts:
  - **Diabetic**
  - **Non-Diabetic**

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, CatBoost  
- **Web Framework:** Flask  
- **Frontend:** HTML, CSS, JavaScript  
- **Version Control:** Git  

---

## Conclusion
This project showcases how machine learning can be applied to healthcare data for predictive analysis.  
The Diabetes Prediction System provides a practical approach for early diabetes risk assessment and demonstrates a complete ML pipeline from data processing to deployment.
