Formula 1 DNF Prediction using Random Forest (R Project)
📘 Overview

This project predicts whether a Formula 1 driver will DNF (Did Not Finish) a race using machine learning techniques implemented in R.
The goal is to analyze race and driver data, preprocess it, and build a predictive model that classifies if a driver will finish or not.

🎯 Objective

To build a classification model that predicts the DNF (Did Not Finish) outcome for Formula 1 drivers.

To identify which factors (driver, car, or race conditions) influence the DNF likelihood.

To evaluate model performance using metrics such as accuracy, confusion matrix, and ROC-AUC.

🧠 Project Workflow
1. Data Collection

Dataset: f1-dnf-classification.csv

Source: Kaggle - F1 DNF Classification Dataset

The dataset includes driver stats, race conditions, and car details.

2. Data Preprocessing

Loaded the dataset using:

f1 <- read.csv("f1-dnf-classification.csv")


Checked missing values and summary statistics:

summary(f1)
colSums(is.na(f1))


Converted target variable to a factor for classification:

f1$target_finish <- as.factor(f1$target_finish)


Split the dataset into training (80%) and testing (20%) sets using:

set.seed(123)
trainIndex <- createDataPartition(f1$target_finish, p = 0.8, list = FALSE)
train <- f1[trainIndex, ]
test  <- f1[-trainIndex, ]

🌲 3. Model Building

Algorithm used: Random Forest Classifier

Libraries:

library(randomForest)
library(caret)
library(pROC)


Model Training:

rf_model <- randomForest(target_finish ~ ., 
                         data = train, 
                         ntree = 300, 
                         mtry = 5, 
                         importance = TRUE)
print(rf_model)

📊 4. Model Evaluation

Predictions on test data:

predictions <- predict(rf_model, newdata = test)


Confusion Matrix:

confusionMatrix(predictions, test$target_finish)


ROC-AUC Score:

roc_obj <- roc(as.numeric(test$target_finish), as.numeric(predictions))
plot(roc_obj)


Accuracy Achieved: ~98.6%

🔍 5. Key Insights

Random Forest performed with high accuracy.

Key influential features (example):

Car reliability

Driver performance

Team consistency

Race conditions

Feature importance from Random Forest helps understand why certain drivers are more prone to DNFs.

🧩 6. Tech Stack
Category	Tools / Libraries
Programming Language	R
Machine Learning	randomForest, caret
Data Handling	tidyverse, data.table
Visualization	ggplot2, plotly
Evaluation	pROC (ROC-AUC)
Dataset Source	Kaggle
📁 7. Folder Structure
F1-DNF-Prediction/
│
├── f1-dnf-classification.csv        # Dataset
├── f1_dnf_project.R                 # Main R script
├── README.md                        # Project Documentation
├── requirements.txt / install.R     # List of packages to install
├── results/                         # Evaluation results and plots
└── presentation/                    # PPT slides for presentation

🧰 8. How to Run the Project

Clone the repository

git clone https://github.com/yourusername/F1-DNF-Prediction.git
cd F1-DNF-Prediction


Install the required R packages

install.packages(c("tidyverse", "caret", "randomForest", "pROC"))


Run the R script

source("f1_dnf_project.R")


View Results

Model accuracy

Confusion matrix

ROC curve

Feature importance chart

🏁 9. Results
Metric	Value
Accuracy	98.6%
Precision	High
Recall	High
ROC-AUC	Excellent
🚀 10. Future Enhancements

Try XGBoost or LightGBM for improved accuracy.

Add real-time race data (weather, pit stop times).

Deploy as a web app using Shiny or Flask (via reticulate).

👨‍💻 Author

Velmurugan
Master of Computer Applications (MCA)
Email: [uvelmurugan218@gmail.com]
GitHub: [https://github.com/Velmurugan-02]









