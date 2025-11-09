# =============================================
# Project: F1 DNF (Did Not Finish) Prediction
# Author: Velmurugan
# Date: October 2025
# =============================================

# ---- Step 1: Install & Load required libraries ----
install.packages("rpart")
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)
install.packages("ggplot2")
library(ggplot2)
install.packages("caret")
install.packages("randomForest")
library(randomForest)
library(caret)

# ---- Setting the path for accessing the path files in locally ---- 
getwd()
setwd("C:/Users/91824/Documents/Velmurugan_717824Z154/R_project")

# ---- Step 2: Read your dataset ----
f1 <- read.csv("f1_dnf.csv")

# ---- Step 3: Clean the data (optional) ----
# Remove missing values if any
f1 <- na.omit(f1)

# ---- Step 4: Convert target column to factor ----
f1$DNF <- as.factor(f1$target_finish)

# ---- Step 5: Split the dataset into training and testing ----
set.seed(123)
trainIndex <- createDataPartition(f1$DNF, p = 0.8, list = FALSE)
train <- f1[trainIndex, ]
test <- f1[-trainIndex, ]

# ---- Step 6: Build a Random Forest model ----
rf_model <- randomForest(DNF ~ ., data = train, ntree = 100, importance = TRUE)
print(rf_model)

# ---- Step 7: Make predictions ----
predictions <- predict(rf_model, test)

# ---- Step 8: Evaluate model performance ----
conf_matrix <- confusionMatrix(predictions, test$DNF)
print(conf_matrix)

# ---- Step 9: Visualize feature importance ----
varImpPlot(rf_model)

# ---- Step 10: Decision tree visualization (optional) ----
tree_model <- rpart(DNF ~ ., data = train, method = "class")
rpart.plot(tree_model, main = "Decision Tree for F1 DNF Prediction")

# ---- Step 11: Accuracy graph (optional) ----
accuracy <- conf_matrix$overall["Accuracy"]
barplot(accuracy,
        main = "Model Accuracy",
        col = "steelblue",
        ylim = c(0, 1))
text(x = 1, y = accuracy / 2, labels = round(accuracy, 4), col = "white", cex = 1.5)

# =============================================
# End of Script
# =============================================
