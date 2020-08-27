#Setting a working directory
# 1. Prepare Problem
# a) Load libraries
library(caret)
library(ggplot2)
library(lattice)
library(corrplot)
library(ggpubr)
library(glmnet)
library(klaR)
library(C50)
# b) Load dataset
filename="churn.csv"
Churn<-read.csv(filename, header = TRUE,sep = ",")

# c)Split-out validation dataset at this step we will split out daset into a training and validation daset
#we will create a list of 80% of the rows in the original dataset we can use for training
set.seed(7)
validationIndex <- createDataPartition(Churn$Exited, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- Churn[-validationIndex,]
# use the remaining 80% of data to training and testing the models
dataset <- Churn[validationIndex,]

# Summarize Data and Descriptive statistics
dim(dataset)
#let s have a look on the data we are working on
head(dataset)
tail(dataset)
sapply(dataset, Exited)
dataset<-dataset[,4:14]
#Transform data and creating dummy values for gender and geograpy attributes
dataset$Exited<-as.factor(dataset$Exited)
dataset$Geography<-as.integer(dataset$Geography)
dataset$Gender<-as.integer(dataset$Gender)
str(dataset)
# convert input values to numeric
for(i in 1:10) {
  dataset[,i] <- as.numeric(as.factor(dataset[,i]))
}
str(dataset)
# Descriptive statistics
summary(dataset[,-c(2,3,8,9,11)])

# b) Data visualizations
#Unimodal Data Visualizations
# let s create a histogram for each attribute
par(mfrow=c(2,5))
for(i in 1:10) {
  hist(dataset[,i], main=names(dataset)[i])
}
# we can also take a look on density plot for each attribute
par(mfrow=c(2,5))
complete_cases <- complete.cases(dataset)
for(i in 1:10) {
  plot(density(dataset[complete_cases,i]), main=names(dataset)[i])
}
#Multimodel visualization
#let s have a look on the correlation between our predictor
cor(dataset[,1:10])
par(mfrow=c(3,3))
# bar plots of each variable by Exited
for(i in 1:10) {
  barplot(table(dataset$Exited,dataset[,i]), main=names(dataset)[i],
          legend.text=unique(dataset$Exited))
}

# Evaluate Algorithms
# a) Test options and evaluation metric
# 10-fold cross validation with 3 repeats
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "Accuracy"
# b) Spot Check Algorithms
# LG (Logistic regression)
set.seed(7)
fit.glm <- train(Exited~., data=dataset, method="glm", metric=metric, trControl=control, na.action=na.omit)
# LDA (Linear Discriminant Analysis)
set.seed(7)
fit.lda <- train(Exited~., data=dataset, method="lda", metric=metric, trControl=control, na.action=na.omit)
# GLMNET (Regularized Regression) 
set.seed(7)
fit.glmnet <- train(Exited~., data=dataset, method="glmnet", metric=metric, trControl=control, na.action=na.omit)
# KNN (k-Nearest Neighbors)
set.seed(7)
fit.knn <- train(Exited~., data=dataset, method="knn", metric=metric, trControl=control, na.action=na.omit)
# CART(Classification and Regression Trees)
set.seed(7)
fit.cart <- train(Exited~., data=dataset, method="rpart", metric=metric, trControl=control, na.action=na.omit)
# Naive Bayes
set.seed(7)
fit.nb <- train(Exited~., data=dataset, method="nb", metric=metric, trControl=control, na.action=na.omit)
# SVM(Support Vector Machine)
set.seed(7)
fit.svm <- train(Exited~., data=dataset, method="svmRadial", metric=metric, trControl=control, na.action=na.omit)
# Compare algorithms
results <- resamples(list(LG=fit.glm, LDA=fit.lda, GLMNET=fit.glmnet, KNN=fit.knn, CART=fit.cart, NB=fit.nb, SVM=fit.svm))
summary(results)
dotplot(results)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

# b) Ensembles
# Ensembles: Boosting and Bagging
# 10-fold cross validation with 3 repeats
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "Accuracy"
# Bagged CART
set.seed(7)
fit.treebag <- train(Exited~., data=dataset, method="treebag", metric=metric, trControl=control, na.action=na.omit)
# Random Forest
set.seed(7)
fit.rf <- train(Exited~., data=dataset, method="rf", metric=metric, trControl=control, na.action=na.omit)
# Stochastic Gradient Boosting
set.seed(7)
fit.gbm <- train(Exited~., data=dataset, method="gbm", metric=metric, trControl=control, verbose=FALSE, na.action=na.omit)
# C5.0
set.seed(7)
fit.c50 <- train(Exited~., data=dataset, method="C5.0", metric=metric,  trControl=control, na.action=na.omit)
# Compare results
ensemble_results <- resamples(list(BAG=fit.treebag, RF=fit.rf, GBM=fit.gbm, C50=fit.c50))
summary(ensemble_results)
dotplot(ensemble_results)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(ensemble_results, scales=scales)

# Improve Accuracy
# a) Algorithm Tuning
# Tune SVM
# 10-fold cross validation with 3 repeats
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "Accuracy"
set.seed(7)
grid <- expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1, 10, by=1))
fit.svm <- train(Exited~., data=dataset, method="svmRadial", metric=metric, tuneGrid=grid, preProc=c("BoxCox"), trControl=control, na.action=na.omit)
print(fit.svm)
plot(fit.svm)

# 6. Finalize Model

# a) Predictions on validation dataset
# prepare the validation dataset
set.seed(7)
# transform the validation dataset and remove id and surname columns as these may not help in the  computation process
validation <- validation[,-c(1,2,3)]
# remove missing values and crating dummy values for geography and Gender attributes
validation <- validation[complete.cases(validation),]
validation$Exited<-as.factor(validation$Exited)
validation$Geography<-as.integer(validation$Geography)
validation$Gender<-as.integer(validation$Gender)
for(i in 1:10) {
  dataset[,i] <- as.numeric(as.factor(validation[,i]))
}
str(validation)
head(validation, n=20)

# b) Create standalone model on entire training dataset
fit.svm2 <- train(Exited~., data=dataset, method="svmRadial",preProc=c("center"), metric=metric,.sigma=0.025,.C=7, trControl=control, na.action=na.omit)

# estimate variable importance
importance <- varImp(fit.svm2, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

#Make a prediction on the validation dataset
final_predictions <- predict(fit.svm2, validation[,1:10])
confusionMatrix(final_predictions2, validation$Exited)

