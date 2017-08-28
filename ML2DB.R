install.packages("caret")
library(caret)
require(datasets)
str(iris)
iris

#create a list of 80% of rows in the original dataset to use them for training
index = createDataPartition(iris$Species, p = 0.80, list = FALSE)
# select 80% of the data for validation(dataset)
iris.Train = iris[index, ]
# select 20% of the data for validation
iris.Test = iris[-index, ]


dim(iris.Train)

sapply(iris.Train, class)

head(iris.Train)

levels(iris.Train$Species)

percentage <- prop.table(table(iris.Train$Species)) * 100
cbind(freq=table(iris.Train$Species), percentage=percentage)

summary(iris.Train)

#Univariate plots to understand each attribute.
x <- iris.Train[,1:4]
y <- iris.Train[,5]

par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i], main=names(iris)[i])
}

plot(y)

#Multivariate plots to understand the relationships between attributes.
install.packages("ellipse")
library(ellipse)
featurePlot(x=x, y=y, plot="ellipse")

featurePlot(x=x, y=y, plot="box")

scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

#Test Harness
install.packages("rpart")
install.packages("kernlab")
install.packages("e1071")
install.packages("randomForest")

library(rpart)
library(kernlab)
library(e1071)
library(randomForest)

control <- trainControl(method="cv", number=10)
metric <- "Accuracy"


# a) linear algorithms
set.seed(7)
fit.lda <- train(Species~., data=iris, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(Species~., data=iris, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(Species~., data=iris, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(Species~., data=iris, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(Species~., data=iris, method="rf", metric=metric, trControl=control)

#Select the Best Model
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

dotplot(results)

#LDA model
print(fit.lda)

#Make Predictions
predictions <- predict(fit.lda, iris.Test)
confusionMatrix(predictions, iris.Test$Species)


