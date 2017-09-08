#install xgboost package
install.packages("xgboost")

#call library
library(xgboost)
library(datasets)

#input test data (Iris data)
str(iris)
iris2 = iris

# manually add a categorical variable as one of the features
# to reflect real-world situation.
iris2$id = factor(sample(c("A","B"),150,replace=T))

# Use sparse matrix in Matrix package
library(Matrix)

# convert all feature variables to a sparse matrix
xdata = sparse.model.matrix(Species ~ .-1, data = iris2)
xdata

# number of categories in response variable
m = nlevels(iris2$Species)
m

# recode Y as 0,1,2,...,m-1
Y = as.integer(iris2$Species) - 1
Y

# set random seed
set.seed(10000)

# xgboost parameters setup
param = list("objective" = "multi:softprob",
             "eval_metric" = "mlogloss",
             "num_class" = m
)

# build the model
result = xgboost(param=param, data=xdata, label=Y, nrounds=20)

# get prediction
Ypred = predict(result,xdata)
Ypred = t(matrix(Ypred,m,length(Ypred)/m))
# colnames(Ypred) = levels(iris2$Species)
Ypred = levels(iris2$Species)[max.col(Ypred)]

# confusion matrix
t0 = table(iris2$Species,Ypred)
t0

# accuracy
sum(diag(t0))/sum(t0)

# variable importance
imp = xgb.importance(names(iris2[,-5]),model=result)
print(imp)

#data visualization
library(Ckmeans.1d.dp)
xgb.plot.importance(imp)

library(DiagrammeR)
xgb.plot.tree(feature_names=names(iris[,-5]),model=result, n_first_tree=2)
