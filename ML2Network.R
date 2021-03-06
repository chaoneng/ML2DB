install.packages("neuralnet")
install.packages("nnet")
install.packages("caret")

library(neuralnet) # for neuralnet(), nn model
library(nnet)      # for class.ind()
library(caret)     # for train(), tune parameters
library(datasets)
str(iris)
iris

data <- iris

# 因為Species是類別型態，這邊轉換成三個output nodes，使用的是class.ind函式()
head(class.ind(data$Species))

# 並和原始的資料合併在一起，cbind意即column-bind
data <- cbind(data, class.ind(data$Species))

# 原始資料就會變成像這樣
head(data)

formula.bpn <- setosa + versicolor + virginica ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
formula.bpn

bpn <- neuralnet(formula = formula.bpn, 
                 data = data,
                 hidden = c(2),       # 一個隱藏層：2個node
                 learningrate = 0.01, # learning rate
                 threshold = 0.01,    # partial derivatives of the error function, a stopping criteria
                 stepmax = 5e5        # 最大的ieration數 = 500000(5*10^5)
                 
)
plot(bpn)


#Tuning Parameters

# nrow()是用來擷取資料筆數，乘上0.8後，表示我們的train set裡面要有多少筆資料(data size)
smp.size <- floor(0.8*nrow(data)) 
# 因為是抽樣，有可能每次抽樣結果都不一樣，因此這裡規定好亂數表，讓每次抽樣的結果一樣
set.seed(100)                     
# 從原始資料裡面，抽出train set所需要的資料筆數(data size)
train.ind <- sample(seq_len(nrow(data)), smp.size)
# 分成train/test
train <- data[train.ind, ]
test <- data[-train.ind, ]

# tune parameters
model <- train(form=formula.bpn,     # formula
               data=train,           # 資料
               method="neuralnet",   # 類神經網路(bpn)
               
               # 最重要的步驟：觀察不同排列組合(第一層1~4個nodes ; 第二層0~4個nodes)
               # 看何種排列組合(多少隱藏層、每層多少個node)，會有最小的RMSE
               tuneGrid = expand.grid(.layer1=c(1:4), .layer2=c(0:4), .layer3=c(0)),               
               
               # 以下的參數設定，和上面的neuralnet內一樣
               learningrate = 0.01,  # learning rate
               threshold = 0.01,     # partial derivatives of the error function, a stopping criteria
               stepmax = 5e5         # 最大的ieration數 = 500000(5*10^5)
)

# 計算出最佳參數：The final values used for the model were layer1 = 1, layer2 = 2 and layer3 = 0.
model

plot(model)


bpn <- neuralnet(formula = formula.bpn, 
                 data = train,
                 hidden = c(1,2),     # The final values used for the model were layer1 = 1, layer2 = 2 and layer3 = 0.
                 learningrate = 0.01, # learning rate
                 threshold = 0.01,    # partial derivatives of the error function, a stopping criteria
                 stepmax = 5e5        # 最大的ieration數 = 500000(5*10^5)
                 
)

# 顯示經過參數評估之類神經網路
plot(bpn)

#Make Predictions
# 取前四個欄位，進行預測
pred <- compute(bpn, test[, 1:4])  

# 預測結果
pred$net.result


# 四捨五入後，變成0/1的狀態
pred.result <- round(pred$net.result)
pred.result

pred.result <- as.data.frame(pred.result)

# 建立一個新欄位，叫做Species
pred.result$Species <- ""

# 把預測結果轉回Species的型態
for(i in 1:nrow(pred.result)){
  if(pred.result[i, 1]==1){ pred.result[i, "Species"] <- "setosa"}
  if(pred.result[i, 2]==1){ pred.result[i, "Species"] <- "versicolor"}
  if(pred.result[i, 3]==1){ pred.result[i, "Species"] <- "virginica"}
}

pred.result

# 混淆矩陣 (預測率有96.67%)
table(real    = test$Species, 
      predict = pred.result$Species)
