setwd("C:/Users/Prasanta/XXX")
library(dplyr)
library(mice)
library(randomForest)
library(ineq)
library(data.table)

### Read Train and impute missing values ###

train <- read.csv("train_ajEneEa.csv")
View(train)
str(train)
summary(train)
summary(train[train$Residence_type == 'Rural',])
summary(train[train$Residence_type == 'Urban',])


trainWoNULL <- NULL
trainWoNULL <- train %>% replace(. =="", NA)
summary(trainWoNULL)
write.csv(trainWoNULL,"trainWoNULL.csv")
traindf <- read.csv("trainWoNULL.csv")
summary(traindf)


miceMod <- mice(traindf[, !names(traindf) %in% "stroke"], method="rf")  # perform mice imputation, based on random forests.
miceOutput <- complete(miceMod)  # generate the completed data.
anyNA(miceOutput)
View(miceOutput)


miceOutput1 <- miceOutput[c(2,11,12)]

traindfnew <- traindf[,-c(11,12)]
View(traindfnew)

traindfFinal <- merge(traindfnew,miceOutput1,by="id")
View(traindfFinal)

### Feature creation: Bin the continuous variables ###
traindfFinal$agecat<-cut(traindfFinal$age, seq(0,85,5), right=FALSE, labels=c(1:17))
traindfFinal$bmicat<-cut(traindfFinal$bmi, seq(10,100,5), right=FALSE, labels=c(1:18))
traindfFinal$avg_gluc_lvl_cat<-cut(traindfFinal$avg_glucose_level, seq(55,295,10), right=FALSE, labels=c(1:24))

traindfFinal <- traindfFinal[,-c(2,4,10,12)]
traindfFinal <- traindfFinal[c(1,8,2,3,4,5,6,7,9,10,11,12)]
View(traindfFinal)

### Read Test and impute missing values ###

test <- read.csv("test_V2akXPA.csv")
testWoNULL <- test %>% replace(. =="", NA)
write.csv(testWoNULL,"testWoNULL.csv")
testdf <- read.csv("testWoNULL.csv")

miceMod1 <- mice(testdf, method="rf")  # perform mice imputation, based on random forests.
miceOutput1 <- complete(miceMod1)  # generate the completed data.
anyNA(miceOutput)
View(miceOutput1)
write.csv(miceOutput1,"testWoNA.csv")
testWoNA <- read.csv("testWoNA.csv")

test <- testWoNA

View(test)
summary(test)

### Feature creation: Bin the continuous variables ###
test$agecat<-cut(test$age, seq(0,85,5), right=FALSE, labels=c(1:17))
test$bmicat<-cut(test$bmi, seq(10,100,5), right=FALSE, labels=c(1:18))
test$avg_gluc_lvl_cat<-cut(test$avg_glucose_level, seq(55,295,10), right=FALSE, labels=c(1:24))

testdf <- test[,-c(1,4,10,11)]
View(testdf)

testdf2 <- testdf[,-c(12,13)]
testdf2$stroke <- NA
testdf2 <- testdf2[,c(1,12,2,3,4,5,6,7,8,9,10,11)]
View(testdf2)

traindfFinal$smoking_status <- as.numeric(traindfFinal$smoking_status)
traindfFinal$gender <- as.numeric(traindfFinal$gender)
traindfFinal$ever_married <- as.numeric(traindfFinal$ever_married)
traindfFinal$work_type <- as.numeric(traindfFinal$work_type)
traindfFinal$Residence_type <- as.numeric(traindfFinal$Residence_type)
traindfFinal$agecat <- as.numeric(traindfFinal$agecat)
traindfFinal$bmicat <- as.numeric(traindfFinal$bmicat)
traindfFinal$avg_gluc_lvl_cat <- as.numeric(traindfFinal$avg_gluc_lvl_cat)

traindfFinal1 <- as.data.table(traindfFinal)


testdf2$smoking_status <- as.numeric(testdf2$smoking_status)
testdf2$gender <- as.numeric(testdf2$gender)
testdf2$ever_married <- as.numeric(testdf2$ever_married)
testdf2$work_type <- as.numeric(testdf2$work_type)
testdf2$Residence_type <- as.numeric(testdf2$Residence_type)
testdf2$agecat <- as.numeric(testdf2$agecat)
testdf2$bmicat <- as.numeric(testdf2$bmicat)
testdf2$avg_gluc_lvl_cat <- as.numeric(testdf2$avg_gluc_lvl_cat)

testdf21 <- as.data.table(testdf2)

data1 <- rbind(traindfFinal1,testdf21)
View(data1)

varnames1 = setdiff(colnames(data1), c("id","stroke"))

#options(warn = 0) 
#debug(as.matrix)
train_sparse1 = Matrix(as.matrix(data1[!is.na(data1$stroke), varnames1, with=FALSE]), sparse=TRUE)
#traceback()
#train_sparse = Matrix(as.matrix(data2[!is.na(data2$target), varnames, with=FALSE]), sparse=TRUE)
test_sparse1  = Matrix(as.matrix(data1[is.na(data1$stroke) , varnames1, with=FALSE]), sparse=TRUE)
#View(test_sparse1)
#attach(data1)

y_train  = data1[!is.na(stroke),stroke]
test_ids = data1[is.na(data1$stroke),id]

lgb.train = lgb.Dataset(data=train_sparse1, label=y_train)

categoricals.vec = colnames(data1)[c(3,4,5,6,7,8,9,10,11,12)]

##Setting up LGBM Parameters

lgb.grid = list(objective = "binary",
                metric = "auc",
                min_sum_hessian_in_leaf = 1,
                feature_fraction = 0.7,
                bagging_fraction = 0.7,
                bagging_freq = 1,
                #                min_data = 100,
                #                max_bin = 50,
                #                lambda_l1 = 8,
                #                lambda_l2 = 1.3,
                min_data_in_bin=50,
                #                min_gain_to_split = 10,
                #                min_data_in_leaf = 30,
                is_unbalance = TRUE)

##Setting up Gini Eval Function
# Gini for Lgb
lgb.normalizedgini = function(preds, dtrain){
  actual = getinfo(dtrain, "label")
  score  = NormalizedGini(preds,actual)
  return(list(name = "gini", value = score, higher_better = TRUE))
}

##Cross Validation
library(MLmetrics)
lgb.model.cv = lgb.cv(params = lgb.grid, data = lgb.train, learning_rate = 0.001, num_leaves = 3,num_threads = 2 , 
                      nrounds = 100, early_stopping_rounds = 10,eval_freq = 2, eval = lgb.normalizedgini,
                      categorical_feature = categoricals.vec, nfold = 5, stratified = TRUE)

best.iter = lgb.model.cv$best_iter
#best.iter = 1

# Train final model
lgb.model = lgb.train(params = lgb.grid, data = lgb.train, learning_rate = 0.001,
                      num_leaves = 3, num_threads = 2 , nrounds = 3,
                      eval_freq = 2, eval = lgb.normalizedgini,
                      categorical_feature = categoricals.vec)

# Create and Submit Predictions
library(data.table)
preds = data.table(id=test_ids, stroke=predict(lgb.model,test_sparse1))
colnames(preds)[1] = "id"
fwrite(preds, "submission.csv")


