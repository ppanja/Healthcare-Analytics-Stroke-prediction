setwd("C:/Users/Prasanta/XXX")
library(dplyr)
library(mice)
library(randomForest)
library(ineq)
library(data.table)
library(MLmetrics)
library(lightgbm)

train <- read.csv("train_ajEneEa.csv")
View(train)
str(train)
summary(train)
summary(train[train$Residence_type == 'Rural',])
summary(train[train$Residence_type == 'Urban',])

library(dplyr)
trainWoNULL <- NULL
trainWoNULL <- train %>% replace(. =="", NA)
summary(trainWoNULL)
write.csv(trainWoNULL,"trainWoNULL.csv")
traindf <- read.csv("trainWoNULL.csv")
summary(traindf)

library(mice)

miceMod <- mice(traindf[, !names(traindf) %in% "stroke"], method="rf")  # perform mice imputation, based on random forests.
miceOutput <- complete(miceMod)  # generate the completed data.
anyNA(miceOutput)
View(miceOutput)

#traindf1 <- traindf[c(2,13)]
#View(traindf1)
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

### Feature creation for test data: Bin the continuous variables ###

test$agecat<-cut(test$age, seq(0,85,5), right=FALSE, labels=c(1:17))
test$bmicat<-cut(test$bmi, seq(10,100,5), right=FALSE, labels=c(1:18))
test$avg_gluc_lvl_cat<-cut(test$avg_glucose_level, seq(55,295,10), right=FALSE, labels=c(1:24))

testdf <- test[,-c(1,4,10,11)]
View(testdf)


#### Preparing to rectify Imbalanced data and model using tuned Random Forest #########
library(DMwR)
library(caTools)
library(randomForest)
prop.table(table(traindfFinal$stroke))
traindfFinal$stroke <- as.factor(traindfFinal$stroke)
trainSMOTE <- SMOTE(stroke ~., data = traindfFinal, perc.over = 500)

prop.table(table(trainSMOTE$stroke))
table(traindfFinal$stroke)
table(trainSMOTE$stroke)

View(trainSMOTE)
tRF2 <- tuneRF(x = trainSMOTE[,-c(1,2)], 
               y=trainSMOTE$stroke,
               mtryStart = 3, 
               ntreeTry=20, 
               stepFactor = 1.5, 
               improve = 0.001, 
               trace=TRUE, 
               plot = TRUE,
               doBest = TRUE,
               nodesize = 3, 
               importance=TRUE
)

trainSMOTE$predict.class = predict(tRF2, trainSMOTE, type="class")
trainSMOTE$predict.score = predict(tRF2, trainSMOTE, type="prob")
head(trainSMOTE)



pred = prediction(trainSMOTE$predict.score[,2], trainSMOTE$stroke)
perf = performance(pred, "tpr", "fpr")
plot(perf)

KS = max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
auc = performance(pred,"auc"); 
auc = as.numeric(auc@y.values)

library(ineq)
gini = ineq(trainSMOTE$predict.score[,2], type="Gini")

with(trainSMOTE, table(stroke, predict.class))
auc
KS
gini

#######################


testdf$predict.class <- predict(tRF2, testdf[,-1], type="class")
testdf$predict.score <- predict(tRF2, testdf[,-1], type="prob")
View(testdf)

testdf$predict.score[,2]

testdf[testdf$predict.class == 1,]

final <- testdf[c(1,12)]

write.csv(final,"sample_submission.csv")