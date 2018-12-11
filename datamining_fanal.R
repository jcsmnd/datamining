# 12/06/2018 Final report. KNN, SVM - MYUNGSIK KIM
library(class)
library(ggplot2)
library(caret)
library(e1071)

######## KNN ##########
#delete all na value(not execute in this case)
ds <- na.omit(ds)

#include only certain columns for knn algorithm
ds <- crimes_preprocessed[ ,c(7,8,10,21,22,26,28)] 

#convert null value to UNKNOWN string
ds[is.na(ds)] <- "UNKNOWN"

#check null value
table(is.na(ds))

#convert char to factor
ds$CrimeGroup <- as.factor(ds$CrimeGroup)
ds$Inside.Outside <- as.factor(ds$Inside.Outside)
ds$District <- as.factor(ds$District)
ds$Weapon <- as.factor(ds$Weapon)
ds$Hour_Factor <- as.factor(ds$Hour_Factor)
ds$Weekday <- as.factor(ds$Weekday)
ds$Month_names <- as.factor(ds$Month_names)

#convert factor to integer
ds$Inside.Outside <- as.integer(ds$Inside.Outside)
ds$District <- as.integer(ds$District)
ds$Weapon <- as.integer(ds$Weapon)
ds$Hour_Factor <- as.integer(ds$Hour_Factor)
ds$Weekday <- as.integer(ds$Weekday)
ds$Month_names <- as.integer(ds$Month_names)

#structure
str(ds)
summary(ds)

#normalize all attributes except target factor and put in to ds2
norm <- function(x){return((x-min(x))/(max(x)-min(x)))}
ds2 <- as.data.frame(lapply(ds[,-7], norm))
summary(ds2)

#knn setting approx. 90:10 ratio
train_d <- ds2[1:240000, ]
test_d <- ds2[240001:276529,]
train_d_t <- ds[1:240000, 7]
test_d_t <- ds[240001:276529, 7]
pred <- knn(train_d, test_d, train_d_t$CrimeGroup, k=13)

#result
confusionMatrix(test_d_t$CrimeGroup, pred)
summary(ds$CrimeGroup)
summary(pred)

#find optimal k value
library(sjPlot)
sjc.elbow(ds2)

trControl <- trainControl(method  = "cv", number  = 5)

fit <- train(ds$CrimeGroup ~ .,
             method     = "knn",
             tuneGrid   = expand.grid(k = 9:10),
             trControl  = trControl,
             metric     = "Accuracy",
             data       = ds)


########### SVM #############
set.seed(9850)

gp <- runif(nrow(ds))
ds <- ds[order(gp),]

ds_train <- ds[1:900, c(2,3,7)]
ds_test <- ds[901:1100, c(2,3,7)]

svmfit <- svm(CrimeGroup ~., data= ds_train, kernel = "linear", cost = 10, scale = FALSE)
hist(svmfit$decision.values)
print(svmfit)

plot(svmfit, ds_train[ ,c("Weapon","District","CrimeGroup")])

svmfit$decision.values

summary(svmfit)
str(svmfit)

tuned <- tune(svm, CrimeGroup ~., data = ds_train, kernal )

p<-predict(svmfit, ds_test[,c("Weapon","District","CrimeGroup")], type="class")
plot(p)
p
table(p, ds_test$CrimeGroup)
mean(p== ds_test$CrimeGroup)
