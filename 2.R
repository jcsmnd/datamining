library(readr)
library(knitr)
library(caret)
library(ggplot2)
library(readr)
library("")
plot(data1$Weapon, data1$Total.Incidents)

qplot(data1$Weapon)  


###########################

library(rpart)
library(rpart.plot)

colnames(train_data)

train_data_n <- train_data[,-1]
test_data_n <- test_data[,-1]

wild_to<-subset(train_data_n, select=c(Wilderness_Area_1,Wilderness_Area_2,Wilderness_Area_3,Wilderness_Area_4))
merge01<-cbind(wild_to[0], Wilderness_Areas = names(wild_to)[max.col(wild_to == 1)])

soil_to<-train_data_n[, 15:54]
merge02<-cbind(soil_to[0], Soil_Types = names(soil_to)[max.col(soil_to == 1)])

merge03 <- cbind(train_data_n, merge01)
merge03 <- cbind(merge03, merge02)

df<-merge03[, c(1,2,3,4,5,6,7,8,9,10,55,56,57)]

df$Soil_Types = substr(df$Soil_Types,1,2)
df$Soil_Types <- as.factor(df$Soil_Types)
str(df)

plot(df$Cover_Type, df$Soil_Types)
qplot(df$Cover_Type)

norm <- function(x){ +return((x-min(x)) / (max(x) - min(x)))}
norm(c(1,2,3,4,5))
df_n <- as.data.frame(lapply(df[,c(1:10)],norm))
summary(df_n)

require(class)
tree_train <- df_n[1:5000,]
tree_test <- df_n[5001:481011,]
tree_train_target <- df[1:5000, 11]
tree_test_target <- df[5001:481011, 11]
m1<-knn(train = tree_train, test = tree_test, cl=tree_train_target, k=5)
table(tree_test_target, m1)

tree_test_target<-as.factor(tree_test_target)

library(lattice)
library(caret)

confusionMatrix(tree_test_target, m1)
qplot(m1)

knn_result<-cbind(m1, df_n)
