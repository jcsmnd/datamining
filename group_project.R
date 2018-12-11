library(readr)
library(knitr)
library(lattice)
library(caret)
library(class)
library(ggplot2)
library(Amelia)
library(VIM)

rm(m_data2)
#######
m_data_original  = read.csv(file.choose(), header = T)
m_data<-m_data_original[, c(7,8,9,10,11,15,20,22,26,28)]
missmap(m_data)
##Method.1 Handle with NA inpute method using kNN function VIM library but not work due to memory problem
m_data2 <- kNN(m_data, variable = c("District"), k = 5)

#exclude weapon column due to too many missing value
m_data <- m_data[,-2]

##Method.2 Handlie with NA elemination <- this is I'm using it
m_data2 <- m_data[complete.cases(m_data),]
missmap(m_data2)

str(m_data2)

#######KNN###########
m_data2$Inside.Outside = as.numeric(m_data2$Inside.Outside)
m_data2$Weapon = as.numeric(m_data2$Weapon)
m_data2$District = as.numeric(m_data2$District)
m_data2$Neighborhood = as.numeric(m_data2$Neighborhood)
m_data2$Premise = as.numeric(m_data2$Premise)
m_data2$Weekday = as.numeric(m_data2$Weekday)
m_data2$Month_names = as.numeric(m_data2$Month_names)

set.seed(9800)
gp<-runif(nrow(m_data2))
m_data2<-m_data2[order(gp),]

##normal min max##
norm <- function(x){+return((x-min(x))/(max(x)-min(x)))}
m_data_n <- as.data.frame(lapply(m_data2[,c(1,2,3,4,5,6,7,8)], norm))

summary(m_data_n)

train_d <- m_data_n[1:205000,]
test_d <- m_data_n[205001:232945,]

train_d_t <- m_data2[1:205000, 9]
test_d_t <- m_data2[205001:232945, 9]

## perfom knn classification, we can change k value)
pred <- knn(train=train_d,test=test_d, cl=train_d_t, k=9)


## result of knn
confusionMatrix(test_d_t, pred)
qplot(pred)
qplot(test_d_t)
###########

original_data = read.csv("BPD_Part_1_Victim_Based_Crime_Data.csv", header = TRUE)

str(original_data)
summary(original_data)

str(original_data$Weapon)
summary(original_data$Weapon)

qplot(original_data$Weapon)

summary(original_data$CrimeCode)
str(original_data$CrimeCode)

qplot(original_data$Description)
ncol(original_data)
names(original_data)


##data preprocess##
rm(modified_data01)
modified_data01<-modified_data02

m.data$Inside.Outside[m.data$Inside.Outside=="O"] <- "Outside"
m.data$Inside.Outside[m.data$Inside.Outside=="I"] <- "Inside"
m.data$Inside.Outside[m.data$Inside.Outside==""] <- NA
m.data$Weapon[m.data$Weapon==""] <- NA
m.data$Premise[m.data$Premise==""] <- NA
m.data$District[m.data$District==""] <- NA
m.data$Location[m.data$Location==""] <- NA
m.data$Location.1[m.data$Location.1==""] <- NA

qplot(m_data$District)
summary(modified_data01$District)
ggplot(data=m_data, aes(Weapon, Description))

qplot(m_data$Weapon, m_data$Description)

c.data <- m.data[complete.cases(m.data),]


missmap(m_data)
boxplot(m_data)
m_data <- m_data[,-10]


summary(c.data)

summary(c.data$Inside.Outside)
str(c.data$Inside.Outside)

c.data$Inside.Outside <- factor(c.data$Inside.Outside)
c.data$Weapon <- factor(c.data$Weapon)
c.data$District <- factor(c.data$District)

##modified one preprocess##
m_data<-m_data[, c(-15)]
missmap(m_data)

ggplot(m_data)+
  aes(x=Weekday)+
  geom_bar(color='black',fill="blue")+
  scale_y_continuous(breaks = seq(5000,50000,5000),limits = c(0,50000))+ 
  geom_text(stat="count",aes(label=..count..),vjust=-1)+
  labs(title="Number of Incidents",x="Weekday",y="Number of Incidents")

qplot(m_data$District, m_data$Month, col=m_data$CrimeGroup)

kNN(m_data)
plot(m_data$Total.Incidents)

qplot(pred)

############## SVM ##############



