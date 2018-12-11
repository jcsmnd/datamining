library(arules)
??arules
?apriori
data(package="arules")
Groceries
Groceries@itemInfo$labels
data(Groceries)
summary(Groceries)
str(Groceries)



inspect(Groceries[1:5])
sort(itemFrequency(Groceries),decreasing=TRUE)
itemFrequencyPlot(Groceries, support=0.01)
itemFrequencyPlot(Groceries, topN=10)

rule <- apriori(Groceries, parameter=list(support=0.005, confidence=0.3, minlen=2))

rule

summary(rule)
inspect(rule[1:10])
inspect(sort(rule, by="confidence")[1:10])
 
