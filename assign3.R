library(ggplot2)
library(lattice)
library(caret)
library(cluster)
library(factoextra)
library(fpc)
library(NbClust)
pkgs <- c("cluster", "fpc", "NbClust")
install.packages(pkgs)

d1<-iris
d1$Species=NULL
head(d1)
km<-kmeans(d1,3)
?kmeans

km
km$size
km$cluster

qplot(iris$Petal.Length, iris$Petal.Width, col=km$cluster)
qplot(iris$Petal.Length, iris$Petal.Width, col=iris$Species)

qplot(iris$Sepal.Length, iris$Sepal.Width, col=km$cluster)
qplot(iris$Sepal.Length, iris$Sepal.Width, col=iris$Species)

table(iris$Species, km$cluster)
confusionMatrix(iris$Species, km$cluster)

km$cluster
km$centers
km$size
km$size

irisData <- iris[,1:4]
totalwSS<-c()
# kmeans clustering for 15 times in a loop
for (i in 1:15)
{
  clusterIRIS <- kmeans(irisData, centers=i)    
  totalwSS[i]<-clusterIRIS$tot.withinss      
}

qplot(x=1:15,                         # x= No of clusters, 1 to 15
     y=totalwSS,                      # tot_wss for each
     type="b",                       # Draw both points as also connect them
     geom=c("point","line"),
     xlab="Number of Clusters",
     ylab="Groups of Sum of squares")
km.res <- eclust(d1, "kmeans", k = 3,
                 nstart = 25, graph = FALSE)

#k-means cluster visulization
fviz_cluster(km.res, geom = "point", frame.type = "norm")
fviz_silhouette(km.res)

fviz_cluster(km.res, geom = "point")
fviz_cluster(pam.res, geom = "point")
#silhousette for k means
plot(silhouette(km$cluster,dist(d1)))
fviz_silhouette(km.res)

head(km.res$silinfo$widths)
km.res$silinfo$widths
sil <- silhouette(km.res$cluster, dist(d1))
si.sum <- summary(sil)

si.sum$clus.avg.widths
si.sum$avg.width
si.sum$clus.sizes

#pam
pam<-pam(d1,3)
pam$clusinfo
pam$medoids
fviz_cluster(pam, geom = "point", frame.type = "norm")
plot(silhouette(km$cluster,dist(d1)),col = c("2", "3", "4"), ylab="Number of Clusters", xlab="Silhouette distance")
plot(silhouette(pam$clustering,dist(d1)),col = c("2", "4", "3"), ylab="Number of Clusters", xlab="Silhouette distance")

# PAM clustering
pam.res <- eclust(d1, "pam", k = 3, graph = FALSE)

qplot(iris$Petal.Length, iris$Petal.Width, col=pam$cluster)
qplot(iris$Petal.Length, iris$Petal.Width, col=km$cluster)
qplot(iris$Petal.Length, iris$Petal.Width, col=iris$Species)


# Enhanced hierarchical clustering
hc2 <- eclust(d1, "hclust", k = 3, method = "ward.D", graph = FALSE) 
fviz_dend(hc2, rect = TRUE, show_labels = TRUE,) 

d1.dist<-dist(d1,method="euclidean")
?dist
hc2 <- hclust(d1.dist, method="ward.D")
?hclust
plot(hc, labels=iris$Species)
plot(hc.res, labels=iris$Species)
plot(hc2, labels=iris$Species)

summary(hc2)

hc2$cluster
