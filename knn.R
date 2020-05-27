knn=read.csv(file="D:\\data science\\dataset\\Housing_loan.csv")
knn
dim(knn)
str(knn)
# As my Gender column is in factor so i will scale the data
Gender = ifelse(knn$Gender == "Male",1,0)
Gender
str(knn)
dim(knn)
# Normalization
library(caret)
preprc1=preProcess(knn[,-1],method=c("range"))
norm_knn=predict(preprc1,knn[,-1])
summary(norm_knn)
dim(norm_knn)
# Train & Test sample
set.seed(1)
d_data=sample(2,nrow(norm_knn),replace = TRUE,prob = c(0.8,0.2))
train_data=norm_knn[d_data==1,]
test_data=norm_knn[d_data==2,]
dim(train_data)
summary(train_data)
dim(test_data)
summary(test_data)
# package installation for knn
# Knn package is avaliable in class package
install.packages("class")
library(class)
knn_model1=knn(train=train_data[,-7],test=test_data[,-7],cl= train_data[,-7],k=5)
dim(train_data[,-7])
summary(train_data[,-7])
knn_model1

knn_actual=test_data[,7]
knn_actual
# Accuracy

knn_accuracy=mean(knn_model1==knn_actual)
knn_accuracy


# we always need to modify my k value so to over come i wil use loop
accuracy<-c()  # create a null set
for (i in 1:50) {
  set.seed(1)
  d_data=sample(2,nrow(norm_knn),replace = TRUE,prob = c(0.8,0.2))
  train_data=norm_knn[d_data==1,]
  test_data=norm_knn[d_data==2,]
  knn_model1=knn(train=train_data[,-7],test=test_data[,-7],cl=train_data[,7],k=i)
  dim(test_data[,-7])
  dim(train_data[-7])
  accuracy[i]=mean(knn_model1==train_data[,7])
  
  
}
print(accuracy,round(2))  
plot(x=1:50,y=accuracy,type='b')


# cross validation using knn.cv
knn.cv(1:50,accuracy)

#cross validation using for loop
knn_matrix=as.data.frame(matrix(0,5,5),stringsAsFactors = FALSE)
knn_matrix

for (i in 1:5) {
  for (j in 1:5) {
    set.seed(i)
    d_data=sample(2,nrow(norm_knn),replace = TRUE,prob = c(0.8,0.2))
    train_data=norm_knn[d_data==1,]
    test_data=norm_knn[d_data==2,]
    knn_model1=knn(train=train_data[,-7],test=test_data[,-7],cl=train_data[,7],k=j)
    knn_matrix[i,j]=mean(knn_model1==train_data[,7])
  }
}

print(knn_matrix)
plot(x=1:50,y=knn_matrix,type='b')









