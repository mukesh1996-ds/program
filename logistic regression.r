
log1=read.csv(file = "D:\\data science\\assignment\\dataset\\Credit.csv")
log1
head(log1)
dim(log1)
str(log1)
summary(log1)
install.packages("ggplot2")
library(caret)
pre1=preProcess(log1[,1:7],method=c("center","scale"))
log1[,1:7]= predict(pre1,log1[,1:7])
summary(log1)


log1$Student = ifelse(log1$Student == "No",1,0)
summary(log1)
log1$Student = factor(log1$Student, levels = c(0,1))

table(log1$Student)
prop.table(table(log1$Student))


set.seed(1)
train = sample(400,200)
traindata = log1[train,]
dim(traindata)


testdata = log1[-train,]
dim(testdata)
table(traindata$Student)
prop.table(table(traindata$Student))


table(testdata$Student)
prop.table(table(testdata$Student))

logitmode = glm(log1$Student~log1$Income+log1$Education+log1$Balance, family = "multinomial",data = log1[train,])
summary(logitmode)

pred = predict(logitmode, newdata = testdata$Student, type = "response")

Y_pred = traindata$Student
Y_act = testdata$Student

table(Y_pred,Y_act)
mean(Y_pred == Y_act)


install.packages("e1071")
library(caret)
confusionMatrix(traindata$Student,testdata$Student, positive = "1")
