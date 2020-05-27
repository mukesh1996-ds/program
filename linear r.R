

#we have to predict the income with the help of linear regression 
#load the dataset into r
l1=read.csv(file = "D:\\R programming\\Credit.csv")
l1
#check the dimension
dim(l1)
#summary
summary(l1)
# as our dataset contain both type of variables i.e
#continous & discrete
#linar regression only work for continous variable so we need to 
# remove the discrete variable.
l1=l1[,-c(7)]
list(l1)
# From the above command i.e l1[,-c(1,8,9,10,10)] which [] represent the entire row &column
# our column contain discrete variable so we have removed the element -c(1,8,9,10,11)
dim(l1)
summary(l1)
# as we can see our data set consists of different ranges so we need to transform it to +3 to -3
# to preform this operation we can do this in two ways i.e STANDIZATION & NORMALIZATION
install.packages("caret")
library(caret)
pre1=preProcess(l1[,1:7],method=c("center","scale"))
l1[,1:7]= predict(pre1,l1[,1:7])
summary(l1)
#once our data is normalized then we can preform our operation
#ploting of graphs
#histogram
hist(l1$Income)
hist(l1$Limit)
hist(l1$Rating)
hist(l1$Cards)
hist(l1$Age)
hist(l1$Education)
hist(l1$Balance)
#boxplot
boxplot(l1$Income)
boxplot(l1$Limit)
boxplot(l1$Rating)
boxplot(l1$Cards)
boxplot(l1$Age)
boxplot(l1$Education)
boxplot(l1$Balance)
#scatter plot
plot(l1$Income,l1$Limit)
plot(l1$Income,l1$Rating)
plot(l1$Income,l1$Cards)
plot(l1$Income,l1$Age)
plot(l1$Income,l1$Education)
plot(l1$Income,l1$Balance)
# we need also to check the corelation 
cor(l1)
#we need to remove all the multicolinearty 
#multicolinearty : Their should not be any kind of relationship between the variables
#limit
#limit + Education
#rating
#rating + education
#cards
#cards + education 
# The above mention are having weak negative (-ve) relation so we need to preform 
#model construction

m1=lm(Income~ Limit,data=l1)
summary(m1)

m2=lm(Income~ Limit+Education,data=l1)
summary(m2)

m3=lm(Income~Rating ,data=l1)
summary(m3)

m4=lm(Income~Rating + Education ,data=l1)
summary(m4)

m5=lm(Income~Cards ,data=l1)
summary(m5)

m6=lm(Income~Cards + Education  ,data=l1)
summary(m6)

# from all the above observation we can clearly see that our R^2 adjusted is not good 
# so we need to change the algorithm.
#lets try it with POLYNOMIAL REGRESSION
#it will create the equation as follows
#y=b0+b1x1+b1x1^2+b1x1^3........
#synatx is (ploy)
# again  follow the same combinations
plot(l1$Income,l1$Limit)
m1=lm(Income~ Limit,data=l1)
summary(m1)
# To calculate mean square error we can do it directly
#FOr every iteration the mean square error will be reduced as you can see it below
mean((l1$Income-m1$fitted.values)^2)


m1.1=lm(Income~poly(Limit,2),data=l1)
summary(m1.1)
mean((l1$Income-m1.1$fitted.values)^2)

m1.2=lm(Income~poly(Limit,3),data=l1)
summary(m1.2)
mean((l1$Income-m1.2$fitted.values)^2)

# now we will apply cross vaildation i.e dividing the entire dataset into two parts 
# training and test
# prediction should be done on test dataset only
# steps followed are
# sample is the syntax to divide the data set into training and test 

# it's command for fixing the fixed value
set.seed(100) 
train = sample(400,240)

m1=lm(Income~ Limit,data=l1,subset = train)
# The below statement help to predict the test data 
mean((l1$Income-predict(m1,l1))[-train]^2)

m8=lm(Income~ poly(Limit,2),data=l1,subset = train)
# The below statement help to predict the test data 
mean((l1$Income-predict(m8,l1))[-train]^2)


m9=lm(Income~ poly(Limit,3),data=l1,subset = train)
# The below statement help to predict the test data 
mean((l1$Income-predict(m9,l1))[-train]^2)


m10=lm(Income~ poly(Limit,4),data=l1,subset = train)
# The below statement help to predict the test data 
mean((l1$Income-predict(m9,l1))[-train]^2)


m11=lm(Income~ poly(Limit,5),data=l1,subset = train)
# The below statement help to predict the test data 
mean((l1$Income-predict(m11,l1))[-train]^2)

# we need to create a empty set
cv.error <-c()

# we can const it in two ways
#first way
for (i in 1:6) {
  set.seed(i)
  train=sample(400,240)
  m1=lm(Income~ poly(Limit,i),data=l1,subset = train)
  cv.error[i]=mean((l1$Income-predict(m1,l1))[-train]^2)
  
    
}
print(cv.error)
plot(x=1:6,y=cv.error,type="b")


## we have construct the complexity for single 
#we can even create a multiple complexity 
#second way
testerror=as.data.frame(matrix(0,6,6),stringsAsFactors = FALSE)
# The above command is used to create 6x6 matrix
# include the variable in it using for loop i.e nested for loop

for(i in 1:6){
  for(j in 1:6){
    set.seed(i)
    train=sample(400,240)
    m1=lm(Income~ poly(Limit,j),data=l1,subset = train)
    testerror[i]=mean((l1$Income-predict(m1,l1))[-train]^2)
  }
}

print(testerror)

# now we need to find the colmean value

colMeans(testerror)

plot(x=1:6,y=colMeans(testerror),type="b")



----------------------------------------------------------------------------------------
testerror1=as.data.frame(matrix(0,10,2),stringsAsFactors = FALSE)
for(i in 1:2){
  set.seed(i)
  train=sample(400,240)
  m1=lm(Income~ poly(Limit,j),data=l1,subset = train)
  m2=lm(Income~Limit+Education,data=l1,subset = train)
  testerror1[i,1]=mean((l1$Income-predict(m1,l1))[-train]^2)
  testerror2[i,2]=mean((l1$Income-predict(m2,l1))[-train]^2)
    
}

print(testerror1)

# now we need to find the colmean value

colMeans(testerror1)

plot(x=1:6,y=colMeans(testerror1),type="b")








