##############################################################
# file input and check the basic thing
h1=read.csv(file = "F:\\my project\\project.csv")
dim(h1)
names(h1)
str(h1)
summary(h1)
#############################################
#once file is loaded and all the basic things are tested the it's time to construct the graphs
#let's construct boxplot first
boxplot(h1$age)
boxplot(h1$weight)
boxplot(h1$bsa)
boxplot(h1$dur)
boxplot(h1$pulse)
boxplot(h1$stress)
###########################################
#second is histogram to verify the symmetric shape
hist(h1$age)
hist(h1$weight)
hist(h1$bsa)
hist(h1$dur)
hist(h1$pulse)
hist(h1$stress)
##############################################
#third is to check the scatter plot
plot(h1$age)
plot(h1$weight)
plot(h1$bsa)
plot(h1$dur)
plot(h1$pulse)
plot(h1$stress)
###############################################
#now check the corelation of all the element in data set
cor(h1)
##################################################
# model construction
ml1=lm(h1$bp~age+weight+bsa+dur+pulse+stress,data=h1)
ml1
summary(ml1)
######################################################
#mean square error
msq1=(sum(h1$bp-ml1$fitted.values)^2)/length(h1$bp)
#########################################################
#data transformation

library(caret)
preprocess2=preProcess(h1[,2:8],method=c("center","scale"))
norm1=predict(preprocess2,h1[,2:8])
summary(norm1)

######################################################
#construct model again

model2=lm(h1$bp~.,data=norm1)
summary(model2)

###################################################
# individual checking the best fit value

ml1=lm(h1$bp~h1$age,data=norm1)
summary(ml1)

ml2=lm(h1$bp~h1$weight,data=norm1)
summary(ml2)

ml3=lm(h1$bp~h1$bsa,data=norm1)
summary(ml3)


ml4=lm(h1$bp~h1$dur,data=norm1)
summary(ml4)

ml5=lm(h1$bp~h1$pulse,data=norm1)
summary(ml5)

ml6=lm(h1$bp~h1$stress,data=norm1)
summary(ml6)

##############################
#combine age + everyother

ml01=lm(h1$bp~h1$age+h1$weight,data=norm1)
summary(ml01)

ml02=lm(h1$bp~h1$age+h1$bsa,data=norm1)
summary(ml02)

ml03=lm(h1$bp~h1$age+h1$dur,data=norm1)
summary(ml03)

ml04=lm(h1$bp~h1$age+h1$pulse,data=norm1)
summary(ml04)

ml05=lm(h1$bp~h1$age+h1$stress,data=norm1)
summary(ml05)


###############################################
#combine weight+ every thing

ml07=lm(h1$bp~h1$weight+h1$age,data=norm1)
summary(ml07)


ml08=lm(h1$bp~h1$weight+h1$bsa,data=norm1)
summary(ml08)


ml09=lm(h1$bp~h1$weight+h1$dur,data=norm1)
summary(ml09)


ml10=lm(h1$bp~h1$weight+h1$pulse,data=norm1)
summary(ml10)


ml11=lm(h1$bp~h1$weight+h1$stress,data=norm1)
summary(ml11)

###############################################
#combine bsa + every thing


ml12=lm(h1$bp~h1$bsa+h1$age,data=norm1)
summary(ml12)



ml13=lm(h1$bp~h1$bsa+h1$weight,data=norm1)
summary(ml13)

ml14=lm(h1$bp~h1$bsa+h1$dur,data=norm1)
summary(ml14)

ml15=lm(h1$bp~h1$bsa+h1$pulse,data=norm1)
summary(ml15)


ml16=lm(h1$bp~h1$bsa+h1$stress,data=norm1)
summary(ml16)

###############################################################
#combine dur + everything

ml17=lm(h1$bp~h1$dur+h1$age,data=norm1)
summary(ml17)


ml18=lm(h1$bp~h1$dur+h1$weight,data=norm1)
summary(ml18)


ml19=lm(h1$bp~h1$dur+h1$bsa,data=norm1)
summary(ml19)


ml20=lm(h1$bp~h1$dur+h1$pulse,data=norm1)
summary(ml20)


ml21=lm(h1$bp~h1$dur+h1$stress,data=norm1)
summary(ml21)

############################################
# pulse + every

ml22=lm(h1$bp~h1$pulse+h1$age,data=norm1)
summary(ml22)


ml23=lm(h1$bp~h1$pulse+h1$weight,data=norm1)
summary(ml23)


ml24=lm(h1$bp~h1$pulse+h1$bsa,data=norm1)
summary(ml24)


ml25=lm(h1$bp~h1$pulse+h1$dur,data=norm1)
summary(ml25)



ml26=lm(h1$bp~h1$pulse+h1$stress,data=norm1)
summary(ml26)


########################################################
# stress + every


ml27=lm(h1$bp~h1$stress+h1$age,data=norm1)
summary(ml27)



ml28=lm(h1$bp~h1$stress+h1$weight,data=norm1)
summary(ml28)



ml29=lm(h1$bp~h1$stress+h1$bsa,data=norm1)
summary(ml29)


ml30=lm(h1$bp~h1$stress+h1$dur,data=norm1)
summary(ml30)


ml31=lm(h1$bp~h1$stress+h1$pulse,data=norm1)
summary(ml31)


################################################################
# totla check


ml06=lm(h1$bp~h1$age+h1$stress+h1$pulse+h1$dur+h1$bsa+h1$weight,data=norm1)
summary(ml06)




#########################################################################
# best fit model

ml32=lm(h1$bp~h1$age+h1$weight+h1$bsa+h1$pulse,data=norm1)
summary(ml32)


ml33=lm(h1$bp~h1$age+h1$weight,data=norm1)
summary(ml33)



#ml34=lm(h1$bp~h1$age+h1$weight+h1$bsa,data=norm1)
#summary(ml34)



ml35=lm(h1$bp~h1$age+h1$pulse+h1$bsa,data=norm1)
summary(ml35)


ml36=lm(h1$bp~h1$weight+h1$pulse+h1$bsa,data=norm1)
summary(ml36)




































