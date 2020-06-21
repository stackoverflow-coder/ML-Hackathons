train <- read.csv("D:/Downloads/Hackathon - Analytics Vidhya/Cross-Selling/train.csv",na.strings="")
train_new <- train

i=0
li = 0
for(i in  1:length(train_new)){
  if((colSums(is.na(train_new[i]))/nrow(train_new))>0.5){
    li = c(li,i)
  }
}

#is.na.data.frame(train)
train_new <- train_new[,-c(li)]
summary(train_new$LEGAL_ENTITY) 
train_new$ZIP_CODE_FINAL = as.factor(train_new$ZIP_CODE_FINAL)
train_omit <- na.omit(train_new)

library(DMwR)
train_SMOTE <- SMOTE(RESPONDERS~., train_omit, perc.over = 600, perc.under = 500)
table(train_SMOTE$RESPONDERS)
train_SMOTE <- train_SMOTE[,-c(4)]
library(rpart)
library(rpart.plot)
library(gmodels)
library(randomForest)
tree <- rpart(RESPONDERS ~., data = train_SMOTE)
rpart.plot(tree)
pred = predict(tree,newdata = train_SMOTE, type = "class")
#print(tree)
CrossTable(train_SMOTE$RESPONDERS,pred,prop.chisq = F)


test <- read.csv("D:/Downloads/Hackathon - Analytics Vidhya/Cross-Selling/test_plBmD8c.csv",na.strings="")
test_new <- test
test_new <- test_new[,-c(li,4)]
summary(test$NEFT_CC_CATEGORY)
summary(train$NEFT_CC_CATEGORY)

test_new$RESPONDERS <- predict(tree, newdata = test_new, type = "class")

