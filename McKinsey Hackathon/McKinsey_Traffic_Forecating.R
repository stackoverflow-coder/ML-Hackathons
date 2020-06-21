train <- read.csv("D:/Downloads/Hackathon - Analytics Vidhya/McKinsey Hackathon/train_aWnotuB.csv")
#View(train)

#plot(train$DateTime, train$Vehicles)

#class(train$DateTime)

#head(train$DateTime)

library(dplyr)
library(timeDate)
#summary(train)
train$day <- as.factor(substr(train$DateTime, start = 9, stop = 10 ))
train$month <- as.factor(substr(train$DateTime, start = 6, stop = 7 ))
train$year <- as.factor(substr(train$DateTime, start = 1, stop = 4 ))
train$hour <- as.factor(substr(train$DateTime, start = 12, stop = 13 ))
train$weekday <- as.factor(weekdays(as.Date(train$DateTime)))
train$weekday_num <- as.factor(recode(train$weekday, 
                        "Sunday"=7,
                        "Monday"=1,
                        "Tuesday"=2,
                        "Wednesday"=3,
                        "Thursday"=4,
                        "Friday"=5,
                        "Saturday"=6))
train$workday <-ifelse(as.numeric(train$weekday_num) < 6, 1, 0)
train$weekend <-ifelse(as.numeric(train$weekday_num) >5, 1, 0)



#class(train$weekday)

#head(train$day)
#head(train$month)
#head(train$year)
#head(train$hour)


#plot(train$year, train$Vehicles)
#aggregate(Vehicles ~ year, data = train, sum)
#aggregate(Vehicles ~ month, data = train, sum)
#aggregate(Vehicles ~ Junction, data = train, sum)
#plot(aggregate(Vehicles ~ weekend, data = train, sum))

train_model <- train[,-c(1,4,9)]
train_1 = subset(train_model,Junction == 1)
train_1$hour_inc = as.numeric(row.names(train_1))
train_2 = subset(train_model,Junction == 2)
train_2$hour_inc = as.numeric(row.names(train_2))
train_3 = subset(train_model,Junction == 3)
train_3$hour_inc = as.numeric(row.names(train_3))
train_4 = subset(train_model,Junction == 4)
train_4$hour_inc = as.numeric(row.names(train_4))


#plot(train_4$DateTime,train_4$Vehicles)


library(MLmetrics)
library(randomForest)
rf_1 = randomForest(Vehicles ~ ., data = train_1, mtry=3)
RMSE(y_pred = predict(rf_1,train_1),y_true = train_1$Vehicles)
rf_2 = randomForest(Vehicles ~ ., data = train_2, mtry=3)
RMSE(y_pred = predict(rf_2,train_2),y_true = train_2$Vehicles)
rf_3 = randomForest(Vehicles ~ ., data = train_3, mtry=3)
RMSE(y_pred = predict(rf_3,train_3),y_true = train_3$Vehicles)
rf_4 = randomForest(Vehicles ~ .-month-year, data = train_4, mtry=3)
RMSE(y_pred = predict(rf_4,train_4),y_true = train_4$Vehicles)


test <- read.csv("D:/Downloads/Hackathon - Analytics Vidhya/McKinsey Hackathon/test_BdBKkAj.csv")
test$day <- as.factor(substr(test$DateTime, start = 9, stop = 10 ))
test$month <- as.factor(substr(test$DateTime, start = 6, stop = 7 ))
test$year <- as.factor(substr(test$DateTime, start = 1, stop = 4 ))
test$hour <- as.factor(substr(test$DateTime, start = 12, stop = 13 ))
test$weekday <- as.factor(weekdays(as.Date(test$DateTime)))
test$weekday_num <- as.factor(recode(test$weekday, 
                                     "Sunday"=7,
                                     "Monday"=1,
                                     "Tuesday"=2,
                                     "Wednesday"=3,
                                     "Thursday"=4,
                                     "Friday"=5,
                                     "Saturday"=6))
test$workday <-ifelse(as.numeric(test$weekday_num) <6, 1, 0)
test$weekend <-ifelse(as.numeric(test$weekday_num) >5, 1, 0)

test_model = test[,-c(1,3,8)]
test_1 = subset(test,Junction == 1)
test_1$hour_inc = as.numeric(row.names(test_1))
test_2 = subset(test,Junction == 2)
test_2$hour_inc = as.numeric(row.names(test_2))
test_3 = subset(test,Junction == 3)
test_3$hour_inc = as.numeric(row.names(test_3))
test_4 = subset(test,Junction == 4)
test_4$hour_inc = as.numeric(row.names(test_4))


test_1$Vehicles  = predict(rf_1,newdata = test_1)
test_2$Vehicles  = predict(rf_2,newdata = test_2)
test_3$Vehicles  = predict(rf_3,newdata = test_3)
test_4$Vehicles  = predict(rf_4,newdata = test_4)

test_pred = rbind(test_1,test_2,test_3,test_4)
head(test_pred)

test_pred = test_pred[,c(3,11)]
write.csv(train_1,file = "D:/Downloads/Hackathon - Analytics Vidhya/McKinsey Hackathon/final_output.csv")



