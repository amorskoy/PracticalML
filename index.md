# ML experiment on movement class

Here you see experiment description for 'classe' prediction on Practical ML assignemnt.
Research plan is:

1. Load all the data and libs

```r
library(caret);
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
setwd("/projects/MOOC/Practical ML/assignment/")
inTrain = read.csv("pml-training.csv", stringsAsFactors=F)
outTest = read.csv("pml-testing.csv", stringsAsFactors=F)
```

2. Inspect dataset and decide on variables conversion for next steps

```
## 'data.frame':	19622 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : chr  "carlitos" "carlitos" "carlitos" "carlitos" ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp          : chr  "05/12/2011 11:23" "05/12/2011 11:23" "05/12/2011 11:23" "05/12/2011 11:23" ...
##  $ new_window              : chr  "no" "no" "no" "no" ...
##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##   [list output truncated]
```

3. Remove row names as it is not useful as predictor

```r
inTrain$X = NULL
outTest$X = NULL
```

4. Convert dates to be useful for prediction and cleanup unused date fields after that

```r
inTrain$cvtd_timestamp = strptime(inTrain$cvtd_timestamp, format="%d/%m/%Y %H:%M")
inTrain$hour = inTrain$cvtd_timestamp$hour
inTrain$weekday = weekdays(inTrain$cvtd_timestamp)
inTrain$cvtd_timestamp = NULL
inTrain$raw_timestamp_part_1 = NULL
inTrain$raw_timestamp_part_2 = NULL
outTest$cvtd_timestamp = strptime(outTest$cvtd_timestamp, format="%d/%m/%Y %H:%M")
outTest$hour = outTest$cvtd_timestamp$hour
outTest$weekday = weekdays(outTest$cvtd_timestamp)
outTest$cvtd_timestamp = NULL
outTest$raw_timestamp_part_1 = NULL
outTest$raw_timestamp_part_2 = NULL
```

5. Factorize some char vars to help predictor in classification

```r
inTrain$weekday = as.factor(inTrain$weekday)
inTrain$new_window = as.factor(inTrain$new_window)
inTrain$user_name = as.factor(inTrain$user_name)
inTrain$classe = as.factor(inTrain$classe)

outTest$weekday = as.factor(outTest$weekday)
outTest$new_window = as.factor(outTest$new_window)
outTest$user_name = as.factor(outTest$user_name)
```

6. Based on incoming data, using apply, table and colMeans, find columns with low information gain (most values are "" or NA) and drop them

```r
inTrain$kurtosis_roll_belt = NULL
inTrain$kurtosis_picth_belt = NULL
inTrain$kurtosis_yaw_belt = NULL
inTrain$skewness_roll_belt = NULL
inTrain$skewness_roll_belt.1 = NULL
inTrain$skewness_yaw_belt = NULL
inTrain$max_yaw_belt = NULL
inTrain$min_yaw_belt = NULL
inTrain$kurtosis_roll_arm = NULL
inTrain$kurtosis_picth_arm = NULL
inTrain$kurtosis_yaw_arm = NULL
inTrain$skewness_roll_arm = NULL
inTrain$skewness_pitch_arm = NULL
inTrain$skewness_yaw_arm = NULL
inTrain$kurtosis_roll_dumbbell = NULL
inTrain$kurtosis_picth_dumbbell = NULL
inTrain$kurtosis_yaw_dumbbell = NULL
inTrain$skewness_roll_dumbbell = NULL
inTrain$skewness_pitch_dumbbell = NULL
inTrain$skewness_yaw_dumbbell = NULL
inTrain$max_yaw_dumbbell = NULL
inTrain$min_yaw_dumbbell = NULL
inTrain$amplitude_yaw_belt = NULL
inTrain$kurtosis_roll_forearm = NULL
inTrain$kurtosis_picth_forearm = NULL
inTrain$kurtosis_yaw_forearm = NULL
inTrain$skewness_roll_forearm = NULL
inTrain$skewness_pitch_forearm = NULL
inTrain$skewness_yaw_forearm = NULL
inTrain$max_yaw_forearm = NULL
inTrain$min_yaw_forearm = NULL
inTrain$amplitude_yaw_forearm = NULL

outTest$amplitude_yaw_dumbbell = ""

naCols = colMeans(is.na(inTrain))
naCols = subset(naCols, naCols>0)
naCols = names(naCols)
inTrain = inTrain[, !(names(inTrain) %in% naCols)]
```
  
7. Split input train set into train and test subset using caret

```r
parts = createDataPartition(y=inTrain$classe, p=0.75, list=F)
train = inTrain[parts,]
test = inTrain[-parts,]
```

8. Inspect on cleanup results

```r
dim(train)
```

```
## [1] 14718    59
```

9. As we droped 2/3 of vars, we still have 59 of them. I prefer to think on non-linear solution.
This way I want to perform cross-validation. Also I know that tree-based solutions are good for
non-linear cases. As of potential noise and uncertainty on some data values, random forest is preffered over single tree - it will boost-reduce noices


```r
tc2 = trainControl(method = "cv", number = 20, verboseIter = T)
modFitLarge = train(classe~., data=train, method='rf', trControl=tc2)
```



10. Inspect result model

```r
modFitLarge$bestTune
```

```
##   mtry
## 2   33
```

```r
modFitLarge$results
```

```
##   mtry  Accuracy     Kappa  AccuracySD     KappaSD
## 1    2 0.9940206 0.9924360 0.003035264 0.003840328
## 2   33 0.9972136 0.9964752 0.002227638 0.002818312
## 3   64 0.9930694 0.9912333 0.003083485 0.003901008
```

```r
modFitLarge$finalModel$confusion
```

```
##      A    B    C    D    E  class.error
## A 4183    1    0    0    1 0.0004778973
## B    4 2841    3    0    0 0.0024578652
## C    0    5 2562    0    0 0.0019477990
## D    0    0    9 2402    1 0.0041459370
## E    0    1    0    4 2701 0.0018477458
```

11. As we see, worst in-sample error is 0.4% which is good signal

12. Ensure high out-of-sample accuracy

```r
pred = predict(modFitLarge, newdata=test)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
table(test$classe, pred)
```

```
##    pred
##        A    B    C    D    E
##   A 1395    0    0    0    0
##   B    1  948    0    0    0
##   C    0    0  855    0    0
##   D    0    0    0  804    0
##   E    0    0    0    0  901
```

```r
sum(diag(table(test$classe, pred)))/nrow(test)
```

```
## [1] 0.9997961
```

13. As of high in-sample accuracy and low error rate, I expect low out-of-sample error rate on problem cases.
Also overfitting is still possible. But due to massive cleanup of 2/3 noisy vars,
this effect reduced. Nice idea is to reseach covariation matrix and apply k-means clustering
for deeper understanding of data - but it makes research more complex.

```r
predProblem = predict(modFitLarge, newdata=outTest)
```

14. Submit that 20 predProblem values
