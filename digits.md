Hand written digits
================
MA
May 14, 2018

We used deep learning to classify hand written digits and the accuracy reached up to 98%. In this project, we will use other methods to classify the hand written digits.

Load the hand written digits data.

``` r
library(keras)
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y
dim(x_train) <- c(nrow(x_train), 784)
dim(x_test) <- c(nrow(x_test), 784)
x_train <- x_train / 255
x_test <- x_test / 255
```

Note, the training data set 'x\_train' has 60,000 rows and 784 columns. Each row stands for a hand written digit and 'y\_train' are the corresponding 60,000 labels.

K Mean Method
-------------

Since this is a labeled data set, so we will use the mean of all images that have the same lable as the correct center for each digit. Given a new image, we will compute the distance between the new image and each center, then choose the one with smallest distance as the final correct classification for this new image.

``` r
#extract all training data with the same label
A0<-x_train[which(y_train==0),]
A1<-x_train[which(y_train==1),]
A2<-x_train[which(y_train==2),]
A3<-x_train[which(y_train==3),]
A4<-x_train[which(y_train==4),]
A5<-x_train[which(y_train==5),]
A6<-x_train[which(y_train==6),]
A7<-x_train[which(y_train==7),]
A8<-x_train[which(y_train==8),]
A9<-x_train[which(y_train==9),]
list.ma<-list(A0,A1,A2,A3,A4,A5,A6,A7,A8,A9)
```

Compute the "Centers"

``` r
#compute the "centers"
mean.ma<-matrix(nrow=10,nc=784)
for(i in 1:10){
  mean.ma[i,]<-colMeans(list.ma[[i]])
}
```

Computer the distance between a given digit and each center.

``` r
label.fun<-function(x,startpoint){#x is a vector with size 784, standing for a new image. And startpoint is a 10 by 784 matrix, standing for the 10 centers. 
  my.distance<-dist(rbind(x,startpoint))[1:10]#compute distance between x and each row of starpoint
  return(which.min(my.distance)-1)#return the lable for this new image
}
```

Classify new images based on the distance.

``` r
predict.ma<-apply(x_test,1,FUN=label.fun,startpoint=mean.ma)
```

Evaluate the K-Mean classification.

``` r
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
confusionMatrix(predict.ma,y_test)
```

    ## $positive
    ## NULL
    ## 
    ## $table
    ##           Reference
    ## Prediction    0    1    2    3    4    5    6    7    8    9
    ##          0  878    0   19    4    1   11   18    2   14   15
    ##          1    0 1092   71   24   22   63   27   59   39   22
    ##          2    7   10  781   25    2    2   22   22   11    7
    ##          3    2    3   33  814    0  118    0    1   83   10
    ##          4    2    0   31    1  811   21   31   20   12   83
    ##          5   58    7    3   49    3  612   32    2   36   12
    ##          6   25    3   23    8   16   27  827    0   13    1
    ##          7    1    0   18   15    1   10    0  856   10   27
    ##          8    7   20   50   58   10   13    1   13  718   18
    ##          9    0    0    3   12  116   15    0   53   38  814
    ## 
    ## $overall
    ##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
    ##      0.8203000      0.8001606      0.8126320      0.8277809      0.1135000 
    ## AccuracyPValue  McnemarPValue 
    ##      0.0000000            NaN 
    ## 
    ## $byClass
    ##          Sensitivity Specificity Pos Pred Value Neg Pred Value Precision
    ## Class: 0   0.8959184   0.9906874      0.9126819      0.9887143 0.9126819
    ## Class: 1   0.9621145   0.9631134      0.7695560      0.9949889 0.7695560
    ## Class: 2   0.7567829   0.9879572      0.8785152      0.9724509 0.8785152
    ## Class: 3   0.8059406   0.9721913      0.7650376      0.9780662 0.7650376
    ## Class: 4   0.8258656   0.9777112      0.8013834      0.9809746 0.8013834
    ## Class: 5   0.6860987   0.9778217      0.7518428      0.9695188 0.7518428
    ## Class: 6   0.8632568   0.9871710      0.8769883      0.9855360 0.8769883
    ## Class: 7   0.8326848   0.9908605      0.9125800      0.9810196 0.9125800
    ## Class: 8   0.7371663   0.9789497      0.7907489      0.9718434 0.7907489
    ## Class: 9   0.8067393   0.9736403      0.7745005      0.9782099 0.7745005
    ##             Recall        F1 Prevalence Detection Rate
    ## Class: 0 0.8959184 0.9042225     0.0980         0.0878
    ## Class: 1 0.9621145 0.8551292     0.1135         0.1092
    ## Class: 2 0.7567829 0.8131182     0.1032         0.0781
    ## Class: 3 0.8059406 0.7849566     0.1010         0.0814
    ## Class: 4 0.8258656 0.8134403     0.0982         0.0811
    ## Class: 5 0.6860987 0.7174678     0.0892         0.0612
    ## Class: 6 0.8632568 0.8700684     0.0958         0.0827
    ## Class: 7 0.8326848 0.8708037     0.1028         0.0856
    ## Class: 8 0.7371663 0.7630181     0.0974         0.0718
    ## Class: 9 0.8067393 0.7902913     0.1009         0.0814
    ##          Detection Prevalence Balanced Accuracy
    ## Class: 0               0.0962         0.9433029
    ## Class: 1               0.1419         0.9626140
    ## Class: 2               0.0889         0.8723701
    ## Class: 3               0.1064         0.8890660
    ## Class: 4               0.1012         0.9017884
    ## Class: 5               0.0814         0.8319602
    ## Class: 6               0.0943         0.9252139
    ## Class: 7               0.0938         0.9117726
    ## Class: 8               0.0908         0.8580580
    ## Class: 9               0.1051         0.8901898
    ## 
    ## $mode
    ## [1] "sens_spec"
    ## 
    ## $dots
    ## list()
    ## 
    ## attr(,"class")
    ## [1] "confusionMatrix"

We can see the overall accuracy is only 82.3%.

K Nearest Neighbor(KNN).
------------------------

KNN classifies an object by a majority vote of its k neighbors. It is easy to implememt and naturally handle multi-class cases. But it suffers expensive computing to find k nearest neighbors when the training set is large.

Especially for this data set, we have 60,000 traing points and 10,000 test points. That is, given a test point, we need to compute the distance between the test point and all 60,000 traing points. And we have to repeat this process 10,000 times in total since there are 10,000 test points.

In general, we might use 'apply' function or a 'for loop' to compute the distance between a new test point and all the 60,000 training points, like below.

``` r
#my.distance<-apply(x_train,1,FUN=function(a) dist(rbind(x_test[i,],a)))
```

There is no problem with this computing, but it takes a long time (about 32 hours) to finish 10,000 repetitions.

In order to be faster, we will vectorize the distance computing funstion instead of using a 'for loop' or 'apply function'. It is better to have a 8GB RAM to use the following vectorized distance computing fuction.

``` r
x_train_ma<-t(x_train)
dim(x_train_ma)<-c(1,60000*784)#convert training data to a vector with size 47,040,000
####faster distance computing function#####################
dist.function<-function(aa){#'aa' is a vector with size 784, standing for a test point
  dist.data<-rep(aa,60000)
  dist.A<-matrix((dist.data-x_train_ma)^2,nr=784,nc=60000)
  dist.B<-sqrt(colSums(dist.A))
  return(dist.B)
}
```

KNN Algorithm

``` r
k<-250 #sqrt of number of data points in training set
predict.ma<-vector(length=10000)
for(i in 1:10000){
  my.distance<-dist.function(x_test[i,])
  labels.ma<-rbind(y_train,my.distance)#put the distance and labels together
  labels.ma2<-labels.ma[,order(labels.ma[2,])]#sort from lowest distance to largest distance
  predict.ma[i]<-names(sort(table(labels.ma2[1,1:k]),decreasing=T)[1])#get the vote from the majority
}
```

Evaluate the performance of KNN

``` r
library(caret)
confusionMatrix(predict.ma,y_test)
```

    ## $positive
    ## NULL
    ## 
    ## $table
    ##           Reference
    ## Prediction    0    1    2    3    4    5    6    7    8    9
    ##          0  964    0   23    0    1    5   11    0   16   11
    ##          1    1 1131   80   16   29   20    9   63   19   13
    ##          2    0    2  866    4    0    0    0    2    2    2
    ##          3    0    1    8  951    0   23    0    0   28    8
    ##          4    0    0    5    1  881    2    5    3   11    9
    ##          5    5    0    1    9    1  806    3    0   25    2
    ##          6    9    1    6    1   11   15  930    0    6    2
    ##          7    1    0   30   12    2    3    0  930   10   18
    ##          8    0    0   13    8    1    1    0    0  830    1
    ##          9    0    0    0    8   56   17    0   30   27  943
    ## 
    ## $overall
    ##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
    ##      0.9232000      0.9146002      0.9178069      0.9283451      0.1135000 
    ## AccuracyPValue  McnemarPValue 
    ##      0.0000000            NaN 
    ## 
    ## $byClass
    ##          Sensitivity Specificity Pos Pred Value Neg Pred Value Precision
    ## Class: 0   0.9836735   0.9925721      0.9350145      0.9982161 0.9350145
    ## Class: 1   0.9964758   0.9717992      0.8189718      0.9995359 0.8189718
    ## Class: 2   0.8391473   0.9986619      0.9863326      0.9818022 0.9863326
    ## Class: 3   0.9415842   0.9924360      0.9332679      0.9934306 0.9332679
    ## Class: 4   0.8971487   0.9960080      0.9607415      0.9888803 0.9607415
    ## Class: 5   0.9035874   0.9949495      0.9460094      0.9905990 0.9460094
    ## Class: 6   0.9707724   0.9943597      0.9480122      0.9968954 0.9480122
    ## Class: 7   0.9046693   0.9915292      0.9244533      0.9891038 0.9244533
    ## Class: 8   0.8521561   0.9973410      0.9718970      0.9842554 0.9718970
    ## Class: 9   0.9345887   0.9846513      0.8723404      0.9926001 0.8723404
    ##             Recall        F1 Prevalence Detection Rate
    ## Class: 0 0.9836735 0.9587270     0.0980         0.0964
    ## Class: 1 0.9964758 0.8990461     0.1135         0.1131
    ## Class: 2 0.8391473 0.9068063     0.1032         0.0866
    ## Class: 3 0.9415842 0.9374076     0.1010         0.0951
    ## Class: 4 0.8971487 0.9278568     0.0982         0.0881
    ## Class: 5 0.9035874 0.9243119     0.0892         0.0806
    ## Class: 6 0.9707724 0.9592573     0.0958         0.0930
    ## Class: 7 0.9046693 0.9144543     0.1028         0.0930
    ## Class: 8 0.8521561 0.9080963     0.0974         0.0830
    ## Class: 9 0.9345887 0.9023923     0.1009         0.0943
    ##          Detection Prevalence Balanced Accuracy
    ## Class: 0               0.1031         0.9881228
    ## Class: 1               0.1381         0.9841375
    ## Class: 2               0.0878         0.9189046
    ## Class: 3               0.1019         0.9670101
    ## Class: 4               0.0917         0.9465783
    ## Class: 5               0.0852         0.9492685
    ## Class: 6               0.0981         0.9825660
    ## Class: 7               0.1006         0.9480992
    ## Class: 8               0.0854         0.9247485
    ## Class: 9               0.1081         0.9596200
    ## 
    ## $mode
    ## [1] "sens_spec"
    ## 
    ## $dots
    ## list()
    ## 
    ## attr(,"class")
    ## [1] "confusionMatrix"

We can see that the overall accuracy of classification is 92.3%.

We also used KNN function from class pakage to classify the digits and it took more than 6 hours. We did not use the system.time function to measure the time cost, just simply displayed the start time and end time.

``` r
#library(class)
#date()
#predictpack<-knn(x_train, x_test, y_train, k = 250, prob=TRUE)
#date()
```

date()

\[1\] "Sat May 12 08:13:20 2018"

predictpack&lt;-knn(x\_train, x\_test, y\_train, k = 250, prob=TRUE)

date()

\[1\] "Sat May 12 14:29:31 2018"

**Similarly, our own computing function took less than 4 hours, which is much faster than the KNN package.**

date()

\[1\] "Mon May 14 15:50:30 2018"

for(i in 1:10000){

my.distance&lt;-dist.function(x\_test\[i,\])

labels.ma&lt;-rbind(y\_train,my.distance)\#put the distance and labels together

lab .... \[TRUNCATED\]

date()

\[1\] "Mon May 14 19:35:27 2018"

Based on our own distance computing function, we also could use parallel computing like 'ParApply' to classify test data points more quickly, but it requires much more RAM, maybe 100 GB at least.

In comparison, deep learning is much better than K mean and KNN.
