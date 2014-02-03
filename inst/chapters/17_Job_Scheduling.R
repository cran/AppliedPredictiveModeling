################################################################################
### R code from Applied Predictive Modeling (2013) by Kuhn and Johnson.
### Copyright 2013 Kuhn and Johnson
### Web Page: http://www.appliedpredictivemodeling.com
### Contact: Max Kuhn (mxkuhn@gmail.com) 
###
### Chapter 17: Case Study: Job Scheduling
###
### Required packages: AppliedPredictiveModeling, C50, caret, doMC (optional),
###                    earth, Hmisc, ipred, tabplot, kernlab, lattice, MASS,
###                    mda, nnet, pls, randomForest, rpart, sparseLDA, 
###
### Data used: The HPC job scheduling data in the AppliedPredictiveModeling
###            package.
###
### Notes: 
### 1) This code is provided without warranty.
###
### 2) This code should help the user reproduce the results in the
### text. There will be differences between this code and what is is
### the computing section. For example, the computing sections show
### how the source functions work (e.g. randomForest() or plsr()),
### which were not directly used when creating the book. Also, there may be 
### syntax differences that occur over time as packages evolve. These files 
### will reflect those changes.
###
### 3) In some cases, the calculations in the book were run in 
### parallel. The sub-processes may reset the random number seed.
### Your results may slightly vary.
###
################################################################################

library(AppliedPredictiveModeling)
data(schedulingData)

### Make a vector of predictor names
predictors <- names(schedulingData)[!(names(schedulingData) %in% c("Class"))]

### A few summaries and plots of the data
library(Hmisc)
describe(schedulingData)

library(tabplot)
tableplot(schedulingData[, c( "Class", predictors)])

mosaicplot(table(schedulingData$Protocol, 
                 schedulingData$Class), 
           main = "")

library(lattice)
xyplot(Compounds ~ InputFields|Protocol,
       data = schedulingData,
       scales = list(x = list(log = 10), y = list(log = 10)),
       groups = Class,
       xlab = "Input Fields",
       auto.key = list(columns = 4),
       aspect = 1,
       as.table = TRUE)


################################################################################
### Section 17.1 Data Splitting and Model Strategy

## Split the data

library(caret)
set.seed(1104)
inTrain <- createDataPartition(schedulingData$Class, p = .8, list = FALSE)

### There are a lot of zeros and the distribution is skewed. We add
### one so that we can log transform the data
schedulingData$NumPending <- schedulingData$NumPending + 1

trainData <- schedulingData[ inTrain,]
testData  <- schedulingData[-inTrain,]

### Create a main effects only model formula to use
### repeatedly. Another formula with nonlinear effects is created
### below.
modForm <- as.formula(Class ~ Protocol + log10(Compounds) +
  log10(InputFields)+ log10(Iterations) +
  log10(NumPending) + Hour + Day)

### Create an expanded set of predictors with interactions. 

modForm2 <- as.formula(Class ~ (Protocol + log10(Compounds) +
  log10(InputFields)+ log10(Iterations) +
  log10(NumPending) + Hour + Day)^2)


### Some of these terms will not be estimable. For example, if there
### are no data points were a particular protocol was run on a
### particular day, the full interaction cannot be computed. We use
### model.matrix() to create the whole set of predictor columns, then
### remove those that are zero variance

expandedTrain <- model.matrix(modForm2, data = trainData)
expandedTest  <- model.matrix(modForm2, data = testData)
expandedTrain <- as.data.frame(expandedTrain)
expandedTest  <-  as.data.frame(expandedTest)

### Some models have issues when there is a zero variance predictor
### within the data of a particular class, so we used caret's
### checkConditionalX() function to find the offending columns and
### remove them

zv <- checkConditionalX(expandedTrain, trainData$Class)

### Keep the expanded set to use for models where we must manually add
### more complex terms (such as logistic regression)

expandedTrain <-  expandedTrain[,-zv]
expandedTest  <-  expandedTest[, -zv]

### Create the cost matrix
costMatrix <- ifelse(diag(4) == 1, 0, 1)
costMatrix[4, 1] <- 10
costMatrix[3, 1] <- 5
costMatrix[4, 2] <- 5
costMatrix[3, 2] <- 5
rownames(costMatrix) <- colnames(costMatrix) <- levels(trainData$Class)

### Create a cost function
cost <- function(pred, obs)
{
  isNA <- is.na(pred)
  if(!all(isNA))
  {
    pred <- pred[!isNA]
    obs <- obs[!isNA]
    
    cost <- ifelse(pred == obs, 0, 1)
    if(any(pred == "VF" & obs == "L")) cost[pred == "L" & obs == "VF"] <- 10
    if(any(pred == "F" & obs == "L")) cost[pred == "F" & obs == "L"] <- 5
    if(any(pred == "F" & obs == "M")) cost[pred == "F" & obs == "M"] <- 5
    if(any(pred == "VF" & obs == "M")) cost[pred == "VF" & obs == "M"] <- 5
    out <- mean(cost)
  } else out <- NA
  out
}

### Make a summary function that can be used with caret's train() function
costSummary <- function (data, lev = NULL, model = NULL)
{
  if (is.character(data$obs))  data$obs <- factor(data$obs, levels = lev)
  c(postResample(data[, "pred"], data[, "obs"]),
    Cost = cost(data[, "pred"], data[, "obs"]))
}

### Create a control object for the models
ctrl <- trainControl(method = "repeatedcv", 
                     repeats = 5,
                     summaryFunction = costSummary)

### Optional: parallel processing can be used via the 'do' packages,
### such as doMC, doMPI etc. We used doMC (not on Windows) to speed
### up the computations.

### WARNING: Be aware of how much memory is needed to parallel
### process. It can very quickly overwhelm the available hardware. The
### estimate of the median memory usage (VSIZE = total memory size) 
### was 3300-4100M per core although the some calculations require as  
### much as 3400M without parallel processing. 

library(doMC)
registerDoMC(14)

### Fit the CART model with and without costs

set.seed(857)
rpFit <- train(x = trainData[, predictors],
               y = trainData$Class,
               method = "rpart",
               metric = "Cost",
               maximize = FALSE,
               tuneLength = 20,
               trControl = ctrl)
rpFit

set.seed(857)
rpFitCost <- train(x = trainData[, predictors],
                   y = trainData$Class,
                   method = "rpart",
                   metric = "Cost",
                   maximize = FALSE,
                   tuneLength = 20,
                   parms =list(loss = costMatrix),
                   trControl = ctrl)
rpFitCost

set.seed(857)
ldaFit <- train(x = expandedTrain,
                y = trainData$Class,
                method = "lda",
                metric = "Cost",
                maximize = FALSE,
                trControl = ctrl)
ldaFit

sldaGrid <- expand.grid(NumVars = seq(2, 112, by = 5),
                        lambda = c(0, 0.01, .1, 1, 10))
set.seed(857)
sldaFit <- train(x = expandedTrain,
                 y = trainData$Class,
                 method = "sparseLDA",
                 tuneGrid = sldaGrid,
                 preProc = c("center", "scale"),
                 metric = "Cost",
                 maximize = FALSE,
                 trControl = ctrl)
sldaFit

set.seed(857)
nnetGrid <- expand.grid(decay = c(0, 0.001, 0.01, .1, .5),
                        size = (1:10)*2 - 1)
nnetFit <- train(modForm, 
                 data = trainData,
                 method = "nnet",
                 metric = "Cost",
                 maximize = FALSE,
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 MaxNWts = 2000,
                 maxit = 1000,
                 preProc = c("center", "scale"),
                 trControl = ctrl)
nnetFit

set.seed(857)
plsFit <- train(x = expandedTrain,
                y = trainData$Class,
                method = "pls",
                metric = "Cost",
                maximize = FALSE,
                tuneLength = 100,
                preProc = c("center", "scale"),
                trControl = ctrl)
plsFit

set.seed(857)
fdaFit <- train(modForm, data = trainData,
                method = "fda",
                metric = "Cost",
                maximize = FALSE,
                tuneLength = 25,
                trControl = ctrl)
fdaFit

set.seed(857)
rfFit <- train(x = trainData[, predictors],
               y = trainData$Class,
               method = "rf",
               metric = "Cost",
               maximize = FALSE,
               tuneLength = 10,
               ntree = 2000,
               importance = TRUE,
               trControl = ctrl)
rfFit

set.seed(857)
rfFitCost <- train(x = trainData[, predictors],
                   y = trainData$Class,
                   method = "rf",
                   metric = "Cost",
                   maximize = FALSE,
                   tuneLength = 10,
                   ntree = 2000,
                   classwt = c(VF = 1, F = 1, M = 5, L = 10),
                   importance = TRUE,
                   trControl = ctrl)
rfFitCost

c5Grid <- expand.grid(trials = c(1, (1:10)*10),
                      model = "tree",
                      winnow = c(TRUE, FALSE))
set.seed(857)
c50Fit <- train(x = trainData[, predictors],
                y = trainData$Class,
                method = "C5.0",
                metric = "Cost",
                maximize = FALSE,
                tuneGrid = c5Grid,
                trControl = ctrl)
c50Fit

set.seed(857)
c50Cost <- train(x = trainData[, predictors],
                 y = trainData$Class,
                 method = "C5.0",
                 metric = "Cost",
                 maximize = FALSE,
                 costs = costMatrix,
                 tuneGrid = c5Grid,
                 trControl = ctrl)
c50Cost

set.seed(857)
bagFit <- train(x = trainData[, predictors],
                y = trainData$Class,
                method = "treebag",
                metric = "Cost",
                maximize = FALSE,
                nbagg = 50,
                trControl = ctrl)
bagFit

### Use the caret bag() function to bag the cost-sensitive CART model
rpCost <- function(x, y)
{
  costMatrix <- ifelse(diag(4) == 1, 0, 1)
  costMatrix[4, 1] <- 10
  costMatrix[3, 1] <- 5
  costMatrix[4, 2] <- 5
  costMatrix[3, 2] <- 5
  library(rpart)
  tmp <- x
  tmp$y <- y
  rpart(y~., data = tmp, control = rpart.control(cp = 0),
        parms =list(loss = costMatrix))
}
rpPredict <- function(object, x) predict(object, x)

rpAgg <- function (x, type = "class")
{
  pooled <- x[[1]] * NA
  n <- nrow(pooled)
  classes <- colnames(pooled)
  for (i in 1:ncol(pooled))
  {
    tmp <- lapply(x, function(y, col) y[, col], col = i)
    tmp <- do.call("rbind", tmp)
    pooled[, i] <- apply(tmp, 2, median)
  }
  pooled <- apply(pooled, 1, function(x) x/sum(x))
  if (n != nrow(pooled)) pooled <- t(pooled)
  out <- factor(classes[apply(pooled, 1, which.max)], levels = classes)
  out
}


set.seed(857)
rpCostBag <- train(trainData[, predictors],
                   trainData$Class,
                   "bag",
                   B = 50,
                   bagControl = bagControl(fit = rpCost,
                                           predict = rpPredict,
                                           aggregate = rpAgg,
                                           downSample = FALSE,
                                           allowParallel = FALSE),
                   trControl = ctrl)
rpCostBag

set.seed(857)
svmRFit <- train(modForm ,
                 data = trainData,
                 method = "svmRadial",
                 metric = "Cost",
                 maximize = FALSE,
                 preProc = c("center", "scale"),
                 tuneLength = 15,
                 trControl = ctrl)
svmRFit

set.seed(857)
svmRFitCost <- train(modForm, data = trainData,
                     method = "svmRadial",
                     metric = "Cost",
                     maximize = FALSE,
                     preProc = c("center", "scale"),
                     class.weights = c(VF = 1, F = 1, M = 5, L = 10),
                     tuneLength = 15,
                     trControl = ctrl)
svmRFitCost

modelList <- list(C5.0 = c50Fit,
                  "C5.0 (Costs)" = c50Cost,
                  CART =rpFit,
                  "CART (Costs)" = rpFitCost,
                  "Bagging (Costs)" = rpCostBag,
                  FDA = fdaFit,
                  SVM = svmRFit,
                  "SVM (Weights)" = svmRFitCost,
                  PLS = plsFit,
                  "Random Forests" = rfFit,
                  LDA = ldaFit,
                  "LDA (Sparse)" = sldaFit,
                  "Neural Networks" = nnetFit,
                  Bagging = bagFit)


################################################################################
### Section 17.2 Results

rs <- resamples(modelList)
summary(rs)

confusionMatrix(rpFitCost, "none")
confusionMatrix(rfFit, "none") 

plot(bwplot(rs, metric = "Cost"))

rfPred <- predict(rfFit, testData)
rpPred <- predict(rpFitCost, testData)

confusionMatrix(rfPred, testData$Class)
confusionMatrix(rpPred, testData$Class)


################################################################################
### Session Information

sessionInfo()

q("no")
