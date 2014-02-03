################################################################################
### R code from Applied Predictive Modeling (2013) by Kuhn and Johnson.
### Copyright 2013 Kuhn and Johnson
### Web Page: http://www.appliedpredictivemodeling.com
### Contact: Max Kuhn (mxkuhn@gmail.com)
###
### Chapter 10: Case Study: Compressive Strength of Concrete Mixtures
###
### Required packages: AppliedPredictiveModeling, caret, Cubist, doMC (optional),
###                    earth, elasticnet, gbm, ipred, lattice, nnet, party, pls,
###                    randomForests, rpart, RWeka      
###
### Data used: The concrete from the AppliedPredictiveModeling package
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

################################################################################
### Load the data and plot the data

library(AppliedPredictiveModeling)
data(concrete)

library(caret)
library(plyr)

featurePlot(concrete[, -9], concrete$CompressiveStrength,
            between = list(x = 1, y = 1),
            type = c("g", "p", "smooth"))


################################################################################
### Section 10.1 Model Building Strategy
### There are replicated mixtures, so take the average per mixture
            
averaged <- ddply(mixtures,
                  .(Cement, BlastFurnaceSlag, FlyAsh, Water, 
                    Superplasticizer, CoarseAggregate, 
                    FineAggregate, Age),
                  function(x) c(CompressiveStrength = 
                    mean(x$CompressiveStrength)))

### Split the data and create a control object for train()

set.seed(975)
inTrain <- createDataPartition(averaged$CompressiveStrength, p = 3/4)[[1]]
training <- averaged[ inTrain,]
testing  <- averaged[-inTrain,]

ctrl <- trainControl(method = "repeatedcv", repeats = 5, number = 10)

### Create a model formula that can be used repeatedly

modForm <- paste("CompressiveStrength ~ (.)^2 + I(Cement^2) + I(BlastFurnaceSlag^2) +",
                 "I(FlyAsh^2)  + I(Water^2) + I(Superplasticizer^2)  +",
                 "I(CoarseAggregate^2) +  I(FineAggregate^2) + I(Age^2)")
modForm <- as.formula(modForm)

### Fit the various models

### Optional: parallel processing can be used via the 'do' packages,
### such as doMC, doMPI etc. We used doMC (not on Windows) to speed
### up the computations.

### WARNING: Be aware of how much memory is needed to parallel
### process. It can very quickly overwhelm the available hardware. The
### estimate of the median memory usage (VSIZE = total memory size) 
### was 2800M for a core although the M5 calculations require about 
### 3700M without parallel processing. 

### WARNING 2: The RWeka package does not work well with some forms of
### parallel processing, such as mutlicore (i.e. doMC). 

library(doMC)
registerDoMC(14)

set.seed(669)
lmFit <- train(modForm, data = training,
               method = "lm",
               trControl = ctrl)

set.seed(669)
plsFit <- train(modForm, data = training,
                method = "pls",
                preProc = c("center", "scale"),
                tuneLength = 15,
                trControl = ctrl)

lassoGrid <- expand.grid(lambda = c(0, .001, .01, .1), 
                         fraction = seq(0.05, 1, length = 20))
set.seed(669)
lassoFit <- train(modForm, data = training,
                  method = "enet",
                  preProc = c("center", "scale"),
                  tuneGrid = lassoGrid,
                  trControl = ctrl)

set.seed(669)
earthFit <- train(CompressiveStrength ~ ., data = training,
                  method = "earth",
                  tuneGrid = expand.grid(degree = 1, 
                                         nprune = 2:25),
                  trControl = ctrl)

set.seed(669)
svmRFit <- train(CompressiveStrength ~ ., data = training,
                 method = "svmRadial",
                 tuneLength = 15,
                 preProc = c("center", "scale"),
                 trControl = ctrl)


nnetGrid <- expand.grid(decay = c(0.001, .01, .1), 
                        size = seq(1, 27, by = 2), 
                        bag = FALSE)
set.seed(669)
nnetFit <- train(CompressiveStrength ~ .,
                 data = training,
                 method = "avNNet",
                 tuneGrid = nnetGrid,
                 preProc = c("center", "scale"),
                 linout = TRUE,
                 trace = FALSE,
                 maxit = 1000,
                 allowParallel = FALSE,
                 trControl = ctrl)

set.seed(669)
rpartFit <- train(CompressiveStrength ~ .,
                  data = training,
                  method = "rpart",
                  tuneLength = 30,
                  trControl = ctrl)

set.seed(669)
treebagFit <- train(CompressiveStrength ~ .,
                    data = training,
                    method = "treebag",
                    trControl = ctrl)

set.seed(669)
ctreeFit <- train(CompressiveStrength ~ .,
                  data = training,
                  method = "ctree",
                  tuneLength = 10,
                  trControl = ctrl)

set.seed(669)
rfFit <- train(CompressiveStrength ~ .,
               data = training,
               method = "rf",
               tuneLength = 10,
               ntrees = 1000,
               importance = TRUE,
               trControl = ctrl)


gbmGrid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                       n.trees = seq(100, 1000, by = 50),
                       shrinkage = c(0.01, 0.1))
set.seed(669)
gbmFit <- train(CompressiveStrength ~ .,
                data = training,
                method = "gbm",
                tuneGrid = gbmGrid,
                verbose = FALSE,
                trControl = ctrl)


cbGrid <- expand.grid(committees = c(1, 5, 10, 50, 75, 100), 
                      neighbors = c(0, 1, 3, 5, 7, 9))
set.seed(669)
cbFit <- train(CompressiveStrength ~ .,
               data = training,
               method = "cubist",
               tuneGrid = cbGrid,
               trControl = ctrl)

### Turn off the parallel processing to use RWeka. 
registerDoSEQ()


set.seed(669)
mtFit <- train(CompressiveStrength ~ .,
               data = training,
               method = "M5",
               trControl = ctrl)

################################################################################
### Section 10.2 Model Performance

### Collect the resampling statistics across all the models

rs <- resamples(list("Linear Reg" = lmFit, "
                     PLS" = plsFit,
                     "Elastic Net" = lassoFit, 
                     MARS = earthFit,
                     SVM = svmRFit, 
                     "Neural Networks" = nnetFit,
                     CART = rpartFit, 
                     "Cond Inf Tree" = ctreeFit,
                     "Bagged Tree" = treebagFit,
                     "Boosted Tree" = gbmFit,
                     "Random Forest" = rfFit,
                     Cubist = cbFit))

#parallelPlot(rs)
#parallelPlot(rs, metric = "Rsquared")

### Get the test set results across several models

nnetPred <- predict(nnetFit, testing)
gbmPred <- predict(gbmFit, testing)
cbPred <- predict(cbFit, testing)

testResults <- rbind(postResample(nnetPred, testing$CompressiveStrength),
                     postResample(gbmPred, testing$CompressiveStrength),
                     postResample(cbPred, testing$CompressiveStrength))
testResults <- as.data.frame(testResults)
testResults$Model <- c("Neural Networks", "Boosted Tree", "Cubist")
testResults <- testResults[order(testResults$RMSE),]

################################################################################
### Section 10.3 Optimizing Compressive Strength

library(proxy)

### Create a function to maximize compressive strength* while keeping
### the predictor values as mixtures. Water (in x[7]) is used as the 
### 'slack variable'. 

### * We are actually minimizing the negative compressive strength

modelPrediction <- function(x, mod, limit = 2500)
{
  if(x[1] < 0 | x[1] > 1) return(10^38)
  if(x[2] < 0 | x[2] > 1) return(10^38)
  if(x[3] < 0 | x[3] > 1) return(10^38)
  if(x[4] < 0 | x[4] > 1) return(10^38)
  if(x[5] < 0 | x[5] > 1) return(10^38)
  if(x[6] < 0 | x[6] > 1) return(10^38)
  
  x <- c(x, 1 - sum(x))
  
  if(x[7] < 0.05) return(10^38)
  
  tmp <- as.data.frame(t(x))
  names(tmp) <- c('Cement','BlastFurnaceSlag','FlyAsh',
                  'Superplasticizer','CoarseAggregate',
                  'FineAggregate', 'Water')
  tmp$Age <- 28
  -predict(mod, tmp)
}

### Get mixtures at 28 days 
subTrain <- subset(training, Age == 28)

### Center and scale the data to use dissimilarity sampling
pp1 <- preProcess(subTrain[, -(8:9)], c("center", "scale"))
scaledTrain <- predict(pp1, subTrain[, 1:7])

### Randomly select a few mixtures as a starting pool

set.seed(91)
startMixture <- sample(1:nrow(subTrain), 1)
starters <- scaledTrain[startMixture, 1:7]
pool <- scaledTrain
index <- maxDissim(starters, pool, 14)
startPoints <- c(startMixture, index)

starters <- subTrain[startPoints,1:7]
startingValues <- starters[, -4]

### For each starting mixture, optimize the Cubist model using
### a simplex search routine

cbResults <- startingValues
cbResults$Water <- NA
cbResults$Prediction <- NA

for(i in 1:nrow(cbResults))
{
  results <- optim(unlist(cbResults[i,1:6]),
                   modelPrediction,
                   method = "Nelder-Mead",
                   control=list(maxit=5000),
                   mod = cbFit)
  cbResults$Prediction[i] <- -results$value
  cbResults[i,1:6] <- results$par
}
cbResults$Water <- 1 - apply(cbResults[,1:6], 1, sum)
cbResults <- subset(cbResults, Prediction > 0 & Water > .02)
cbResults <- cbResults[order(-cbResults$Prediction),][1:3,]
cbResults$Model <- "Cubist"

### Do the same for the neural network model

nnetResults <- startingValues
nnetResults$Water <- NA
nnetResults$Prediction <- NA

for(i in 1:nrow(nnetResults))
{
  results <- optim(unlist(nnetResults[i, 1:6,]),
                   modelPrediction,
                   method = "Nelder-Mead",
                   control=list(maxit=5000),
                   mod = nnetFit)
  nnetResults$Prediction[i] <- -results$value
  nnetResults[i,1:6] <- results$par
}
nnetResults$Water <- 1 - apply(nnetResults[,1:6], 1, sum)
nnetResults <- subset(nnetResults, Prediction > 0 & Water > .02)
nnetResults <- nnetResults[order(-nnetResults$Prediction),][1:3,]
nnetResults$Model <- "NNet"

### Convert the predicted mixtures to PCA space and plot

pp2 <- preProcess(subTrain[, 1:7], "pca")
pca1 <- predict(pp2, subTrain[, 1:7])
pca1$Data <- "Training Set"
pca1$Data[startPoints] <- "Starting Values"
pca3 <- predict(pp2, cbResults[, names(subTrain[, 1:7])])
pca3$Data <- "Cubist"
pca4 <- predict(pp2, nnetResults[, names(subTrain[, 1:7])])
pca4$Data <- "Neural Network"

pcaData <- rbind(pca1, pca3, pca4)
pcaData$Data <- factor(pcaData$Data,
                       levels = c("Training Set","Starting Values",
                                  "Cubist","Neural Network"))

lim <- extendrange(pcaData[, 1:2])

xyplot(PC2 ~ PC1, 
       data = pcaData, 
       groups = Data,
       auto.key = list(columns = 2),
       xlim = lim, 
       ylim = lim,
       type = c("g", "p"))


################################################################################
### Session Information

sessionInfo()

q("no")
