################################################################################
### R code from Applied Predictive Modeling (2013) by Kuhn and Johnson.
### Copyright 2013 Kuhn and Johnson
### Web Page: http://www.appliedpredictivemodeling.com
### Contact: Max Kuhn (mxkuhn@gmail.com) 
###
### Chapter 8: Regression Trees and Rule-Based Models 
###
### Required packages: AppliedPredictiveModeling, caret, Cubis, doMC (optional),
###                    gbm, lattice, party, partykit, randomForest, rpart, RWeka
###
### Data used: The solubility from the AppliedPredictiveModeling package
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
### Load the data

library(AppliedPredictiveModeling)
data(solubility)

### Create a control function that will be used across models. We
### create the fold assignments explicitly instead of relying on the
### random number seed being set to identical values.

library(caret)
set.seed(100)
indx <- createFolds(solTrainY, returnTrain = TRUE)
ctrl <- trainControl(method = "cv", index = indx)

################################################################################
### Section 8.1 Basic Regression Trees

library(rpart)

### Fit two CART models to show the initial splitting process. rpart 
### only uses formulas, so we put the predictors and outcome into
### a common data frame first.

trainData <- solTrainXtrans
trainData$y <- solTrainY

rpStump <- rpart(y ~ ., data = trainData, 
                 control = rpart.control(maxdepth = 1))
rpSmall <- rpart(y ~ ., data = trainData, 
                 control = rpart.control(maxdepth = 2))

### Tune the model
library(caret)

set.seed(100)
cartTune <- train(x = solTrainXtrans, y = solTrainY,
                  method = "rpart",
                  tuneLength = 25,
                  trControl = ctrl)
cartTune
## cartTune$finalModel


### Plot the tuning results
plot(cartTune, scales = list(x = list(log = 10)))

### Use the partykit package to make some nice plots. First, convert
### the rpart objects to party objects.

# library(partykit)
# 
# cartTree <- as.party(cartTune$finalModel)
# plot(cartTree)

### Get the variable importance. 'competes' is an argument that
### controls whether splits not used in the tree should be included
### in the importance calculations.

cartImp <- varImp(cartTune, scale = FALSE, competes = FALSE)
cartImp

### Save the test set results in a data frame                 
testResults <- data.frame(obs = solTestY,
                          CART = predict(cartTune, solTestXtrans))

### Tune the conditional inference tree

cGrid <- data.frame(mincriterion = sort(c(.95, seq(.75, .99, length = 2))))

set.seed(100)
ctreeTune <- train(x = solTrainXtrans, y = solTrainY,
                   method = "ctree",
                   tuneGrid = cGrid,
                   trControl = ctrl)
ctreeTune
plot(ctreeTune)

##ctreeTune$finalModel               
plot(ctreeTune$finalModel)

testResults$cTree <- predict(ctreeTune, solTestXtrans)

################################################################################
### Section 8.2 Regression Model Trees and 8.3 Rule-Based Models

### Tune the model tree. Using method = "M5" actually tunes over the
### tree- and rule-based versions of the model. M = 10 is also passed
### in to make sure that there are larger terminal nodes for the
### regression models.

set.seed(100)
m5Tune <- train(x = solTrainXtrans, y = solTrainY,
                method = "M5",
                trControl = ctrl,
                control = Weka_control(M = 10))
m5Tune

plot(m5Tune)

## m5Tune$finalModel

## plot(m5Tune$finalModel)

### Show the rule-based model too

ruleFit <- M5Rules(y~., data = trainData, control = Weka_control(M = 10))
ruleFit

################################################################################
### Section 8.4 Bagged Trees

### Optional: parallel processing can be used via the 'do' packages,
### such as doMC, doMPI etc. We used doMC (not on Windows) to speed
### up the computations.

### WARNING: Be aware of how much memory is needed to parallel
### process. It can very quickly overwhelm the available hardware. The
### estimate of the median memory usage (VSIZE = total memory size) 
### was 9706M for a core, but could range up to 9706M. This becomes 
### severe when parallelizing randomForest() and (especially) calls 
### to cforest(). 

### WARNING 2: The RWeka package does not work well with some forms of
### parallel processing, such as mutlicore (i.e. doMC).

library(doMC)
registerDoMC(5)

set.seed(100)

treebagTune <- train(x = solTrainXtrans, y = solTrainY,
                     method = "treebag",
                     nbagg = 50,
                     trControl = ctrl)

treebagTune

################################################################################
### Section 8.5 Random Forests

mtryGrid <- data.frame(mtry = floor(seq(10, ncol(solTrainXtrans), length = 10)))


### Tune the model using cross-validation
set.seed(100)
rfTune <- train(x = solTrainXtrans, y = solTrainY,
                method = "rf",
                tuneGrid = mtryGrid,
                ntree = 1000,
                importance = TRUE,
                trControl = ctrl)
rfTune

plot(rfTune)

rfImp <- varImp(rfTune, scale = FALSE)
rfImp

### Tune the model using the OOB estimates
ctrlOOB <- trainControl(method = "oob")
set.seed(100)
rfTuneOOB <- train(x = solTrainXtrans, y = solTrainY,
                   method = "rf",
                   tuneGrid = mtryGrid,
                   ntree = 1000,
                   importance = TRUE,
                   trControl = ctrlOOB)
rfTuneOOB

plot(rfTuneOOB)

### Tune the conditional inference forests
set.seed(100)
condrfTune <- train(x = solTrainXtrans, y = solTrainY,
                    method = "cforest",
                    tuneGrid = mtryGrid,
                    controls = cforest_unbiased(ntree = 1000),
                    trControl = ctrl)
condrfTune

plot(condrfTune)

set.seed(100)
condrfTuneOOB <- train(x = solTrainXtrans, y = solTrainY,
                       method = "cforest",
                       tuneGrid = mtryGrid,
                       controls = cforest_unbiased(ntree = 1000),
                       trControl = trainControl(method = "oob"))
condrfTuneOOB

plot(condrfTuneOOB)

################################################################################
### Section 8.6 Boosting

gbmGrid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                       n.trees = seq(100, 1000, by = 50),
                       shrinkage = c(0.01, 0.1))
set.seed(100)
gbmTune <- train(x = solTrainXtrans, y = solTrainY,
                 method = "gbm",
                 tuneGrid = gbmGrid,
                 trControl = ctrl,
                 verbose = FALSE)
gbmTune

plot(gbmTune, auto.key = list(columns = 4, lines = TRUE))

gbmImp <- varImp(gbmTune, scale = FALSE)
gbmImp

################################################################################
### Section 8.7 Cubist

cbGrid <- expand.grid(committees = c(1:10, 20, 50, 75, 100), 
                      neighbors = c(0, 1, 5, 9))

set.seed(100)
cubistTune <- train(solTrainXtrans, solTrainY,
                    "cubist",
                    tuneGrid = cbGrid,
                    trControl = ctrl)
cubistTune

plot(cubistTune, auto.key = list(columns = 4, lines = TRUE))

cbImp <- varImp(cubistTune, scale = FALSE)
cbImp

################################################################################
### Session Information

sessionInfo()

q("no")
