################################################################################
### R code from Applied Predictive Modeling (2013) by Kuhn and Johnson.
### Copyright 2013 Kuhn and Johnson
### Web Page: http://www.appliedpredictivemodeling.com
### Contact: Max Kuhn (mxkuhn@gmail.com) 
###
### Chapter 12 Discriminant Analysis and Other Linear Classification Models
###
### Required packages: AppliedPredictiveModeling, caret, doMC (optional),  
###                    glmnet, lattice, MASS, pamr, pls, pROC, sparseLDA
###
### Data used: The grant application data. See the file 'CreateGrantData.R'
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
### Section 12.1 Case Study: Predicting Successful Grant Applications

load("grantData.RData")

library(caret)
library(doMC)
registerDoMC(12)
library(plyr)
library(reshape2)

## Look at two different ways to split and resample the data. A support vector
## machine is used to illustrate the differences. The full set of predictors
## is used. 

pre2008Data <- training[pre2008,]
year2008Data <- rbind(training[-pre2008,], testing)

set.seed(552)
test2008 <- createDataPartition(year2008Data$Class, p = .25)[[1]]

allData <- rbind(pre2008Data, year2008Data[-test2008,])
holdout2008 <- year2008Data[test2008,]

## Use a common tuning grid for both approaches. 
svmrGrid <- expand.grid(sigma = c(.00007, .00009, .0001, .0002),
                        C = 2^(-3:8))

## Evaluate the model using overall 10-fold cross-validation
ctrl0 <- trainControl(method = "cv",
                      summaryFunction = twoClassSummary,
                      classProbs = TRUE)
set.seed(477)
svmFit0 <- train(pre2008Data[,fullSet], pre2008Data$Class,
                 method = "svmRadial",
                 tuneGrid = svmrGrid,
                 preProc = c("center", "scale"),
                 metric = "ROC",
                 trControl = ctrl0)
svmFit0

### Now fit the single 2008 test set
ctrl00 <- trainControl(method = "LGOCV",
                       summaryFunction = twoClassSummary,
                       classProbs = TRUE,
                       index = list(TestSet = 1:nrow(pre2008Data)))


set.seed(476)
svmFit00 <- train(allData[,fullSet], allData$Class,
                  method = "svmRadial",
                  tuneGrid = svmrGrid,
                  preProc = c("center", "scale"),
                  metric = "ROC",
                  trControl = ctrl00)
svmFit00

## Combine the two sets of results and plot

grid0 <- subset(svmFit0$results,  sigma == svmFit0$bestTune$sigma)
grid0$Model <- "10-Fold Cross-Validation"

grid00 <- subset(svmFit00$results,  sigma == svmFit00$bestTune$sigma)
grid00$Model <- "Single 2008 Test Set"

plotData <- rbind(grid00, grid0)

plotData <- plotData[!is.na(plotData$ROC),]
xyplot(ROC ~ C, data = plotData,
       groups = Model,
       type = c("g", "o"),
       scales = list(x = list(log = 2)),
       auto.key = list(columns = 1))

################################################################################
### Section 12.2 Logistic Regression

modelFit <- glm(Class ~ Day, data = training[pre2008,], family = binomial)
dataGrid <- data.frame(Day = seq(0, 365, length = 500))
dataGrid$Linear <- 1 - predict(modelFit, dataGrid, type = "response")
linear2008 <- auc(roc(response = training[-pre2008, "Class"],
                      predictor = 1 - predict(modelFit, 
                                              training[-pre2008,], 
                                              type = "response"),
                      levels = rev(levels(training[-pre2008, "Class"]))))


modelFit2 <- glm(Class ~ Day + I(Day^2), 
                 data = training[pre2008,], 
                 family = binomial)
dataGrid$Quadratic <- 1 - predict(modelFit2, dataGrid, type = "response")
quad2008 <- auc(roc(response = training[-pre2008, "Class"],
                    predictor = 1 - predict(modelFit2, 
                                            training[-pre2008,], 
                                            type = "response"),
                    levels = rev(levels(training[-pre2008, "Class"]))))

dataGrid <- melt(dataGrid, id.vars = "Day")

byDay <- training[pre2008, c("Day", "Class")]
byDay$Binned <- cut(byDay$Day, seq(0, 360, by = 5))

observedProps <- ddply(byDay, .(Binned),
                       function(x) c(n = nrow(x), mean = mean(x$Class == "successful")))
observedProps$midpoint <- seq(2.5, 357.5, by = 5)

xyplot(value ~ Day|variable, data = dataGrid,
       ylab = "Probability of A Successful Grant",
       ylim = extendrange(0:1),
       between = list(x = 1),
       panel = function(...)
       {
         panel.xyplot(x = observedProps$midpoint, observedProps$mean,
                      pch = 16., col = rgb(.2, .2, .2, .5))
         panel.xyplot(..., type = "l", col = "black", lwd = 2)
       })

## For the reduced set of factors, fit the logistic regression model (linear and
## quadratic) and evaluate on the 
training$Day2 <- training$Day^2
testing$Day2 <- testing$Day^2
fullSet <- c(fullSet, "Day2")
reducedSet <- c(reducedSet, "Day2")

## This control object will be used across multiple models so that the
## data splitting is consistent

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     index = list(TrainSet = pre2008),
                     savePredictions = TRUE)

set.seed(476)
lrFit <- train(x = training[,reducedSet], 
               y = training$Class,
               method = "glm",
               metric = "ROC",
               trControl = ctrl)
lrFit
set.seed(476)
lrFit2 <- train(x = training[,c(fullSet, "Day2")], 
                y = training$Class,
                method = "glm",
                metric = "ROC",
                trControl = ctrl)
lrFit2

lrFit$pred <- merge(lrFit$pred,  lrFit$bestTune)

## Get the confusion matrices for the hold-out set
lrCM <- confusionMatrix(lrFit, norm = "none")
lrCM
lrCM2 <- confusionMatrix(lrFit2, norm = "none")
lrCM2

## Get the area under the ROC curve for the hold-out set
lrRoc <- roc(response = lrFit$pred$obs,
             predictor = lrFit$pred$successful,
             levels = rev(levels(lrFit$pred$obs)))
lrRoc2 <- roc(response = lrFit2$pred$obs,
              predictor = lrFit2$pred$successful,
              levels = rev(levels(lrFit2$pred$obs)))
lrImp <- varImp(lrFit, scale = FALSE)

plot(lrRoc, legacy.axes = TRUE)

################################################################################
### Section 12.3 Linear Discriminant Analysis

## Fit the model to the reduced set
set.seed(476)
ldaFit <- train(x = training[,reducedSet], 
                y = training$Class,
                method = "lda",
                preProc = c("center","scale"),
                metric = "ROC",
                trControl = ctrl)
ldaFit

ldaFit$pred <- merge(ldaFit$pred,  ldaFit$bestTune)
ldaCM <- confusionMatrix(ldaFit, norm = "none")
ldaCM
ldaRoc <- roc(response = ldaFit$pred$obs,
              predictor = ldaFit$pred$successful,
              levels = rev(levels(ldaFit$pred$obs)))
plot(lrRoc, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(ldaRoc, add = TRUE, type = "s", legacy.axes = TRUE)

################################################################################
### Section 12.4 Partial Least Squares Discriminant Analysis

## This model uses all of the predictors
set.seed(476)
plsFit <- train(x = training[,fullSet], 
                y = training$Class,
                method = "pls",
                tuneGrid = expand.grid(ncomp = 1:10),
                preProc = c("center","scale"),
                metric = "ROC",
                probMethod = "Bayes",
                trControl = ctrl)
plsFit

plsImpGrant <- varImp(plsFit, scale = FALSE)

bestPlsNcomp <- plsFit$results[best(plsFit$results, "ROC", maximize = TRUE), "ncomp"]
bestPlsROC <- plsFit$results[best(plsFit$results, "ROC", maximize = TRUE), "ROC"]

## Only keep the final tuning parameter data
plsFit$pred <- merge(plsFit$pred,  plsFit$bestTune)

plsRoc <- roc(response = plsFit$pred$obs,
              predictor = plsFit$pred$successful,
              levels = rev(levels(plsFit$pred$obs)))

### PLS confusion matrix information
plsCM <- confusionMatrix(plsFit, norm = "none")
plsCM

## Now fit a model that uses a smaller set of predictors chosen by unsupervised 
## filtering. 

set.seed(476)
plsFit2 <- train(x = training[,reducedSet], 
                 y = training$Class,
                 method = "pls",
                 tuneGrid = expand.grid(ncomp = 1:10),
                 preProc = c("center","scale"),
                 metric = "ROC",
                 probMethod = "Bayes",
                 trControl = ctrl)
plsFit2

bestPlsNcomp2 <- plsFit2$results[best(plsFit2$results, "ROC", maximize = TRUE), "ncomp"]
bestPlsROC2 <- plsFit2$results[best(plsFit2$results, "ROC", maximize = TRUE), "ROC"]

plsFit2$pred <- merge(plsFit2$pred,  plsFit2$bestTune)

plsRoc2 <- roc(response = plsFit2$pred$obs,
               predictor = plsFit2$pred$successful,
               levels = rev(levels(plsFit2$pred$obs)))
plsCM2 <- confusionMatrix(plsFit2, norm = "none")
plsCM2

pls.ROC <- cbind(plsFit$results,Descriptors="Full Set")
pls2.ROC <- cbind(plsFit2$results,Descriptors="Reduced Set")

plsCompareROC <- data.frame(rbind(pls.ROC,pls2.ROC))

xyplot(ROC ~ ncomp,
       data = plsCompareROC,
       xlab = "# Components",
       ylab = "ROC (2008 Hold-Out Data)",
       auto.key = list(columns = 2),
       groups = Descriptors,
       type = c("o", "g"))

## Plot ROC curves and variable importance scores
plot(ldaRoc, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(lrRoc, type = "s", col = rgb(.2, .2, .2, .2), add = TRUE, legacy.axes = TRUE)
plot(plsRoc2, type = "s", add = TRUE, legacy.axes = TRUE)

plot(plsImpGrant, top=20, scales = list(y = list(cex = .95)))

################################################################################
### Section 12.5 Penalized Models

## The glmnet model
glmnGrid <- expand.grid(alpha = c(0,  .1,  .2, .4, .6, .8, 1),
                        lambda = seq(.01, .2, length = 40))
set.seed(476)
glmnFit <- train(x = training[,fullSet], 
                 y = training$Class,
                 method = "glmnet",
                 tuneGrid = glmnGrid,
                 preProc = c("center", "scale"),
                 metric = "ROC",
                 trControl = ctrl)
glmnFit

glmnet2008 <- merge(glmnFit$pred,  glmnFit$bestTune)
glmnetCM <- confusionMatrix(glmnFit, norm = "none")
glmnetCM

glmnetRoc <- roc(response = glmnet2008$obs,
                 predictor = glmnet2008$successful,
                 levels = rev(levels(glmnet2008$obs)))

glmnFit0 <- glmnFit
glmnFit0$results$lambda <- format(round(glmnFit0$results$lambda, 3))

glmnPlot <- plot(glmnFit0,
                 plotType = "level",
                 cuts = 15,
                 scales = list(x = list(rot = 90, cex = .65)))

update(glmnPlot,
       ylab = "Mixing Percentage\nRidge <---------> Lasso",
       sub = "",
       main = "Area Under the ROC Curve",
       xlab = "Amount of Regularization")

plot(plsRoc2, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(ldaRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(lrRoc, type = "s", col = rgb(.2, .2, .2, .2), add = TRUE, legacy.axes = TRUE)
plot(glmnetRoc, type = "s", add = TRUE, legacy.axes = TRUE)

## Sparse logistic regression

set.seed(476)
spLDAFit <- train(x = training[,fullSet], 
                  y = training$Class,
                  "sparseLDA",
                  tuneGrid = expand.grid(lambda = c(.1),
                                         NumVars = c(1:20, 50, 75, 100, 250, 500, 750, 1000)),
                  preProc = c("center", "scale"),
                  metric = "ROC",
                  trControl = ctrl)
spLDAFit

spLDA2008 <- merge(spLDAFit$pred,  spLDAFit$bestTune)
spLDACM <- confusionMatrix(spLDAFit, norm = "none")
spLDACM

spLDARoc <- roc(response = spLDA2008$obs,
                predictor = spLDA2008$successful,
                levels = rev(levels(spLDA2008$obs)))

update(plot(spLDAFit, scales = list(x = list(log = 10))),
       ylab = "ROC AUC (2008 Hold-Out Data)")

plot(plsRoc2, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(glmnetRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(ldaRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(lrRoc, type = "s", col = rgb(.2, .2, .2, .2), add = TRUE, legacy.axes = TRUE)
plot(spLDARoc, type = "s", add = TRUE, legacy.axes = TRUE)

################################################################################
### Section 12.6 Nearest Shrunken Centroids

set.seed(476)
nscFit <- train(x = training[,fullSet], 
                y = training$Class,
                method = "pam",
                preProc = c("center", "scale"),
                tuneGrid = data.frame(threshold = seq(0, 25, length = 30)),
                metric = "ROC",
                trControl = ctrl)
nscFit

nsc2008 <- merge(nscFit$pred,  nscFit$bestTune)
nscCM <- confusionMatrix(nscFit, norm = "none")
nscCM
nscRoc <- roc(response = nsc2008$obs,
              predictor = nsc2008$successful,
              levels = rev(levels(nsc2008$obs)))
update(plot(nscFit), ylab = "ROC AUC (2008 Hold-Out Data)")


plot(plsRoc2, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(glmnetRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(ldaRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(lrRoc, type = "s", col = rgb(.2, .2, .2, .2), add = TRUE, legacy.axes = TRUE)
plot(spLDARoc, type = "s", col = rgb(.2, .2, .2, .2), add = TRUE, legacy.axes = TRUE)
plot(nscRoc, type = "s", add = TRUE, legacy.axes = TRUE)

sessionInfo()

q("no")

