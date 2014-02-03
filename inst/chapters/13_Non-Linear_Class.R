################################################################################
### R code from Applied Predictive Modeling (2013) by Kuhn and Johnson.
### Copyright 2013 Kuhn and Johnson
### Web Page: http://www.appliedpredictivemodeling.com
### Contact: Max Kuhn (mxkuhn@gmail.com) 
###
### Chapter 13 Non-Linear Classification Models
###
### Required packages: AppliedPredictiveModeling, caret, doMC (optional) 
###                    kernlab, klaR, lattice, latticeExtra, MASS, mda, nnet,
###                    pROC
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
### Section 13.1 Nonlinear Discriminant Analysis


load("grantData.RData")

library(caret)

### Optional: parallel processing can be used via the 'do' packages,
### such as doMC, doMPI etc. We used doMC (not on Windows) to speed
### up the computations.
 
### WARNING: Be aware of how much memory is needed to parallel
### process. It can very quickly overwhelm the available hardware. We
### estimate the memory usage (VSIZE = total memory size) to be 
### 2700M/core.

library(doMC)
registerDoMC(12)

## This control object will be used across multiple models so that the
## data splitting is consistent

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     index = list(TrainSet = pre2008),
                     savePredictions = TRUE)

set.seed(476)
mdaFit <- train(x = training[,reducedSet], 
                y = training$Class,
                method = "mda",
                metric = "ROC",
                tries = 40,
                tuneGrid = expand.grid(subclasses = 1:8),
                trControl = ctrl)
mdaFit

mdaFit$results <- mdaFit$results[!is.na(mdaFit$results$ROC),]                
mdaFit$pred <- merge(mdaFit$pred,  mdaFit$bestTune)
mdaCM <- confusionMatrix(mdaFit, norm = "none")
mdaCM

mdaRoc <- roc(response = mdaFit$pred$obs,
              predictor = mdaFit$pred$successful,
              levels = rev(levels(mdaFit$pred$obs)))
mdaRoc

update(plot(mdaFit,
            ylab = "ROC AUC (2008 Hold-Out Data)"))

################################################################################
### Section 13.2 Neural Networks

nnetGrid <- expand.grid(size = 1:10, decay = c(0, .1, 1, 2))
maxSize <- max(nnetGrid$size)


## Four different models are evaluate based on the data pre-processing and 
## whethera single or multiple models are used

set.seed(476)
nnetFit <- train(x = training[,reducedSet], 
                 y = training$Class,
                 method = "nnet",
                 metric = "ROC",
                 preProc = c("center", "scale"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 2000,
                 MaxNWts = 1*(maxSize * (length(reducedSet) + 1) + maxSize + 1),
                 trControl = ctrl)
nnetFit

set.seed(476)
nnetFit2 <- train(x = training[,reducedSet], 
                  y = training$Class,
                  method = "nnet",
                  metric = "ROC",
                  preProc = c("center", "scale", "spatialSign"),
                  tuneGrid = nnetGrid,
                  trace = FALSE,
                  maxit = 2000,
                  MaxNWts = 1*(maxSize * (length(reducedSet) + 1) + maxSize + 1),
                  trControl = ctrl)
nnetFit2

nnetGrid$bag <- FALSE

set.seed(476)
nnetFit3 <- train(x = training[,reducedSet], 
                  y = training$Class,
                  method = "avNNet",
                  metric = "ROC",
                  preProc = c("center", "scale"),
                  tuneGrid = nnetGrid,
                  repeats = 10,
                  trace = FALSE,
                  maxit = 2000,
                  MaxNWts = 10*(maxSize * (length(reducedSet) + 1) + maxSize + 1),
                  allowParallel = FALSE, ## this will cause to many workers to be launched.
                  trControl = ctrl)
nnetFit3

set.seed(476)
nnetFit4 <- train(x = training[,reducedSet], 
                  y = training$Class,
                  method = "avNNet",
                  metric = "ROC",
                  preProc = c("center", "scale", "spatialSign"),
                  tuneGrid = nnetGrid,
                  trace = FALSE,
                  maxit = 2000,
                  repeats = 10,
                  MaxNWts = 10*(maxSize * (length(reducedSet) + 1) + maxSize + 1),
                  allowParallel = FALSE, 
                  trControl = ctrl)
nnetFit4

nnetFit4$pred <- merge(nnetFit4$pred,  nnetFit4$bestTune)
nnetCM <- confusionMatrix(nnetFit4, norm = "none")
nnetCM

nnetRoc <- roc(response = nnetFit4$pred$obs,
               predictor = nnetFit4$pred$successful,
               levels = rev(levels(nnetFit4$pred$obs)))


nnet1 <- nnetFit$results
nnet1$Transform <- "No Transformation"
nnet1$Model <- "Single Model"

nnet2 <- nnetFit2$results
nnet2$Transform <- "Spatial Sign"
nnet2$Model <- "Single Model"

nnet3 <- nnetFit3$results
nnet3$Transform <- "No Transformation"
nnet3$Model <- "Model Averaging"
nnet3$bag <- NULL

nnet4 <- nnetFit4$results
nnet4$Transform <- "Spatial Sign"
nnet4$Model <- "Model Averaging"
nnet4$bag <- NULL

nnetResults <- rbind(nnet1, nnet2, nnet3, nnet4)
nnetResults$Model <- factor(as.character(nnetResults$Model),
                            levels = c("Single Model", "Model Averaging"))
library(latticeExtra)
useOuterStrips(
  xyplot(ROC ~ size|Model*Transform,
         data = nnetResults,
         groups = decay,
         as.table = TRUE,
         type = c("p", "l", "g"),
         lty = 1,
         ylab = "ROC AUC (2008 Hold-Out Data)",
         xlab = "Number of Hidden Units",
         auto.key = list(columns = 4, 
                         title = "Weight Decay", 
                         cex.title = 1)))

plot(nnetRoc, type = "s", legacy.axes = TRUE)

################################################################################
### Section 13.3 Flexible Discriminant Analysis

set.seed(476)
fdaFit <- train(x = training[,reducedSet], 
                y = training$Class,
                method = "fda",
                metric = "ROC",
                tuneGrid = expand.grid(degree = 1, nprune = 2:25),
                trControl = ctrl)
fdaFit

fdaFit$pred <- merge(fdaFit$pred,  fdaFit$bestTune)
fdaCM <- confusionMatrix(fdaFit, norm = "none")
fdaCM

fdaRoc <- roc(response = fdaFit$pred$obs,
              predictor = fdaFit$pred$successful,
              levels = rev(levels(fdaFit$pred$obs)))

update(plot(fdaFit), ylab = "ROC AUC (2008 Hold-Out Data)")

plot(nnetRoc, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(fdaRoc, type = "s", add = TRUE, legacy.axes = TRUE)


################################################################################
### Section 13.4 Support Vector Machines

library(kernlab)

set.seed(201)
sigmaRangeFull <- sigest(as.matrix(training[,fullSet]))
svmRGridFull <- expand.grid(sigma =  as.vector(sigmaRangeFull)[1],
                            C = 2^(-3:4))
set.seed(476)
svmRFitFull <- train(x = training[,fullSet], 
                     y = training$Class,
                     method = "svmRadial",
                     metric = "ROC",
                     preProc = c("center", "scale"),
                     tuneGrid = svmRGridFull,
                     trControl = ctrl)
svmRFitFull

set.seed(202)
sigmaRangeReduced <- sigest(as.matrix(training[,reducedSet]))
svmRGridReduced <- expand.grid(sigma = sigmaRangeReduced[1],
                               C = 2^(seq(-4, 4)))
set.seed(476)
svmRFitReduced <- train(x = training[,reducedSet], 
                        y = training$Class,
                        method = "svmRadial",
                        metric = "ROC",
                        preProc = c("center", "scale"),
                        tuneGrid = svmRGridReduced,
                        trControl = ctrl)
svmRFitReduced

svmPGrid <-  expand.grid(degree = 1:2,
                         scale = c(0.01, .005),
                         C = 2^(seq(-6, -2, length = 10)))

set.seed(476)
svmPFitFull <- train(x = training[,fullSet], 
                     y = training$Class,
                     method = "svmPoly",
                     metric = "ROC",
                     preProc = c("center", "scale"),
                     tuneGrid = svmPGrid,
                     trControl = ctrl)
svmPFitFull

svmPGrid2 <-  expand.grid(degree = 1:2,
                          scale = c(0.01, .005),
                          C = 2^(seq(-6, -2, length = 10)))
set.seed(476)
svmPFitReduced <- train(x = training[,reducedSet], 
                        y = training$Class,
                        method = "svmPoly",
                        metric = "ROC",
                        preProc = c("center", "scale"),
                        tuneGrid = svmPGrid2,
                        fit = FALSE,
                        trControl = ctrl)
svmPFitReduced

svmPFitReduced$pred <- merge(svmPFitReduced$pred,  svmPFitReduced$bestTune)
svmPCM <- confusionMatrix(svmPFitReduced, norm = "none")
svmPRoc <- roc(response = svmPFitReduced$pred$obs,
               predictor = svmPFitReduced$pred$successful,
               levels = rev(levels(svmPFitReduced$pred$obs)))


svmRadialResults <- rbind(svmRFitReduced$results,
                          svmRFitFull$results)
svmRadialResults$Set <- c(rep("Reduced Set", nrow(svmRFitReduced$result)),
                          rep("Full Set", nrow(svmRFitFull$result)))
svmRadialResults$Sigma <- paste("sigma = ", 
                                format(svmRadialResults$sigma, 
                                       scientific = FALSE, digits= 5))
svmRadialResults <- svmRadialResults[!is.na(svmRadialResults$ROC),]
xyplot(ROC ~ C|Set, data = svmRadialResults,
       groups = Sigma, type = c("g", "o"),
       xlab = "Cost",
       ylab = "ROC (2008 Hold-Out Data)",
       auto.key = list(columns = 2),
       scales = list(x = list(log = 2)))

svmPolyResults <- rbind(svmPFitReduced$results,
                        svmPFitFull$results)
svmPolyResults$Set <- c(rep("Reduced Set", nrow(svmPFitReduced$result)),
                        rep("Full Set", nrow(svmPFitFull$result)))
svmPolyResults <- svmPolyResults[!is.na(svmPolyResults$ROC),]
svmPolyResults$scale <- paste("scale = ", 
                              format(svmPolyResults$scale, 
                                     scientific = FALSE))
svmPolyResults$Degree <- "Linear"
svmPolyResults$Degree[svmPolyResults$degree == 2] <- "Quadratic"
useOuterStrips(xyplot(ROC ~ C|Degree*Set, data = svmPolyResults,
                      groups = scale, type = c("g", "o"),
                      xlab = "Cost",
                      ylab = "ROC (2008 Hold-Out Data)",
                      auto.key = list(columns = 2),
                      scales = list(x = list(log = 2))))

plot(nnetRoc, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(fdaRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(svmPRoc, type = "s", add = TRUE, legacy.axes = TRUE)

################################################################################
### Section 13.5 K-Nearest Neighbors


set.seed(476)
knnFit <- train(x = training[,reducedSet], 
                y = training$Class,
                method = "knn",
                metric = "ROC",
                preProc = c("center", "scale"),
                tuneGrid = data.frame(k = c(4*(0:5)+1,20*(1:5)+1,50*(2:9)+1)),
                trControl = ctrl)
knnFit

knnFit$pred <- merge(knnFit$pred,  knnFit$bestTune)
knnCM <- confusionMatrix(knnFit, norm = "none")
knnCM
knnRoc <- roc(response = knnFit$pred$obs,
              predictor = knnFit$pred$successful,
              levels = rev(levels(knnFit$pred$obs)))

update(plot(knnFit, ylab = "ROC (2008 Hold-Out Data)"))

plot(fdaRoc, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(nnetRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(svmPRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(knnRoc, type = "s", add = TRUE, legacy.axes = TRUE)

################################################################################
### Section 13.6 Naive Bayes

## Create factor versions of some of the predictors so that they are treated
## as categories and not dummy variables

factors <- c("SponsorCode", "ContractValueBand", "Month", "Weekday")
nbPredictors <- factorPredictors[factorPredictors %in% reducedSet]
nbPredictors <- c(nbPredictors, factors)
nbPredictors <- nbPredictors[nbPredictors != "SponsorUnk"]

nbTraining <- training[, c("Class", nbPredictors)]
nbTesting <- testing[, c("Class", nbPredictors)]

for(i in nbPredictors)
{
  if(length(unique(training[,i])) <= 15)
  {
    nbTraining[, i] <- factor(nbTraining[,i], levels = paste(sort(unique(training[,i]))))
    nbTesting[, i] <- factor(nbTesting[,i], levels = paste(sort(unique(training[,i]))))
  }
}

set.seed(476)
nBayesFit <- train(x = nbTraining[,nbPredictors],
                   y = nbTraining$Class,
                   method = "nb",
                   metric = "ROC",
                   tuneGrid = data.frame(usekernel = c(TRUE, FALSE), fL = 2),
                   trControl = ctrl)
nBayesFit

nBayesFit$pred <- merge(nBayesFit$pred,  nBayesFit$bestTune)
nBayesCM <- confusionMatrix(nBayesFit, norm = "none")
nBayesCM
nBayesRoc <- roc(response = nBayesFit$pred$obs,
                 predictor = nBayesFit$pred$successful,
                 levels = rev(levels(nBayesFit$pred$obs)))
nBayesRoc


sessionInfo()

q("no")
