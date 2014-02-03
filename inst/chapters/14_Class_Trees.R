################################################################################
### R code from Applied Predictive Modeling (2013) by Kuhn and Johnson.
### Copyright 2013 Kuhn and Johnson
### Web Page: http://www.appliedpredictivemodeling.com
### Contact: Max Kuhn (mxkuhn@gmail.com) 
###
### Chapter 14 Classification Trees and Rule Based Models
###
### Required packages: AppliedPredictiveModeling, C50, caret, doMC (optional),
###                    gbm, lattice, partykit, pROC, randomForest, reshape2,
###                    rpart, RWeka
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

### NOTE: Many of the models here are computationally expensive. If
### this script is run as-is, the memory requirements will accumulate
### until it exceeds 32gb. 

################################################################################
### Section 14.1 Basic Classification Trees

library(caret)

load("grantData.RData")

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     index = list(TrainSet = pre2008),
                     savePredictions = TRUE)

set.seed(476)
rpartFit <- train(x = training[,fullSet], 
                  y = training$Class,
                  method = "rpart",
                  tuneLength = 30,
                  metric = "ROC",
                  trControl = ctrl)
rpartFit

library(partykit)
plot(as.party(rpartFit$finalModel))

rpart2008 <- merge(rpartFit$pred,  rpartFit$bestTune)
rpartCM <- confusionMatrix(rpartFit, norm = "none")
rpartCM
rpartRoc <- roc(response = rpartFit$pred$obs,
                predictor = rpartFit$pred$successful,
                levels = rev(levels(rpartFit$pred$obs)))

set.seed(476)
rpartFactorFit <- train(x = training[,factorPredictors], 
                        y = training$Class,
                        method = "rpart",
                        tuneLength = 30,
                        metric = "ROC",
                        trControl = ctrl)
rpartFactorFit 
plot(as.party(rpartFactorFit$finalModel))

rpartFactor2008 <- merge(rpartFactorFit$pred,  rpartFactorFit$bestTune)
rpartFactorCM <- confusionMatrix(rpartFactorFit, norm = "none")
rpartFactorCM

rpartFactorRoc <- roc(response = rpartFactorFit$pred$obs,
                      predictor = rpartFactorFit$pred$successful,
                      levels = rev(levels(rpartFactorFit$pred$obs)))

plot(rpartRoc, type = "s", print.thres = c(.5),
     print.thres.pch = 3,
     print.thres.pattern = "",
     print.thres.cex = 1.2,
     col = "red", legacy.axes = TRUE,
     print.thres.col = "red")
plot(rpartFactorRoc,
     type = "s",
     add = TRUE,
     print.thres = c(.5),
     print.thres.pch = 16, legacy.axes = TRUE,
     print.thres.pattern = "",
     print.thres.cex = 1.2)
legend(.75, .2,
       c("Grouped Categories", "Independent Categories"),
       lwd = c(1, 1),
       col = c("black", "red"),
       pch = c(16, 3))

set.seed(476)
j48FactorFit <- train(x = training[,factorPredictors], 
                      y = training$Class,
                      method = "J48",
                      metric = "ROC",
                      trControl = ctrl)
j48FactorFit

j48Factor2008 <- merge(j48FactorFit$pred,  j48FactorFit$bestTune)
j48FactorCM <- confusionMatrix(j48FactorFit, norm = "none")
j48FactorCM

j48FactorRoc <- roc(response = j48FactorFit$pred$obs,
                    predictor = j48FactorFit$pred$successful,
                    levels = rev(levels(j48FactorFit$pred$obs)))

set.seed(476)
j48Fit <- train(x = training[,fullSet], 
                y = training$Class,
                method = "J48",
                metric = "ROC",
                trControl = ctrl)

j482008 <- merge(j48Fit$pred,  j48Fit$bestTune)
j48CM <- confusionMatrix(j48Fit, norm = "none")
j48CM

j48Roc <- roc(response = j48Fit$pred$obs,
              predictor = j48Fit$pred$successful,
              levels = rev(levels(j48Fit$pred$obs)))


plot(j48FactorRoc, type = "s", print.thres = c(.5), 
     print.thres.pch = 16, print.thres.pattern = "", 
     print.thres.cex = 1.2, legacy.axes = TRUE)
plot(j48Roc, type = "s", print.thres = c(.5), 
     print.thres.pch = 3, print.thres.pattern = "", 
     print.thres.cex = 1.2, legacy.axes = TRUE,
     add = TRUE, col = "red", print.thres.col = "red")
legend(.75, .2,
       c("Grouped Categories", "Independent Categories"),
       lwd = c(1, 1),
       col = c("black", "red"),
       pch = c(16, 3))

plot(rpartFactorRoc, type = "s", add = TRUE, 
     col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)

################################################################################
### Section 14.2 Rule-Based Models

set.seed(476)
partFit <- train(x = training[,fullSet], 
                 y = training$Class,
                 method = "PART",
                 metric = "ROC",
                 trControl = ctrl)
partFit

part2008 <- merge(partFit$pred,  partFit$bestTune)
partCM <- confusionMatrix(partFit, norm = "none")
partCM

partRoc <- roc(response = partFit$pred$obs,
               predictor = partFit$pred$successful,
               levels = rev(levels(partFit$pred$obs)))
partRoc

set.seed(476)
partFactorFit <- train(training[,factorPredictors], training$Class,
                       method = "PART",
                       metric = "ROC",
                       trControl = ctrl)
partFactorFit

partFactor2008 <- merge(partFactorFit$pred,  partFactorFit$bestTune)
partFactorCM <- confusionMatrix(partFactorFit, norm = "none")
partFactorCM

partFactorRoc <- roc(response = partFactorFit$pred$obs,
                     predictor = partFactorFit$pred$successful,
                     levels = rev(levels(partFactorFit$pred$obs)))
partFactorRoc

################################################################################
### Section 14.3 Bagged Trees

set.seed(476)
treebagFit <- train(x = training[,fullSet], 
                    y = training$Class,
                    method = "treebag",
                    nbagg = 50,
                    metric = "ROC",
                    trControl = ctrl)
treebagFit

treebag2008 <- merge(treebagFit$pred,  treebagFit$bestTune)
treebagCM <- confusionMatrix(treebagFit, norm = "none")
treebagCM

treebagRoc <- roc(response = treebagFit$pred$obs,
                  predictor = treebagFit$pred$successful,
                  levels = rev(levels(treebagFit$pred$obs)))
set.seed(476)
treebagFactorFit <- train(x = training[,factorPredictors], 
                          y = training$Class,
                          method = "treebag",
                          nbagg = 50,
                          metric = "ROC",
                          trControl = ctrl)
treebagFactorFit

treebagFactor2008 <- merge(treebagFactorFit$pred,  treebagFactorFit$bestTune)
treebagFactorCM <- confusionMatrix(treebagFactorFit, norm = "none")
treebagFactorCM
treebagFactorRoc <- roc(response = treebagFactorFit$pred$obs,
                        predictor = treebagFactorFit$pred$successful,
                        levels = rev(levels(treebagFactorFit$pred$obs)))


plot(rpartRoc, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(j48FactorRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), 
     legacy.axes = TRUE)
plot(treebagRoc, type = "s", add = TRUE, print.thres = c(.5), 
     print.thres.pch = 3, legacy.axes = TRUE, print.thres.pattern = "", 
     print.thres.cex = 1.2,
     col = "red", print.thres.col = "red")
plot(treebagFactorRoc, type = "s", add = TRUE, print.thres = c(.5), 
     print.thres.pch = 16, print.thres.pattern = "", legacy.axes = TRUE, 
     print.thres.cex = 1.2)
legend(.75, .2,
       c("Grouped Categories", "Independent Categories"),
       lwd = c(1, 1),
       col = c("black", "red"),
       pch = c(16, 3))

################################################################################
### Section 14.4 Random Forests

### For the book, this model was run with only 500 trees (by
### accident). More than 1000 trees usually required to get consistent
### results.

mtryValues <- c(5, 10, 20, 32, 50, 100, 250, 500, 1000)
set.seed(476)
rfFit <- train(x = training[,fullSet], 
               y = training$Class,
               method = "rf",
               ntree = 500,
               tuneGrid = data.frame(mtry = mtryValues),
               importance = TRUE,
               metric = "ROC",
               trControl = ctrl)
rfFit

rf2008 <- merge(rfFit$pred,  rfFit$bestTune)
rfCM <- confusionMatrix(rfFit, norm = "none")
rfCM

rfRoc <- roc(response = rfFit$pred$obs,
             predictor = rfFit$pred$successful,
             levels = rev(levels(rfFit$pred$obs)))

gc()

## The randomForest package cannot handle factors with more than 32
## levels, so we make a new set of predictors where the sponsor code
## factor is entered as dummy variables instead of a single factor. 

sponsorVars <- grep("Sponsor", names(training), value = TRUE)
sponsorVars <- sponsorVars[sponsorVars != "SponsorCode"]

rfPredictors <- factorPredictors
rfPredictors <- rfPredictors[rfPredictors != "SponsorCode"]
rfPredictors <- c(rfPredictors, sponsorVars)

set.seed(476)
rfFactorFit <- train(x = training[,rfPredictors], 
                     y = training$Class,
                     method = "rf",
                     ntree = 1500,
                     tuneGrid = data.frame(mtry = mtryValues),
                     importance = TRUE,
                     metric = "ROC",
                     trControl = ctrl)
rfFactorFit

rfFactor2008 <- merge(rfFactorFit$pred,  rfFactorFit$bestTune)
rfFactorCM <- confusionMatrix(rfFactorFit, norm = "none")
rfFactorCM

rfFactorRoc <- roc(response = rfFactorFit$pred$obs,
                   predictor = rfFactorFit$pred$successful,
                   levels = rev(levels(rfFactorFit$pred$obs)))

plot(treebagRoc, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(rpartRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(j48FactorRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), 
     legacy.axes = TRUE)
plot(rfRoc, type = "s", add = TRUE, print.thres = c(.5), 
     print.thres.pch = 3, legacy.axes = TRUE, print.thres.pattern = "", 
     print.thres.cex = 1.2,
     col = "red", print.thres.col = "red")
plot(rfFactorRoc, type = "s", add = TRUE, print.thres = c(.5), 
     print.thres.pch = 16, print.thres.pattern = "", legacy.axes = TRUE, 
     print.thres.cex = 1.2)
legend(.75, .2,
       c("Grouped Categories", "Independent Categories"),
       lwd = c(1, 1),
       col = c("black", "red"),
       pch = c(16, 3))


################################################################################
### Section 14.5 Boosting

gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (1:20)*100,
                       shrinkage = c(.01, .1))

set.seed(476)
gbmFit <- train(x = training[,fullSet], 
                y = training$Class,
                method = "gbm",
                tuneGrid = gbmGrid,
                metric = "ROC",
                verbose = FALSE,
                trControl = ctrl)
gbmFit

gbmFit$pred <- merge(gbmFit$pred,  gbmFit$bestTune)
gbmCM <- confusionMatrix(gbmFit, norm = "none")
gbmCM

gbmRoc <- roc(response = gbmFit$pred$obs,
              predictor = gbmFit$pred$successful,
              levels = rev(levels(gbmFit$pred$obs)))

set.seed(476)
gbmFactorFit <- train(x = training[,factorPredictors], 
                      y = training$Class,
                      method = "gbm",
                      tuneGrid = gbmGrid,
                      verbose = FALSE,
                      metric = "ROC",
                      trControl = ctrl)
gbmFactorFit

gbmFactorFit$pred <- merge(gbmFactorFit$pred,  gbmFactorFit$bestTune)
gbmFactorCM <- confusionMatrix(gbmFactorFit, norm = "none")
gbmFactorCM

gbmFactorRoc <- roc(response = gbmFactorFit$pred$obs,
                    predictor = gbmFactorFit$pred$successful,
                    levels = rev(levels(gbmFactorFit$pred$obs)))

gbmROCRange <- extendrange(cbind(gbmFactorFit$results$ROC,gbmFit$results$ROC))

plot(gbmFactorFit, ylim = gbmROCRange, 
     auto.key = list(columns = 4, lines = TRUE))


plot(gbmFit, ylim = gbmROCRange, 
     auto.key = list(columns = 4, lines = TRUE))


plot(treebagRoc, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(rpartRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(j48FactorRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(rfFactorRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(gbmRoc, type = "s", print.thres = c(.5), print.thres.pch = 3, 
     print.thres.pattern = "", print.thres.cex = 1.2,
     add = TRUE, col = "red", print.thres.col = "red", legacy.axes = TRUE)
plot(gbmFactorRoc, type = "s", print.thres = c(.5), print.thres.pch = 16, 
     legacy.axes = TRUE, print.thres.pattern = "", print.thres.cex = 1.2,
     add = TRUE)
legend(.75, .2,
       c("Grouped Categories", "Independent Categories"),
       lwd = c(1, 1),
       col = c("black", "red"),
       pch = c(16, 3))

################################################################################
### Section 14.5 C5.0

c50Grid <- expand.grid(trials = c(1:9, (1:10)*10),
                       model = c("tree", "rules"),
                       winnow = c(TRUE, FALSE))
set.seed(476)
c50FactorFit <- train(training[,factorPredictors], training$Class,
                      method = "C5.0",
                      tuneGrid = c50Grid,
                      verbose = FALSE,
                      metric = "ROC",
                      trControl = ctrl)
c50FactorFit

c50FactorFit$pred <- merge(c50FactorFit$pred,  c50FactorFit$bestTune)
c50FactorCM <- confusionMatrix(c50FactorFit, norm = "none")
c50FactorCM

c50FactorRoc <- roc(response = c50FactorFit$pred$obs,
                    predictor = c50FactorFit$pred$successful,
                    levels = rev(levels(c50FactorFit$pred$obs)))

set.seed(476)
c50Fit <- train(training[,fullSet], training$Class,
                method = "C5.0",
                tuneGrid = c50Grid,
                metric = "ROC",
                verbose = FALSE,
                trControl = ctrl)
c50Fit

c50Fit$pred <- merge(c50Fit$pred,  c50Fit$bestTune)
c50CM <- confusionMatrix(c50Fit, norm = "none")
c50CM

c50Roc <- roc(response = c50Fit$pred$obs,
              predictor = c50Fit$pred$successful,
              levels = rev(levels(c50Fit$pred$obs)))

update(plot(c50FactorFit), ylab = "ROC AUC (2008 Hold-Out Data)")


plot(treebagRoc, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(rpartRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(j48FactorRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(rfFactorRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(gbmRoc, type = "s",  col = rgb(.2, .2, .2, .2), add = TRUE, legacy.axes = TRUE)
plot(c50Roc, type = "s", print.thres = c(.5), print.thres.pch = 3, 
     print.thres.pattern = "", print.thres.cex = 1.2,
     add = TRUE, col = "red", print.thres.col = "red", legacy.axes = TRUE)
plot(c50FactorRoc, type = "s", print.thres = c(.5), print.thres.pch = 16, 
     print.thres.pattern = "", print.thres.cex = 1.2,
     add = TRUE, legacy.axes = TRUE)
legend(.75, .2,
       c("Grouped Categories", "Independent Categories"),
       lwd = c(1, 1),
       col = c("black", "red"),
       pch = c(16, 3))

################################################################################
### Section 14.7 Comparing Two Encodings of Categorical Predictors

## Pull the hold-out results from each model and merge

rp1 <- caret:::getTrainPerf(rpartFit)
names(rp1) <- gsub("Train", "Independent", names(rp1))
rp2 <- caret:::getTrainPerf(rpartFactorFit)
rp2$Label <- "CART"
names(rp2) <- gsub("Train", "Grouped", names(rp2))
rp <- cbind(rp1, rp2)

j481 <- caret:::getTrainPerf(j48Fit)
names(j481) <- gsub("Train", "Independent", names(j481))
j482 <- caret:::getTrainPerf(j48FactorFit)
j482$Label <- "J48"
names(j482) <- gsub("Train", "Grouped", names(j482))
j48 <- cbind(j481, j482)

part1 <- caret:::getTrainPerf(partFit)
names(part1) <- gsub("Train", "Independent", names(part1))
part2 <- caret:::getTrainPerf(partFactorFit)
part2$Label <- "PART"
names(part2) <- gsub("Train", "Grouped", names(part2))
part <- cbind(part1, part2)

tb1 <- caret:::getTrainPerf(treebagFit)
names(tb1) <- gsub("Train", "Independent", names(tb1))
tb2 <- caret:::getTrainPerf(treebagFactorFit)
tb2$Label <- "Bagged Tree"
names(tb2) <- gsub("Train", "Grouped", names(tb2))
tb <- cbind(tb1, tb2)

rf1 <- caret:::getTrainPerf(rfFit)
names(rf1) <- gsub("Train", "Independent", names(rf1))
rf2 <- caret:::getTrainPerf(rfFactorFit)
rf2$Label <- "Random Forest"
names(rf2) <- gsub("Train", "Grouped", names(rf2))
rf <- cbind(rf1, rf2)

gbm1 <- caret:::getTrainPerf(gbmFit)
names(gbm1) <- gsub("Train", "Independent", names(gbm1))
gbm2 <- caret:::getTrainPerf(gbmFactorFit)
gbm2$Label <- "Boosted Tree"
names(gbm2) <- gsub("Train", "Grouped", names(gbm2))
bst <- cbind(gbm1, gbm2)


c501 <- caret:::getTrainPerf(c50Fit)
names(c501) <- gsub("Train", "Independent", names(c501))
c502 <- caret:::getTrainPerf(c50FactorFit)
c502$Label <- "C5.0"
names(c502) <- gsub("Train", "Grouped", names(c502))
c5 <- cbind(c501, c502)


trainPerf <- rbind(rp, j48, part, tb, rf, bst, c5)

library(lattice)
library(reshape2)
trainPerf <- melt(trainPerf)
trainPerf$metric <- "ROC"
trainPerf$metric[grepl("Sens", trainPerf$variable)] <- "Sensitivity"
trainPerf$metric[grepl("Spec", trainPerf$variable)] <- "Specificity"
trainPerf$model <- "Grouped"
trainPerf$model[grepl("Independent", trainPerf$variable)] <- "Independent"

trainPerf <- melt(trainPerf)
trainPerf$metric <- "ROC"
trainPerf$metric[grepl("Sens", trainPerf$variable)] <- "Sensitivity"
trainPerf$metric[grepl("Spec", trainPerf$variable)] <- "Specificity"
trainPerf$model <- "Independent"
trainPerf$model[grepl("Grouped", trainPerf$variable)] <- "Grouped"
trainPerf$Label <- factor(trainPerf$Label,
                          levels = rev(c("CART", "Cond. Trees", "J48", "Ripper",
                                         "PART", "Bagged Tree", "Random Forest", 
                                         "Boosted Tree", "C5.0")))

dotplot(Label ~ value|metric,
        data = trainPerf,
        groups = model,
        horizontal = TRUE,
        auto.key = list(columns = 2),
        between = list(x = 1),
        xlab = "")


sessionInfo()

q("no")
