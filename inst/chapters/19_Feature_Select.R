################################################################################
### R code from Applied Predictive Modeling (2013) by Kuhn and Johnson.
### Copyright 2013 Kuhn and Johnson
### Web Page: http://www.appliedpredictivemodeling.com
### Contact: Max Kuhn (mxkuhn@gmail.com) 
###
### Chapter 19: An Introduction to Feature Selection
###
### Required packages: AppliedPredictiveModeling, caret, MASS, corrplot,
###                    RColorBrewer, randomForest, kernlab, klaR,
###                   
###
### Data used: The Alzheimer disease data from the AppliedPredictiveModeling 
###            package
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
### Section 19.6 Case Study: Predicting Cognitive Impairment


library(AppliedPredictiveModeling)
data(AlzheimerDisease)

## The baseline set of predictors
bl <- c("Genotype", "age", "tau", "p_tau", "Ab_42", "male")

## The set of new assays
newAssays <- colnames(predictors)
newAssays <- newAssays[!(newAssays %in% c("Class", bl))]

## Decompose the genotype factor into binary dummy variables

predictors$E2 <- predictors$E3 <- predictors$E4 <- 0
predictors$E2[grepl("2", predictors$Genotype)] <- 1
predictors$E3[grepl("3", predictors$Genotype)] <- 1
predictors$E4[grepl("4", predictors$Genotype)] <- 1
genotype <-  predictors$Genotype

## Partition the data
library(caret)
set.seed(730)
split <- createDataPartition(diagnosis, p = .8, list = FALSE)

adData <- predictors
adData$Class <- diagnosis

training <- adData[ split, ]
testing  <- adData[-split, ]

predVars <- names(adData)[!(names(adData) %in% c("Class",  "Genotype"))]

## This summary function is used to evaluate the models.
fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))

## We create the cross-validation files as a list to use with different 
## functions

set.seed(104)
index <- createMultiFolds(training$Class, times = 5)

## The candidate set of the number of predictors to evaluate
varSeq <- seq(1, length(predVars)-1, by = 2)

## We can also use parallel processing to run each resampled RFE
## iteration (or resampled model with train()) using different
## workers.

library(doMC)
registerDoMC(15)


## The rfe() function in the caret package is used for recursive feature 
## elimiation. We setup control functions for this and train() that use
## the same cross-validation folds. The 'ctrl' object will be modifed several
## times as we try different models

ctrl <- rfeControl(method = "repeatedcv", repeats = 5,
                   saveDetails = TRUE,
                   index = index,
                   returnResamp = "final")

fullCtrl <- trainControl(method = "repeatedcv",
                         repeats = 5,
                         summaryFunction = fiveStats,
                         classProbs = TRUE,
                         index = index)

## The correlation matrix of the new data
predCor <- cor(training[, newAssays])

library(RColorBrewer)
cols <- c(rev(brewer.pal(7, "Blues")),
          brewer.pal(7, "Reds"))
library(corrplot)
corrplot(predCor,
         order = "hclust",
         tl.pos = "n",addgrid.col = rgb(1,1,1,.01),
         col = colorRampPalette(cols)(51))

## Fit a series of models with the full set of predictors
set.seed(721)
rfFull <- train(training[, predVars],
                training$Class,
                method = "rf",
                metric = "ROC",
                tuneGrid = data.frame(mtry = floor(sqrt(length(predVars)))),
                ntree = 1000,
                trControl = fullCtrl)
rfFull

set.seed(721)
ldaFull <- train(training[, predVars],
                 training$Class,
                 method = "lda",
                 metric = "ROC",
                 ## The 'tol' argument helps lda() know when a matrix is 
                 ## singular. One of the predictors has values very close to 
                 ## zero, so we raise the vaue to be smaller than the default
                 ## value of 1.0e-4.
                 tol = 1.0e-12,
                 trControl = fullCtrl)
ldaFull

set.seed(721)
svmFull <- train(training[, predVars],
                 training$Class,
                 method = "svmRadial",
                 metric = "ROC",
                 tuneLength = 12,
                 preProc = c("center", "scale"),
                 trControl = fullCtrl)
svmFull

set.seed(721)
nbFull <- train(training[, predVars],
                training$Class,
                method = "nb",
                metric = "ROC",
                trControl = fullCtrl)
nbFull

lrFull <- train(training[, predVars],
                training$Class,
                method = "glm",
                metric = "ROC",
                trControl = fullCtrl)
lrFull

set.seed(721)
knnFull <- train(training[, predVars],
                 training$Class,
                 method = "knn",
                 metric = "ROC",
                 tuneLength = 20,
                 preProc = c("center", "scale"),
                 trControl = fullCtrl)
knnFull

## Now fit the RFE versions. To do this, the 'functions' argument of the rfe()
## object is modified to the approproate functions. For model details about 
## these functions and their arguments, see 
##
##   http://caret.r-forge.r-project.org/featureSelection.html
##
## for more information.




ctrl$functions <- rfFuncs
ctrl$functions$summary <- fiveStats
set.seed(721)
rfRFE <- rfe(training[, predVars],
             training$Class,
             sizes = varSeq,
             metric = "ROC",
             ntree = 1000,
             rfeControl = ctrl)
rfRFE

ctrl$functions <- ldaFuncs
ctrl$functions$summary <- fiveStats

set.seed(721)
ldaRFE <- rfe(training[, predVars],
              training$Class,
              sizes = varSeq,
              metric = "ROC",
              tol = 1.0e-12,
              rfeControl = ctrl)
ldaRFE

ctrl$functions <- nbFuncs
ctrl$functions$summary <- fiveStats
set.seed(721)
nbRFE <- rfe(training[, predVars],
              training$Class,
              sizes = varSeq,
              metric = "ROC",
              rfeControl = ctrl)
nbRFE

## Here, the caretFuncs list allows for a model to be tuned at each iteration 
## of feature seleciton.

ctrl$functions <- caretFuncs
ctrl$functions$summary <- fiveStats

## This options tells train() to run it's model tuning
## sequentially. Otherwise, there would be parallel processing at two
## levels, which is possible but requires W^2 workers. On our machine,
## it was more efficient to only run the RFE process in parallel. 

cvCtrl <- trainControl(method = "cv",
                       verboseIter = FALSE,
                       classProbs = TRUE,
                       allowParallel = FALSE)

set.seed(721)
svmRFE <- rfe(training[, predVars],
              training$Class,
              sizes = varSeq,
              rfeControl = ctrl,
              metric = "ROC",
              ## Now arguments to train() are used.
              method = "svmRadial",
              tuneLength = 12,
              preProc = c("center", "scale"),
              trControl = cvCtrl)
svmRFE

ctrl$functions <- lrFuncs
ctrl$functions$summary <- fiveStats

set.seed(721)
lrRFE <- rfe(training[, predVars],
               training$Class,
               sizes = varSeq,
               metric = "ROC",
               rfeControl = ctrl)
lrRFE

ctrl$functions <- caretFuncs
ctrl$functions$summary <- fiveStats

set.seed(721)
knnRFE <- rfe(training[, predVars],
              training$Class,
              sizes = varSeq,
              metric = "ROC",
              method = "knn",
              tuneLength = 20,
              preProc = c("center", "scale"),
              trControl = cvCtrl,
              rfeControl = ctrl)
knnRFE

## Each of these models can be evaluate using the plot() function to see
## the profile across subset sizes.

## Test set ROC results:
rfROCfull <- roc(testing$Class,
                 predict(rfFull, testing[,predVars], type = "prob")[,1])
rfROCfull
rfROCrfe <- roc(testing$Class,
                predict(rfRFE, testing[,predVars])$Impaired)
rfROCrfe

ldaROCfull <- roc(testing$Class,
                  predict(ldaFull, testing[,predVars], type = "prob")[,1])
ldaROCfull
ldaROCrfe <- roc(testing$Class,
                 predict(ldaRFE, testing[,predVars])$Impaired)
ldaROCrfe

nbROCfull <- roc(testing$Class,
                  predict(nbFull, testing[,predVars], type = "prob")[,1])
nbROCfull
nbROCrfe <- roc(testing$Class,
                 predict(nbRFE, testing[,predVars])$Impaired)
nbROCrfe

svmROCfull <- roc(testing$Class,
                  predict(svmFull, testing[,predVars], type = "prob")[,1])
svmROCfull
svmROCrfe <- roc(testing$Class,
                 predict(svmRFE, testing[,predVars])$Impaired)
svmROCrfe

lrROCfull <- roc(testing$Class,
                  predict(lrFull, testing[,predVars], type = "prob")[,1])
lrROCfull
lrROCrfe <- roc(testing$Class,
                 predict(lrRFE, testing[,predVars])$Impaired)
lrROCrfe

knnROCfull <- roc(testing$Class,
                  predict(knnFull, testing[,predVars], type = "prob")[,1])
knnROCfull
knnROCrfe <- roc(testing$Class,
                 predict(knnRFE, testing[,predVars])$Impaired)
knnROCrfe


## For filter methods, the sbf() function (named for Selection By Filter) is
## used. It has similar arguments to rfe() to control the model fitting and
## filtering methods. 

## P-values are created for filtering. 

## A set of four LDA models are fit based on two factors: p-value adjustment 
## using a Bonferroni adjustment and whether the predictors should be 
## pre-screened for high correlations. 

sbfResamp <- function(x, fun = mean)
{
  x <- unlist(lapply(x$variables, length))
  fun(x)
}
sbfROC <- function(mod) auc(roc(testing$Class, predict(mod, testing)$Impaired))

## This function calculates p-values using either a t-test (when the predictor
## has 2+ distinct values) or using Fisher's Exact Test otherwise.

pScore <- function(x, y)
  {
    numX <- length(unique(x))
    if(numX > 2)
      {
       out <- t.test(x ~ y)$p.value
      } else {
       out <- fisher.test(factor(x), y)$p.value
      }
    out
  }
ldaWithPvalues <- ldaSBF
ldaWithPvalues$score <- pScore
ldaWithPvalues$summary <- fiveStats

## Predictors are retained if their p-value is less than the completely 
## subjective cut-off of 0.05.

ldaWithPvalues$filter <- function (score, x, y)
{
  keepers <- score <= 0.05
  keepers
}

sbfCtrl <- sbfControl(method = "repeatedcv",
                      repeats = 5,
                      verbose = TRUE,
                      functions = ldaWithPvalues,
                      index = index)

rawCorr <- sbf(training[, predVars],
               training$Class,
               tol = 1.0e-12,
               sbfControl = sbfCtrl)
rawCorr

ldaWithPvalues$filter <- function (score, x, y)
{
  score <- p.adjust(score,  "bonferroni")
  keepers <- score <= 0.05
  keepers
}
sbfCtrl <- sbfControl(method = "repeatedcv",
                      repeats = 5,
                      verbose = TRUE,
                      functions = ldaWithPvalues,
                      index = index)

adjCorr <- sbf(training[, predVars],
               training$Class,
               tol = 1.0e-12,
               sbfControl = sbfCtrl)
adjCorr

ldaWithPvalues$filter <- function (score, x, y)
{
  keepers <- score <= 0.05
  corrMat <- cor(x[,keepers])
  tooHigh <- findCorrelation(corrMat, .75)
  if(length(tooHigh) > 0) keepers[tooHigh] <- FALSE
  keepers
}
sbfCtrl <- sbfControl(method = "repeatedcv",
                      repeats = 5,
                      verbose = TRUE,
                      functions = ldaWithPvalues,
                      index = index)

rawNoCorr <- sbf(training[, predVars],
                 training$Class,
                 tol = 1.0e-12,
                 sbfControl = sbfCtrl)
rawNoCorr

ldaWithPvalues$filter <- function (score, x, y)
{
  score <- p.adjust(score,  "bonferroni")
  keepers <- score <= 0.05
  corrMat <- cor(x[,keepers])
  tooHigh <- findCorrelation(corrMat, .75)
  if(length(tooHigh) > 0) keepers[tooHigh] <- FALSE
  keepers
}
sbfCtrl <- sbfControl(method = "repeatedcv",
                      repeats = 5,
                      verbose = TRUE,
                      functions = ldaWithPvalues,
                      index = index)

adjNoCorr <- sbf(training[, predVars],
                 training$Class,
                 tol = 1.0e-12,
                 sbfControl = sbfCtrl)
adjNoCorr

## Filter methods test set ROC results:

sbfROC(rawCorr)
sbfROC(rawNoCorr)
sbfROC(adjCorr)
sbfROC(adjNoCorr)

## Get the resampling results for all the models

rfeResamples <- resamples(list(RF = rfRFE,
                               "Logistic Reg." = lrRFE,
                               "SVM" = svmRFE,
                               "$K$--NN" = knnRFE,
                               "N. Bayes" = nbRFE,
                               "LDA" = ldaRFE))
summary(rfeResamples)

fullResamples <- resamples(list(RF = rfFull,
                                "Logistic Reg." = lrFull,
                                "SVM" = svmFull,
                                "$K$--NN" = knnFull,
                                "N. Bayes" = nbFull,
                                "LDA" = ldaFull))
summary(fullResamples)

filteredResamples <- resamples(list("No Adjustment, Corr Vars" = rawCorr,
                                    "No Adjustment, No Corr Vars" = rawNoCorr,
                                    "Bonferroni, Corr Vars" = adjCorr,
                                    "Bonferroni, No Corr Vars" = adjNoCorr))
summary(filteredResamples)

sessionInfo()


