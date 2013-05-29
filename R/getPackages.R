

getPackages <- function(chapter, ...)
{
  if(is.numeric(chapter)) chapter <- paste(chapter)
  pkg <- list()
  pkg[["2"]] <- c("earth", "caret", "lattice")
  pkg[["3"]] <- c("e1071", "caret", "corrplot")
  pkg[["4"]] <- c("kernlab", "caret")
  pkg[["6"]] <- c("lattice", "corrplot", "pls", "elasticnet")
  pkg[["7"]] <- c("caret", "earth", "kernlab","lattice", "nnet")  
  pkg[["8"]] <- c("caret", "Cubist", "gbm", "lattice", "party", "partykit", 
                  "randomForest", "rpart",  "RWeka") 
  pkg[["10"]] <- c("caret", "Cubist", "earth", "elasticnet", "gbm", "ipred", 
                   "lattice", "nnet", "party","pls", "randomForests", "rpart", 
                   "RWeka")  
  pkg[["11"]] <- c("caret", "MASS", "randomForest", "pROC", "klaR")  
  pkg[["12"]] <- c("caret", "glmnet", "lattice", 
                   "MASS", "pamr", "pls", "pROC", "sparseLDA")  
  pkg[["13"]] <- c("caret", "kernlab", "klaR",  "lattice", "latticeExtra", 
                   "MASS", "mda", "nnet", "pROC")  
  pkg[["14"]] <- c("C50", "caret", "gbm",  "lattice", "partykit", "pROC", 
                   "randomForest", "reshape2", 
                   "rpart", "RWeka")
  pkg[["16"]] <- c("caret", "C50", "earth", "DMwR", "DWD", " kernlab", "mda", 
                   "pROC", "randomForest", "rpart") 
  pkg[["17"]] <- c("C50", "caret", "earth", "Hmisc", "ipred", "tabplot", 
                   "kernlab", "lattice", "MASS", "mda", "nnet", "pls", 
                   "randomForest", "rpart", "sparseLDA")
  pkg[["18"]] <- c("caret", "CORElearn", "corrplot", "pROC", "minerva")
  pkg[["19"]] <- c("caret", "MASS", "corrplot", "RColorBrewer", "randomForest", 
                   "kernlab", "klaR")
  plist <- paste(paste("'", names(pkg), "'", sep = ""), collapse = ", ")
  if(!any(chapter %in% names(pkg))) stop(paste("'chapter' must be: ",
                                               paste(plist, collapse = ", ")),
                                         sep = "")  
  
  
  pkg <- unlist(pkg[chapter])
  pkg <- pkg[!is.na(pkg)]
  pkg <- pkg[pkg != ""]
  pkg <- pkg[order(tolower(pkg))]

  install.packages(pkg, ...)
}
