\name{schedulingData}
\docType{data}
\alias{schedulingData}
\title{HPC Job Scheduling Data}
\description{
These data consist of information on 4331 jobs in a high performance computing environment. Seven attributes were recorded for each job along with a discrete class describing the execution time. 

The predictors are: \code{Protocol} (the type of computation), \code{Compounds} (the number of data points for each jobs), \code{InputFields} (the number of characteristic being estimated), \code{Iterations} (maximum number of iterations for the computations), \code{NumPending} (the number of other jobs pending at the time of launch), \code{Hour} (decimal hour of day for launch time) and \code{Day} (of launch time).

The classes are: \code{VF} (very fast), \code{F} (fast), \code{M} (moderate) and \code{L} (long).
}

\usage{data(schedulingData)}

\value{
  \item{schedulingData}{a data frame with 4331 rows and 8 columns}
  }

\examples{
data(schedulingData)

library(caret)

set.seed(1104)
inTrain <- createDataPartition(schedulingData$Class, p = .8, list = FALSE)

schedulingData$NumPending <- schedulingData$NumPending + 1

trainData <- schedulingData[ inTrain,]
testData  <- schedulingData[-inTrain,]

modForm <- as.formula(Class ~ Protocol + log10(Compounds) +
                      log10(InputFields)+ log10(Iterations) +
                      log10(NumPending) + Hour + Day)


}

\keyword{datasets}  

