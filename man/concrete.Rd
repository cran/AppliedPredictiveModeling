\name{concrete}
\docType{data}
\alias{concrete}
\alias{mixtures}
\title{Compressive Strength of Concrete from Yeh (1998)}
\description{
Yeh (1998) describes a collection of data sets from different sources that can be used for modeling the compressive strength of concrete formulations as a functions of their ingredients and age. 
}

\usage{data(concrete)}

\value{
  \item{concrete}{data frame of data with predictor columns \code{Cement}, \code{BlastFurnaceSlag}, \code{FlyAsh}, \code{Water}, \code{Superplasticizer}, \code{CoarseAggregate}, \code{FineAggregate} and \code{Age} with response column \code{CompressiveStrength}. These are the amounts.}
  \item{mixtures}{The same data where all the ingredients have been converted to proportions of the total amounts.}
  }

\details{
The data are from Yeh (1998) and taken from the UCI ML website \url{http://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength}.

There are 1030 data points from the UCI website, but the paper states that approximately 1,000 samples were made, but only 727 were analyzed in the source material. It is unclear which samples were excluded.
}

\source{Yeh, I. C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. \emph{Cement and Concrete Research}, 28(12), 1797-1808. Elsevier.
}

\examples{
data(concrete)

library(caret)

### Split used in the book:
set.seed(975)
inTrain <- createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training <- mixtures[ inTrain,]
testing  <- mixtures[-inTrain,]

}

\keyword{datasets}  

