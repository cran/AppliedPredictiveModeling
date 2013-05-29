\name{twoClassData}
\docType{data}
\alias{twoClassData}
\alias{classes}
\title{Two Class Example Data}
\description{
These data contain two predictors measured for 208 samples. Of these, 111 samples are 
labeled as \code{Class1} and the remaining 97 are \code{Class2}.
}

\usage{data(twoClassData)}
\value{
  \item{predictors}{data frame of two predictors}
  \item{classes}{a factor vector of class labeled}  
  }

\examples{
data(twoClassData)

library(lattice)
xyplot(PredictorB ~ PredictorA, 
       data = predictors, 
       groups = classes, 
       auto.key = TRUE)

}

\keyword{datasets}  

