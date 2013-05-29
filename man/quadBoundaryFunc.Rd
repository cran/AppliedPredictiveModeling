\name{quadBoundaryFunc}
\alias{quadBoundaryFunc}
\alias{easyBoundaryFunc}

\title{
Functions for Simulating Data
}
\description{
These functions simulate data that are used in the text. 
}
\usage{
quadBoundaryFunc(n)

easyBoundaryFunc(n, intercept = 0, interaction = 2)
}
\arguments{
  \item{n}{the sample size}
  \item{intercept}{the coefficient for the logistic regression intercept term}
  \item{interaction}{the coefficient for the logistic regression interaction term}

}
\details{
The \code{quadBoundaryFunc} function creates a class boundary that is a function of both predictors. The probability values are based on a logistic regression model with model equation: \eqn{-1-2X_1 -0.2X_1^2 + 2X_2^2}{-1-2*X1 -0.2*X1^2 + 2*X2^2}. The predictors here are multivariate normal with mean (1, 0) and a moderate degree of positive correlation.

Similarly, the \code{easyBoundaryFunc} uses a logistic regression model with model equation: \eqn{intercept -4X_1 + 4X_2 + interaction \times X_1 \times X_2}{intercept -4*X1 + 4*X2 + interaction*X1*X2}. The predictors here are multivariate normal with mean (1, 0) and a strong positive correlation.
}
\value{
Both functions return data frames with columns
\item{X1}{numeric predictor value}
\item{X2}{numeric predictor value}
\item{prob }{numeric value reflecting the true probability of the first class}
\item{class }{a factor variable with levels 'Class1' and 'Class2'}
}

\author{
Max Kuhn
}

\examples{
## in Chapter 11, 'Measuring Performance in Classification Model'
set.seed(975)
training <- quadBoundaryFunc(500)
testing <- quadBoundaryFunc(1000)
 

## in Chapter 20, 'Factors That Can Affect Model Performance'
set.seed(615)
dat <- easyBoundaryFunc(200, interaction = 3, intercept = 3)
dat$X1 <- scale(dat$X1)
dat$X2 <- scale(dat$X2)
dat$Data <- "Original"
dat$prob <- NULL

## in Chapter X, 'An Introduction to Feature Selection'

set.seed(874)
reliefEx3 <- easyBoundaryFunc(500)
reliefEx3$X1 <- scale(reliefEx3$X1)
reliefEx3$X2 <- scale(reliefEx3$X2)
reliefEx3$prob <- NULL

}

\keyword{utilities}

