#' Functions for Simulating Data
#' 
#' These functions simulate data that are used in the text.
#' 
#' The \code{quadBoundaryFunc} function creates a class boundary that is a
#' function of both predictors. The probability values are based on a logistic
#' regression model with model equation: \eqn{-1-2X_1 -0.2X_1^2 +
#' 2X_2^2}{-1-2*X1 -0.2*X1^2 + 2*X2^2}. The predictors here are multivariate
#' normal with mean (1, 0) and a moderate degree of positive correlation.
#' 
#' Similarly, the \code{easyBoundaryFunc} uses a logistic regression model with
#' model equation: \eqn{intercept -4X_1 + 4X_2 + interaction \times X_1 \times
#' X_2}{intercept -4*X1 + 4*X2 + interaction*X1*X2}. The predictors here are
#' multivariate normal with mean (1, 0) and a strong positive correlation.
#' 
#' @aliases quadBoundaryFunc easyBoundaryFunc
#' @param n the sample size
#' @param intercept the coefficient for the logistic regression intercept term
#' @param interaction the coefficient for the logistic regression interaction
#' term
#' @return Both functions return data frames with columns \item{X1}{numeric
#' predictor value} \item{X2}{numeric predictor value} \item{prob }{numeric
#' value reflecting the true probability of the first class} \item{class }{a
#' factor variable with levels 'Class1' and 'Class2'}
#' @author Max Kuhn
#' @keywords utilities
#' @examples
#' 
#' ## in Chapter 11, 'Measuring Performance in Classification Model'
#' set.seed(975)
#' training <- quadBoundaryFunc(500)
#' testing <- quadBoundaryFunc(1000)
#'  
#' 
#' ## in Chapter 20, 'Factors That Can Affect Model Performance'
#' set.seed(615)
#' dat <- easyBoundaryFunc(200, interaction = 3, intercept = 3)
#' dat$X1 <- scale(dat$X1)
#' dat$X2 <- scale(dat$X2)
#' dat$Data <- "Original"
#' dat$prob <- NULL
#' 
#' ## in Chapter X, 'An Introduction to Feature Selection'
#' 
#' set.seed(874)
#' reliefEx3 <- easyBoundaryFunc(500)
#' reliefEx3$X1 <- scale(reliefEx3$X1)
#' reliefEx3$X2 <- scale(reliefEx3$X2)
#' reliefEx3$prob <- NULL
#' 
#' 
#' @export quadBoundaryFunc
quadBoundaryFunc <- function(n) {
  sigma <- matrix(c(1, .7, .7, 2), 2, 2)
  
  tmpData <- data.frame(MASS::mvrnorm(n = n, c(1, 0), sigma))
  xSeq <- seq(-4, 4, length = 40)
  plotGrid <- expand.grid(x = xSeq, y = xSeq)
  zFoo <- function(x, y)
    - 1 - 2 * x - 0 * y - .2 * x ^ 2 + 2 * y ^ 2
  z2p <- function(x)
    1 / (1 + exp(-x))
  
  tmpData$prob <- z2p(zFoo(tmpData$X1, tmpData$X2))
  tmpData$class <-
    factor(ifelse(stats::runif(length(tmpData$prob)) <= tmpData$prob, "Class1", "Class2"))
  tmpData
}


#' @rdname quadBoundaryFunc
#' @export 
easyBoundaryFunc <- function(n,
                             intercept = 0,
                             interaction = 2) {
  sigma <- matrix(c(2, 1.3, 1.3, 2), 2, 2)
  
  tmpData <- data.frame(MASS::mvrnorm(n = n, c(0, 0), sigma))
  xSeq <- seq(-4, 4, length = 40)
  plotGrid <- expand.grid(x = xSeq, y = xSeq)
  zFoo <- function(x, y)
    intercept - 4 * x + 4 * y + interaction * x * y
  z2p <- function(x)
    1 / (1 + exp(-x))
  
  tmpData$prob <- z2p(zFoo(tmpData$X1, tmpData$X2))
  tmpData$class <-
    factor(ifelse(stats::runif(length(tmpData$prob)) <= tmpData$prob, "Class1", "Class2"))
  tmpData
}
