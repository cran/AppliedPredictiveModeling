
easyBoundaryFunc <- function(n, intercept = 0, interaction = 2)
{
   require(MASS)
   sigma <- matrix(c(2,1.3,1.3,2),2,2)
   
   tmpData <- data.frame(mvrnorm(n=n, c(0,0), sigma))
   xSeq <- seq(-4, 4, length=40)
   plotGrid <- expand.grid(x = xSeq, y = xSeq)
   zFoo <- function(x, y) intercept -4 * x + 4* y + interaction*x*y
   z2p <- function(x) 1/(1+exp(-x))
 
   tmpData$prob <- z2p(zFoo(tmpData$X1, tmpData$X2))
   tmpData$class <- factor(ifelse(runif(length(tmpData$prob)) <= tmpData$prob, "Class1", "Class2"))
   tmpData
}
