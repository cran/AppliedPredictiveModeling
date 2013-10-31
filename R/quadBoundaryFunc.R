quadBoundaryFunc <-
function(n)
{
   sigma <- matrix(c(1,.7,.7,2),2,2)
   
   tmpData <- data.frame(mvrnorm(n=n, c(1,0), sigma))
   xSeq <- seq(-4, 4, length=40)
   plotGrid <- expand.grid(x = xSeq, y = xSeq)
   zFoo <- function(x, y) -1 - 2 * x - 0 * y - .2 * x^2 + 2 * y^2
   z2p <- function(x) 1/(1+exp(-x))
 
   tmpData$prob <- z2p(zFoo(tmpData$X1, tmpData$X2))
   tmpData$class <- factor(ifelse(runif(length(tmpData$prob)) <= tmpData$prob, "Class1", "Class2"))
   tmpData
}
