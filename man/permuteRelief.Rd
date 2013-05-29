\name{permuteRelief}
\alias{permuteRelief}
\title{
Permutation Statistics for the Relief Algorithm
}
\description{
This function uses a permutation approach to determining the relative magnitude of Relief scores (Kira and Rendell, 1992 and Kononenko, 1994).
}
\usage{
permuteRelief(x, y, nperm = 100, ...)
}
\arguments{
  \item{x}{
a data frame of predictor data
}
  \item{y}{
a vector of outcomes
}
  \item{nperm}{
the number of random permutations of the data
}
  \item{\dots}{
options to pass to \code{\link[CORElearn]{attrEval}}, such as the exact Relief algorithm, to use
}
}
\details{
The scores for each predictor are computed using the original data and after outcome data are randomly scrambled (\code{nprem} times). The mean and standard deviation of the permuted values are determined and a standardized version of the observed scores are determined by subtracting the permuted means from the original values, then dividing each by the corresponding standard deviation. 
}
\value{
a list with elements
\item{standardized }{a vector of standardized predictor scores}
\item{permutations }{the values of the permuted scores, for plotting to assess the permutation distribution}
\item{observed}{the observed scores}
\item{options}{a list of options passed using \ldots}
}
\references{
Kira, K., & Rendell, L. (1992). The feature selection problem: Traditional methods and a new algorithm. \emph{Proceedings of the Eleventh International Conference on Machine Learning}, 129-129.

Kononenko, I. (1994). Estimating attributes: analysis and extensions of RELIEF. Machine Learning: ECML-94, 171-182.
}
\author{
Max Kuhn
}


\seealso{
 \code{\link[CORElearn]{attrEval}}
}
\examples{
set.seed(874)
reliefEx3 <- easyBoundaryFunc(500)
reliefEx3$X1 <- scale(reliefEx3$X1)
reliefEx3$X2 <- scale(reliefEx3$X2)
reliefEx3$prob <- NULL

standardized <- permuteRelief(reliefEx3[, 1:2], reliefEx3$class, 
                              ## For efficiency, a small number of
                              ## permutations are used here.
                              nperm = 50,           
                              estimator="ReliefFequalK", 
                              ReliefIterations= 50)

}

\keyword{htest}

